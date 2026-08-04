"""Deterministic, authenticated chunks over a lossless document hierarchy."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from lexnlp.nlp.en.segments.hierarchy import (
    DocumentHierarchy,
    HierarchyManifest,
    Segment,
    SegmentKind,
    StructuralMode,
    StructuralSpan,
    StructureProfile,
    segment_document,
)

DEFAULT_MAX_CHARS = 4000
CHUNKING_SCHEMA_VERSION = 1
CHUNKING_SERIALIZER_VERSION = 1
TokenCounter: TypeAlias = Callable[[str], int]


class ContainerPolicy(str, Enum):
    PRESERVE = "preserve"
    PACK_SIBLINGS = "pack_siblings"


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _enum(value: object, enum_type: type[Enum], name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {choices}") from exc


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _digest_value(digest: "hashlib._Hash", value: object) -> None:
    """Frame supported scalar types without cross-type or sentinel collisions."""
    if value is None:
        type_tag = b"N"
        payload = b""
    elif isinstance(value, bool):
        type_tag = b"B"
        payload = b"1" if value else b"0"
    elif isinstance(value, int):
        type_tag = b"I"
        payload = str(value).encode("ascii")
    elif isinstance(value, str):
        type_tag = b"S"
        payload = value.encode("utf-8", "surrogatepass")
    else:
        raise TypeError(f"unsupported digest value type: {type(value).__name__}")
    digest.update(type_tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _attributes(value: object) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("attributes must be an iterable of string pairs") from exc
    seen: set[str] = set()
    for item in result:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each attribute must be a (name, value) tuple")
        name, attribute_value = item
        if not isinstance(name, str) or not isinstance(attribute_value, str):
            raise TypeError("attribute names and values must be strings")
        if name in seen:
            raise ValueError(f"duplicate attribute name {name!r}")
        seen.add(name)
    return result


@dataclass(frozen=True, slots=True)
class ChunkingManifest:
    source_sha256: str
    document_id: str | None
    unit_kind: str
    budget: int
    overlap: int
    respect_boundaries: bool
    container_policy: ContainerPolicy
    hierarchy_manifest: HierarchyManifest
    token_counter_id: str | None = None
    schema_version: int = CHUNKING_SCHEMA_VERSION
    serializer_version: int = CHUNKING_SERIALIZER_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if self.document_id is not None and (
            not isinstance(self.document_id, str) or not self.document_id
        ):
            raise ValueError("document_id must be None or a non-empty string")
        if self.unit_kind not in {"characters", "tokens"}:
            raise ValueError("unit_kind must be 'characters' or 'tokens'")
        _integer(self.budget, "budget", minimum=1)
        _integer(self.overlap, "overlap", minimum=0)
        if self.overlap >= self.budget:
            raise ValueError("overlap must be smaller than budget")
        if not isinstance(self.respect_boundaries, bool):
            raise TypeError("respect_boundaries must be a boolean")
        object.__setattr__(
            self,
            "container_policy",
            _enum(self.container_policy, ContainerPolicy, "container_policy"),
        )
        if not isinstance(self.hierarchy_manifest, HierarchyManifest):
            raise TypeError("hierarchy_manifest must be a HierarchyManifest")
        if self.hierarchy_manifest.tree_sha256 is None:
            raise ValueError("hierarchy_manifest must identify the realised tree")
        _integer(self.schema_version, "schema_version", minimum=1)
        _integer(self.serializer_version, "serializer_version", minimum=1)
        if self.unit_kind == "tokens":
            if (
                not isinstance(self.token_counter_id, str)
                or not self.token_counter_id.strip()
            ):
                raise ValueError("token mode requires a non-empty token_counter_id")
        elif self.token_counter_id is not None:
            raise ValueError("token_counter_id is only valid in token mode")

    @property
    def source_id(self) -> str:
        hashed = f"sha256:{self.source_sha256}"
        return f"{self.document_id}@{hashed}" if self.document_id is not None else hashed

    @property
    def canonical_json(self) -> str:
        hierarchy = self.hierarchy_manifest
        payload = {
            "budget": self.budget,
            "container_policy": self.container_policy.value,
            "document_id": self.document_id,
            "hierarchy_manifest": {
                "paragraph_backend_id": hierarchy.paragraph_backend_id,
                "schema_version": hierarchy.schema_version,
                "sentence_backend_id": hierarchy.sentence_backend_id,
                "structure_profile": hierarchy.structure_profile.value,
                "structural_detector_id": hierarchy.structural_detector_id,
                "structural_mode": hierarchy.structural_mode.value,
                "tree_sha256": hierarchy.tree_sha256,
            },
            "overlap": self.overlap,
            "respect_boundaries": self.respect_boundaries,
            "schema_version": self.schema_version,
            "serializer_version": self.serializer_version,
            "source_sha256": self.source_sha256,
            "token_counter_id": self.token_counter_id,
            "unit_kind": self.unit_kind,
        }
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def manifest_id(self) -> str:
        encoded = self.canonical_json.encode("ascii")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SegmentReference:
    segment_id: str
    kind: SegmentKind
    start: int
    end: int
    label: str | None = None
    level: int | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, SegmentKind, "kind"))
        _integer(self.start, "start", minimum=0)
        _integer(self.end, "end", minimum=0)
        if self.end <= self.start:
            raise ValueError("segment references must have positive length")
        expected = f"{self.kind.value}:{self.start}:{self.end}"
        if not isinstance(self.segment_id, str) or self.segment_id != expected:
            raise ValueError(f"segment_id must equal {expected!r}")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("label must be None or a string")
        if self.level is not None:
            _integer(self.level, "level", minimum=0)
        object.__setattr__(self, "attributes", _attributes(self.attributes))

    @classmethod
    def from_segment(cls, segment: Segment) -> "SegmentReference":
        if not isinstance(segment, Segment):
            raise TypeError("segment must be a Segment")
        return cls(
            segment.segment_id,
            segment.kind,
            segment.start,
            segment.end,
            segment.label,
            segment.level,
            segment.attributes,
        )


def _labels(
    segments: Sequence[SegmentReference],
    kind: SegmentKind,
) -> tuple[str, ...]:
    return tuple(
        reference.label
        for reference in segments
        if reference.kind is kind and reference.label is not None
    )


@dataclass(frozen=True, slots=True)
class ChunkProvenance:
    segments: tuple[SegmentReference, ...] = ()
    section_labels: tuple[str, ...] = ()
    clause_labels: tuple[str, ...] = ()
    list_labels: tuple[str, ...] = ()
    table_labels: tuple[str, ...] = ()
    segment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise TypeError("segments must be an iterable of SegmentReference") from exc
        if any(not isinstance(item, SegmentReference) for item in segments):
            raise TypeError("segments must contain only SegmentReference objects")
        identifiers = tuple(reference.segment_id for reference in segments)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("provenance segment references must be unique")

        derived = {
            "section_labels": _labels(segments, SegmentKind.SECTION),
            "clause_labels": _labels(segments, SegmentKind.CLAUSE),
            "list_labels": _labels(segments, SegmentKind.LIST_ITEM),
            "table_labels": _labels(segments, SegmentKind.TABLE),
            "segment_ids": identifiers,
        }
        object.__setattr__(self, "segments", segments)
        for name, expected in derived.items():
            supplied = tuple(getattr(self, name))
            if supplied and supplied != expected:
                raise ValueError(f"{name} must be derived exactly from segments")
            object.__setattr__(self, name, expected)


def compute_provenance_sha256(
    provenance: ChunkProvenance,
    content_provenance: ChunkProvenance,
    overlap_provenance: ChunkProvenance,
) -> str:
    partitions = (
        ("full", provenance),
        ("content", content_provenance),
        ("overlap", overlap_provenance),
    )
    if any(not isinstance(item, ChunkProvenance) for _, item in partitions):
        raise TypeError("all provenance partitions must be ChunkProvenance objects")
    digest = hashlib.sha256()
    digest.update(b"lexnlp.chunk-provenance.v1\0")
    for partition_name, partition in partitions:
        _digest_value(digest, partition_name)
        _digest_value(digest, len(partition.segments))
        for reference in partition.segments:
            for value in (
                reference.segment_id,
                reference.kind.value,
                reference.start,
                reference.end,
                reference.label,
                reference.level,
                len(reference.attributes),
            ):
                _digest_value(digest, value)
            for name, value in reference.attributes:
                _digest_value(digest, name)
                _digest_value(digest, value)
    return digest.hexdigest()


def compute_chunk_metadata_sha256(
    *,
    manifest: ChunkingManifest,
    index: int,
    start: int,
    end: int,
    new_content_start: int,
    unit_count: int,
    text_sha256: str,
    provenance_sha256: str,
) -> str:
    if not isinstance(manifest, ChunkingManifest):
        raise TypeError("manifest must be a ChunkingManifest")
    payload = {
        "end": end,
        "index": index,
        "manifest_id": manifest.manifest_id,
        "new_content_start": new_content_start,
        "provenance_sha256": provenance_sha256,
        "schema_version": 1,
        "start": start,
        "text_sha256": text_sha256,
        "unit_count": unit_count,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    index: int
    start: int
    end: int
    text: str
    text_sha256: str
    provenance: ChunkProvenance
    content_provenance: ChunkProvenance
    overlap_provenance: ChunkProvenance
    provenance_sha256: str
    chunk_metadata_sha256: str
    new_content_start: int
    unit_count: int
    manifest: ChunkingManifest

    def __post_init__(self) -> None:
        _integer(self.index, "index", minimum=0)
        _integer(self.start, "start", minimum=0)
        _integer(self.end, "end", minimum=0)
        _integer(self.new_content_start, "new_content_start", minimum=0)
        _integer(self.unit_count, "unit_count", minimum=0)
        if self.end <= self.start:
            raise ValueError("chunk must have positive length")
        if not self.start <= self.new_content_start < self.end:
            raise ValueError("new_content_start must lie inside the chunk")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if len(self.text) != self.end - self.start:
            raise ValueError("text length must equal the source span length")
        actual_text_digest = _digest(self.text)
        if self.text_sha256 != actual_text_digest:
            raise ValueError("text does not match text_sha256")
        for name in (
            "provenance",
            "content_provenance",
            "overlap_provenance",
        ):
            if not isinstance(getattr(self, name), ChunkProvenance):
                raise TypeError(f"{name} must be ChunkProvenance")
        actual_provenance_digest = compute_provenance_sha256(
            self.provenance,
            self.content_provenance,
            self.overlap_provenance,
        )
        if self.provenance_sha256 != actual_provenance_digest:
            raise ValueError("provenance partitions do not match provenance_sha256")
        if not isinstance(self.manifest, ChunkingManifest):
            raise TypeError("manifest must be a ChunkingManifest")
        if self.unit_count > self.manifest.budget:
            raise ValueError("unit_count exceeds the manifest budget")
        if self.manifest.unit_kind == "characters":
            if self.unit_count != len(self.text):
                raise ValueError("character unit_count must equal len(text)")
            if self.new_content_start - self.start > self.manifest.overlap:
                raise ValueError("character overlap exceeds the manifest overlap")
        expected_metadata = compute_chunk_metadata_sha256(
            manifest=self.manifest,
            index=self.index,
            start=self.start,
            end=self.end,
            new_content_start=self.new_content_start,
            unit_count=self.unit_count,
            text_sha256=self.text_sha256,
            provenance_sha256=self.provenance_sha256,
        )
        if self.chunk_metadata_sha256 != expected_metadata:
            raise ValueError("chunk metadata does not match chunk_metadata_sha256")

    @property
    def content(self) -> str:
        return self.text[self.new_content_start - self.start :]

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def chunk_id(self) -> str:
        return (
            f"{self.manifest.source_id}:manifest:{self.manifest.manifest_id}"
            f":chunk:sha256:{self.chunk_metadata_sha256}"
        )

    @property
    def overlap_span(self) -> tuple[int, int]:
        return self.start, self.new_content_start

    @property
    def overlap_char_count(self) -> int:
        return self.new_content_start - self.start

    @property
    def context_provenance(self) -> ChunkProvenance:
        return self.overlap_provenance


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return sum(1 for _ in _TOKEN_RE.finditer(text))


class _HierarchyIndex:
    def __init__(self, hierarchy: DocumentHierarchy) -> None:
        self._root = hierarchy.root
        self._child_ends: dict[int, tuple[int, ...]] = {}
        stack = [hierarchy.root]
        boundaries = {0, len(hierarchy.source)}
        heading_intervals: set[tuple[int, int]] = set()
        while stack:
            node = stack.pop()
            self._child_ends[id(node)] = tuple(child.end for child in node.children)
            if node.kind is not SegmentKind.DOCUMENT:
                boundaries.add(node.start)
                boundaries.add(node.end)
                for name, value in node.attributes:
                    if name == "heading_end":
                        try:
                            heading_end = int(value)
                        except ValueError:
                            continue
                        if node.start < heading_end <= node.end:
                            heading_intervals.add((node.start, heading_end))
            stack.extend(reversed(node.children))
        self.boundaries = tuple(sorted(boundaries))
        self.headings = _HeadingIndex(heading_intervals)

    def references(self, start: int, end: int) -> tuple[SegmentReference, ...]:
        if start >= end:
            return ()
        result: list[SegmentReference] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.end <= start or node.start >= end:
                continue
            if node.kind is not SegmentKind.DOCUMENT:
                result.append(SegmentReference.from_segment(node))
            if not node.children:
                continue
            child_ends = self._child_ends[id(node)]
            first = bisect.bisect_right(child_ends, start)
            stop = first
            while stop < len(node.children) and node.children[stop].start < end:
                stop += 1
            # Push only the intersecting index range in reverse preorder.
            # Slicing node.children[first:] copies every later sibling and
            # becomes quadratic across thousands of narrow chunks.
            for child_index in range(stop - 1, first - 1, -1):
                stack.append(node.children[child_index])
        return tuple(result)


_PROTECTED_KINDS = {
    SegmentKind.SECTION,
    SegmentKind.CLAUSE,
    SegmentKind.TABLE,
}


def _preserved_units(root: Segment) -> tuple[tuple[int, int], ...]:
    """Partition around protected containers at any descendant depth."""
    contains_protected: dict[int, bool] = {}
    stack: list[tuple[Segment, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children))
            continue
        contains_protected[id(node)] = (
            node.kind in _PROTECTED_KINDS
            or any(contains_protected[id(child)] for child in node.children)
        )

    units: list[tuple[int, int, bool]] = []

    def emit(start: int, end: int, protected: bool) -> None:
        if start >= end:
            return
        if (
            not protected
            and units
            and not units[-1][2]
            and units[-1][1] == start
        ):
            units[-1] = (units[-1][0], end, False)
        else:
            units.append((start, end, protected))

    def visit(node: Segment) -> None:
        protected_children = [
            child for child in node.children if contains_protected[id(child)]
        ]
        if node.kind in _PROTECTED_KINDS and not protected_children:
            emit(node.start, node.end, True)
            return
        if not protected_children:
            emit(node.start, node.end, False)
            return

        cursor = node.start
        for child in protected_children:
            emit(cursor, child.start, False)
            visit(child)
            cursor = child.end
        emit(cursor, node.end, False)

    visit(root)
    return tuple((start, end) for start, end, _protected in units)


class _HeadingIndex:
    """Bisect-backed heading containment checks used by every chunk."""

    def __init__(self, intervals: Iterable[tuple[int, int]]) -> None:
        ordered = tuple(sorted(set(intervals)))
        self.intervals = ordered
        self.starts = tuple(start for start, _end in ordered)
        self.ends = tuple(end for _start, end in ordered)

    def containing(self, position: int) -> tuple[int, int] | None:
        index = bisect.bisect_right(self.starts, position) - 1
        if (
            index >= 0
            and self.starts[index] < position < self.ends[index]
        ):
            return self.starts[index], self.ends[index]
        return None

    def safe_hard_end(self, fresh_start: int, hard_end: int) -> int:
        containing = self.containing(hard_end)
        if containing is None:
            return hard_end
        heading_start, _heading_end = containing
        if heading_start > fresh_start:
            return heading_start
        # An individually oversized heading must be hard-split.
        return hard_end

    def is_safe(self, boundary: int) -> bool:
        return self.containing(boundary) is None


def _boundary_end(
    boundaries: Sequence[int],
    headings: _HeadingIndex,
    *,
    fresh_start: int,
    hard_end: int,
    respect_boundaries: bool,
) -> int:
    hard_end = headings.safe_hard_end(fresh_start, hard_end)
    if not respect_boundaries:
        return hard_end
    position = bisect.bisect_right(boundaries, hard_end)
    while position:
        position -= 1
        candidate = boundaries[position]
        if candidate <= fresh_start:
            break
        if headings.is_safe(candidate):
            return candidate
    return hard_end


class _TokenCountCache:
    """A per-planning-step bounded cache for black-box token counters."""

    def __init__(
        self,
        source: str,
        counter: TokenCounter,
        *,
        max_entries: int = 256,
    ) -> None:
        self.source = source
        self.counter = counter
        self.max_entries = max_entries
        self.values: dict[tuple[int, int], int] = {}

    def count(self, start: int, end: int) -> int:
        key = (start, end)
        cached = self.values.get(key)
        if cached is not None:
            return cached
        value = self.counter(self.source[start:end])
        _integer(value, "token counter result", minimum=0)
        if len(self.values) >= self.max_entries:
            self.values.clear()
        self.values[key] = value
        return value


def _token_context_start(
    source: str,
    counter: TokenCounter,
    unit_start: int,
    fresh_start: int,
    overlap: int,
) -> int:
    if overlap == 0 or fresh_start == unit_start:
        return fresh_start
    cache = _TokenCountCache(source, counter)
    best = fresh_start
    distance = 1
    first_over: int | None = None
    while True:
        candidate = max(unit_start, fresh_start - distance)
        if cache.count(candidate, fresh_start) <= overlap:
            best = candidate
            if candidate == unit_start:
                return candidate
            distance *= 2
        else:
            first_over = candidate
            break

    low = first_over
    high = best
    while high - low > 1:
        middle = (low + high) // 2
        if cache.count(middle, fresh_start) <= overlap:
            high = middle
        else:
            low = middle
    return high


def _raw_token_end(
    cache: _TokenCountCache,
    context_start: int,
    fresh_start: int,
    limit: int,
    budget: int,
) -> int | None:
    if fresh_start >= limit:
        return None

    first = fresh_start + 1
    if cache.count(context_start, first) <= budget:
        last_feasible = first
    else:
        last_feasible = 0
        # A black-box counter is not required to be monotonic.  Search for a
        # feasible advancing slice before declaring that the budget is
        # impossible.
        for candidate in range(first + 1, limit + 1):
            if cache.count(context_start, candidate) <= budget:
                last_feasible = candidate
                break
        if not last_feasible:
            return None

    distance = max(1, last_feasible - fresh_start)
    first_over: int | None = None
    while last_feasible < limit:
        candidate = min(limit, fresh_start + distance * 2)
        if candidate <= last_feasible:
            candidate = min(limit, last_feasible + 1)
        if cache.count(context_start, candidate) <= budget:
            last_feasible = candidate
            if candidate == limit:
                return candidate
            distance = candidate - fresh_start
        else:
            first_over = candidate
            break

    if first_over is None:
        return last_feasible

    low = last_feasible
    high = first_over
    while high - low > 1:
        middle = (low + high) // 2
        if cache.count(context_start, middle) <= budget:
            low = middle
        else:
            high = middle
    return low


def _token_end(
    source: str,
    counter: TokenCounter,
    boundaries: Sequence[int],
    headings: _HeadingIndex,
    *,
    context_start: int,
    fresh_start: int,
    limit: int,
    budget: int,
    respect_boundaries: bool,
) -> tuple[int, int] | None:
    cache = _TokenCountCache(source, counter)
    raw = _raw_token_end(cache, context_start, fresh_start, limit, budget)
    if raw is None:
        return None

    # A heading boundary is advisory when using a caller-supplied token counter.
    # An arbitrary counter need not be monotonic: shortening a feasible span can
    # make it over budget.  Retain the known-feasible raw endpoint unless the
    # adjusted heading-safe endpoint is independently feasible.
    adjusted = headings.safe_hard_end(fresh_start, raw)
    boundary_limit = raw
    if (
        adjusted > fresh_start
        and cache.count(context_start, adjusted) <= budget
    ):
        boundary_limit = adjusted

    candidate = boundary_limit
    if respect_boundaries:
        position = bisect.bisect_right(boundaries, boundary_limit)
        candidate = 0
        while position:
            position -= 1
            boundary = boundaries[position]
            if boundary <= fresh_start:
                break
            if (
                headings.is_safe(boundary)
                and cache.count(context_start, boundary) <= budget
            ):
                candidate = boundary
                break
        if candidate == 0:
            candidate = boundary_limit
    count = cache.count(context_start, candidate)
    if count > budget or candidate <= fresh_start:
        return None
    return candidate, count


def _provenance_for(
    index: _HierarchyIndex,
    start: int,
    end: int,
    new_content_start: int,
) -> tuple[ChunkProvenance, ChunkProvenance, ChunkProvenance]:
    full_references = index.references(start, end)
    content_references = tuple(
        reference
        for reference in full_references
        if reference.end > new_content_start and reference.start < end
    )
    overlap_references = (
        ()
        if start == new_content_start
        else tuple(
            reference
            for reference in full_references
            if reference.end > start and reference.start < new_content_start
        )
    )
    return (
        ChunkProvenance(full_references),
        ChunkProvenance(content_references),
        ChunkProvenance(overlap_references),
    )


def _make_chunk(
    *,
    index_number: int,
    source: str,
    hierarchy_index: _HierarchyIndex,
    start: int,
    end: int,
    new_content_start: int,
    unit_count: int,
    manifest: ChunkingManifest,
) -> DocumentChunk:
    text = source[start:end]
    text_sha256 = _digest(text)
    provenance, content_provenance, overlap_provenance = _provenance_for(
        hierarchy_index,
        start,
        end,
        new_content_start,
    )
    provenance_sha256 = compute_provenance_sha256(
        provenance,
        content_provenance,
        overlap_provenance,
    )
    metadata_sha256 = compute_chunk_metadata_sha256(
        manifest=manifest,
        index=index_number,
        start=start,
        end=end,
        new_content_start=new_content_start,
        unit_count=unit_count,
        text_sha256=text_sha256,
        provenance_sha256=provenance_sha256,
    )
    return DocumentChunk(
        index_number,
        start,
        end,
        text,
        text_sha256,
        provenance,
        content_provenance,
        overlap_provenance,
        provenance_sha256,
        metadata_sha256,
        new_content_start,
        unit_count,
        manifest,
    )


def _normalise_document(
    document: str | DocumentHierarchy,
    *,
    sentence_segmenter,
    paragraph_segmenter,
    structural_spans,
    structural_mode,
    structure_profile,
    sentence_backend_id,
    paragraph_backend_id,
    structural_backend_id,
) -> DocumentHierarchy:
    if isinstance(document, DocumentHierarchy):
        mode = _enum(structural_mode, StructuralMode, "structural_mode")
        profile = _enum(structure_profile, StructureProfile, "structure_profile")
        if (
            sentence_segmenter is not None
            or paragraph_segmenter is not None
            or structural_spans is not None
            or sentence_backend_id is not None
            or paragraph_backend_id is not None
            or structural_backend_id is not None
            or mode is not StructuralMode.REPLACE
            or profile is not StructureProfile.CONSERVATIVE
        ):
            raise ValueError(
                "segmentation options cannot be supplied with DocumentHierarchy"
            )
        return document
    if not isinstance(document, str):
        raise TypeError("document must be a string or DocumentHierarchy")
    return segment_document(
        document,
        sentence_segmenter=sentence_segmenter,
        paragraph_segmenter=paragraph_segmenter,
        structural_spans=structural_spans,
        structural_mode=structural_mode,
        structure_profile=structure_profile,
        sentence_backend_id=sentence_backend_id,
        paragraph_backend_id=paragraph_backend_id,
        structural_backend_id=structural_backend_id,
    )


def iter_chunks(
    document: str | DocumentHierarchy,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    token_counter_id: str | None = None,
    overlap_chars: int = 0,
    overlap_tokens: int = 0,
    respect_boundaries: bool = True,
    container_policy: ContainerPolicy | str = ContainerPolicy.PRESERVE,
    document_id: str | None = None,
    sentence_segmenter=None,
    paragraph_segmenter=None,
    structural_spans: Iterable[StructuralSpan] | None = None,
    structural_mode: StructuralMode | str = StructuralMode.REPLACE,
    structure_profile: StructureProfile | str = StructureProfile.CONSERVATIVE,
    sentence_backend_id: str | None = None,
    paragraph_backend_id: str | None = None,
    structural_backend_id: str | None = None,
) -> Iterator[DocumentChunk]:
    if max_chars is not None and max_tokens is not None:
        raise ValueError("max_chars and max_tokens are mutually exclusive")
    if not isinstance(respect_boundaries, bool):
        raise TypeError("respect_boundaries must be a boolean")
    policy = _enum(container_policy, ContainerPolicy, "container_policy")

    if max_tokens is not None:
        budget = _integer(max_tokens, "max_tokens", minimum=1)
        if token_counter is None or not callable(token_counter):
            raise ValueError("max_tokens requires an explicit token_counter")
        if not isinstance(token_counter_id, str) or not token_counter_id.strip():
            raise ValueError("max_tokens requires a non-empty token_counter_id")
        if max_chars is not None or overlap_chars:
            raise ValueError("character budget options cannot be used in token mode")
        overlap = _integer(overlap_tokens, "overlap_tokens", minimum=0)
        unit_kind = "tokens"
    else:
        budget = _integer(
            DEFAULT_MAX_CHARS if max_chars is None else max_chars,
            "max_chars",
            minimum=1,
        )
        if token_counter is not None or token_counter_id is not None:
            raise ValueError("token_counter options require max_tokens")
        if overlap_tokens:
            raise ValueError("overlap_tokens requires max_tokens")
        overlap = _integer(overlap_chars, "overlap_chars", minimum=0)
        unit_kind = "characters"
    if overlap >= budget:
        raise ValueError("overlap must be smaller than the budget")

    hierarchy = _normalise_document(
        document,
        sentence_segmenter=sentence_segmenter,
        paragraph_segmenter=paragraph_segmenter,
        structural_spans=structural_spans,
        structural_mode=structural_mode,
        structure_profile=structure_profile,
        sentence_backend_id=sentence_backend_id,
        paragraph_backend_id=paragraph_backend_id,
        structural_backend_id=structural_backend_id,
    )
    source = hierarchy.source
    if not source:
        return

    manifest = ChunkingManifest(
        _digest(source),
        document_id,
        unit_kind,
        budget,
        overlap,
        respect_boundaries,
        policy,
        hierarchy.manifest,
        token_counter_id,
    )
    hierarchy_index = _HierarchyIndex(hierarchy)
    units = (
        ((0, len(source)),)
        if policy is ContainerPolicy.PACK_SIBLINGS
        else _preserved_units(hierarchy.root)
    )
    chunk_index = 0

    for unit_start, unit_end in units:
        fresh_start = unit_start
        first_in_unit = True
        while fresh_start < unit_end:
            if unit_kind == "characters":
                context_start = (
                    fresh_start
                    if first_in_unit
                    else max(unit_start, fresh_start - overlap)
                )
                if context_start + budget <= fresh_start:
                    context_start = fresh_start
                hard_end = min(unit_end, context_start + budget)
                end = _boundary_end(
                    hierarchy_index.boundaries,
                    hierarchy_index.headings,
                    fresh_start=fresh_start,
                    hard_end=hard_end,
                    respect_boundaries=respect_boundaries,
                )
                if end <= fresh_start:
                    context_start = fresh_start
                    hard_end = min(unit_end, context_start + budget)
                    end = _boundary_end(
                        hierarchy_index.boundaries,
                        hierarchy_index.headings,
                        fresh_start=fresh_start,
                        hard_end=hard_end,
                        respect_boundaries=respect_boundaries,
                    )
                if end <= fresh_start:
                    end = min(unit_end, fresh_start + budget)
                unit_count = end - context_start
            else:
                assert token_counter is not None
                context_start = (
                    fresh_start
                    if first_in_unit
                    else _token_context_start(
                        source,
                        token_counter,
                        unit_start,
                        fresh_start,
                        overlap,
                    )
                )
                planned = _token_end(
                    source,
                    token_counter,
                    hierarchy_index.boundaries,
                    hierarchy_index.headings,
                    context_start=context_start,
                    fresh_start=fresh_start,
                    limit=unit_end,
                    budget=budget,
                    respect_boundaries=respect_boundaries,
                )
                if planned is None and context_start != fresh_start:
                    context_start = fresh_start
                    planned = _token_end(
                        source,
                        token_counter,
                        hierarchy_index.boundaries,
                        hierarchy_index.headings,
                        context_start=context_start,
                        fresh_start=fresh_start,
                        limit=unit_end,
                        budget=budget,
                        respect_boundaries=respect_boundaries,
                    )
                if planned is None:
                    raise ValueError(
                        "token_counter cannot fit advancing source content "
                        "within max_tokens"
                    )
                end, unit_count = planned

            if end <= fresh_start or unit_count > budget:
                raise RuntimeError("chunk planner violated its strict progress invariant")
            yield _make_chunk(
                index_number=chunk_index,
                source=source,
                hierarchy_index=hierarchy_index,
                start=context_start,
                end=end,
                new_content_start=fresh_start,
                unit_count=unit_count,
                manifest=manifest,
            )
            chunk_index += 1
            fresh_start = end
            first_in_unit = False


def chunk_document(
    document: str | DocumentHierarchy,
    *,
    max_chars: int | None = None,
    max_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    token_counter_id: str | None = None,
    overlap_chars: int = 0,
    overlap_tokens: int = 0,
    respect_boundaries: bool = True,
    container_policy: ContainerPolicy | str = ContainerPolicy.PRESERVE,
    document_id: str | None = None,
    sentence_segmenter=None,
    paragraph_segmenter=None,
    structural_spans: Iterable[StructuralSpan] | None = None,
    structural_mode: StructuralMode | str = StructuralMode.REPLACE,
    structure_profile: StructureProfile | str = StructureProfile.CONSERVATIVE,
    sentence_backend_id: str | None = None,
    paragraph_backend_id: str | None = None,
    structural_backend_id: str | None = None,
) -> list[DocumentChunk]:
    return list(
        iter_chunks(
            document,
            max_chars=max_chars,
            max_tokens=max_tokens,
            token_counter=token_counter,
            token_counter_id=token_counter_id,
            overlap_chars=overlap_chars,
            overlap_tokens=overlap_tokens,
            respect_boundaries=respect_boundaries,
            container_policy=container_policy,
            document_id=document_id,
            sentence_segmenter=sentence_segmenter,
            paragraph_segmenter=paragraph_segmenter,
            structural_spans=structural_spans,
            structural_mode=structural_mode,
            structure_profile=structure_profile,
            sentence_backend_id=sentence_backend_id,
            paragraph_backend_id=paragraph_backend_id,
            structural_backend_id=structural_backend_id,
        )
    )


def reconstruct_chunks(chunks: Iterable[DocumentChunk]) -> str:
    materialised = tuple(chunks)
    if not materialised:
        return ""
    if any(not isinstance(chunk, DocumentChunk) for chunk in materialised):
        raise TypeError("chunks must contain only DocumentChunk objects")
    manifest = materialised[0].manifest
    expected_start = 0
    pieces: list[str] = []
    for expected_index, chunk in enumerate(materialised):
        if chunk.index != expected_index:
            raise ValueError("chunks must be ordered with contiguous indices")
        if chunk.manifest != manifest:
            raise ValueError("chunks do not share one manifest")
        if chunk.new_content_start != expected_start:
            raise ValueError("chunks do not provide contiguous fresh source coverage")
        pieces.append(chunk.content)
        expected_start = chunk.end
    reconstructed = "".join(pieces)
    if _digest(reconstructed) != manifest.source_sha256:
        raise ValueError("reconstructed chunks do not match the manifest source SHA-256")
    return reconstructed


__all__ = [
    "CHUNKING_SCHEMA_VERSION",
    "CHUNKING_SERIALIZER_VERSION",
    "DEFAULT_MAX_CHARS",
    "ChunkingManifest",
    "ChunkProvenance",
    "ContainerPolicy",
    "DocumentChunk",
    "SegmentReference",
    "TokenCounter",
    "chunk_document",
    "compute_chunk_metadata_sha256",
    "compute_provenance_sha256",
    "count_tokens",
    "iter_chunks",
    "reconstruct_chunks",
]
