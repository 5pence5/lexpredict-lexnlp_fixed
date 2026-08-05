"""Lossless, source-mapped hierarchical segmentation for English legal text.

The module is deliberately dependency-light.  It preserves source offsets exactly,
keeps legacy segmenters lazy, and accepts exact-span paragraph, sentence, and
layout backends without importing them.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeAlias

HIERARCHY_SCHEMA_VERSION = 1
MAX_HIERARCHY_DEPTH = 128


class SegmentKind(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    CLAUSE = "clause"
    LIST_ITEM = "list_item"
    TABLE = "table"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    TEXT = "text"
    SEPARATOR = "separator"


class StructuralMode(str, Enum):
    REPLACE = "replace"
    AUGMENT = "augment"


class StructureProfile(str, Enum):
    CONSERVATIVE = "conservative"
    STATUTE = "statute"


SegmentAttribute: TypeAlias = tuple[str, str]
SentenceSpan: TypeAlias = tuple[int, int] | tuple[int, int, str]
SentenceSegmenter: TypeAlias = Callable[[str], Iterable[SentenceSpan]]
ParagraphSpan: TypeAlias = tuple[int, int] | tuple[int, int, str]
ParagraphSegmenter: TypeAlias = Callable[[str], Iterable[ParagraphSpan]]


def _enum(value: object, enum_type: type[Enum], name: str):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise ValueError(f"{name} must be one of {choices}") from exc


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _attributes(value: object) -> tuple[SegmentAttribute, ...]:
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


def _label(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError("label must be None or a string")
    return value


@dataclass(frozen=True, slots=True)
class HierarchyManifest:
    structural_detector_id: str
    structural_mode: StructuralMode
    paragraph_backend_id: str
    sentence_backend_id: str
    structure_profile: StructureProfile = StructureProfile.CONSERVATIVE
    tree_sha256: str | None = None
    schema_version: int = HIERARCHY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "structural_detector_id",
            "paragraph_backend_id",
            "sentence_backend_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "structural_mode",
            _enum(self.structural_mode, StructuralMode, "structural_mode"),
        )
        object.__setattr__(
            self,
            "structure_profile",
            _enum(self.structure_profile, StructureProfile, "structure_profile"),
        )
        _integer(self.schema_version, "schema_version", minimum=1)
        if self.tree_sha256 is not None:
            if (
                not isinstance(self.tree_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.tree_sha256) is None
            ):
                raise ValueError("tree_sha256 must be a lowercase SHA-256 digest")


_CALLER_TREE_MANIFEST = HierarchyManifest(
    structural_detector_id="caller_tree:unspecified",
    structural_mode=StructuralMode.REPLACE,
    paragraph_backend_id="caller_tree:embedded",
    sentence_backend_id="caller_tree:embedded",
)


@dataclass(frozen=True, slots=True)
class Segment:
    kind: SegmentKind
    start: int
    end: int
    children: tuple["Segment", ...] = ()
    label: str | None = None
    level: int | None = None
    attributes: tuple[SegmentAttribute, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, SegmentKind, "kind"))
        _integer(self.start, "start", minimum=0)
        _integer(self.end, "end", minimum=0)
        if self.end < self.start:
            raise ValueError("segment end must not precede start")
        if self.kind is not SegmentKind.DOCUMENT and self.end == self.start:
            raise ValueError("non-document segments must have positive length")
        try:
            children = tuple(self.children)
        except TypeError as exc:
            raise TypeError("children must be an iterable of Segment objects") from exc
        if any(not isinstance(child, Segment) for child in children):
            raise TypeError("children must contain only Segment objects")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "label", _label(self.label))
        if self.level is not None:
            _integer(self.level, "level", minimum=0)
        object.__setattr__(self, "attributes", _attributes(self.attributes))

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def segment_id(self) -> str:
        return f"{self.kind.value}:{self.start}:{self.end}"

    def text(self, source: str) -> str:
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if self.end > len(source):
            raise ValueError("segment lies outside source")
        return source[self.start:self.end]

    def walk(self, kind: SegmentKind | str | None = None) -> Iterator["Segment"]:
        wanted = None if kind is None else _enum(kind, SegmentKind, "kind")
        stack = [self]
        while stack:
            node = stack.pop()
            if wanted is None or node.kind is wanted:
                yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> Iterator["Segment"]:
        stack = [self]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(reversed(node.children))
            else:
                yield node

    def reconstruct(self, source: str) -> str:
        return "".join(leaf.text(source) for leaf in self.leaves())


@dataclass(frozen=True, slots=True)
class StructuralSpan:
    kind: SegmentKind
    start: int
    end: int
    label: str | None = None
    level: int | None = None
    attributes: tuple[SegmentAttribute, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, SegmentKind, "kind"))
        if self.kind not in {
            SegmentKind.SECTION,
            SegmentKind.CLAUSE,
            SegmentKind.LIST_ITEM,
            SegmentKind.TABLE,
        }:
            raise ValueError("StructuralSpan kind must be section, clause, list_item, or table")
        _integer(self.start, "start", minimum=0)
        _integer(self.end, "end", minimum=0)
        if self.end <= self.start:
            raise ValueError("structural spans must have positive length")
        object.__setattr__(self, "label", _label(self.label))
        if self.level is not None:
            _integer(self.level, "level", minimum=0)
        object.__setattr__(self, "attributes", _attributes(self.attributes))


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


def _tree_sha256(root: Segment) -> str:
    digest = hashlib.sha256()
    digest.update(b"lexnlp.hierarchy-tree.v1\0")
    stack = [root]
    while stack:
        node = stack.pop()
        for value in (
            node.kind.value,
            node.start,
            node.end,
            node.label,
            node.level,
            len(node.children),
            len(node.attributes),
        ):
            _digest_value(digest, value)
        for name, value in node.attributes:
            _digest_value(digest, name)
            _digest_value(digest, value)
        stack.extend(reversed(node.children))
    return digest.hexdigest()


def _validate_hierarchy(source: str, root: Segment) -> None:
    if root.kind is not SegmentKind.DOCUMENT:
        raise ValueError("root segment must have kind DOCUMENT")
    if root.start != 0 or root.end != len(source):
        raise ValueError("root must cover the exact source span")
    if source and not root.children:
        raise ValueError("a non-empty document root must contain partitioning children")

    seen: set[str] = set()
    stack: list[tuple[Segment, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_HIERARCHY_DEPTH:
            raise ValueError(
                f"hierarchy exceeds MAX_HIERARCHY_DEPTH={MAX_HIERARCHY_DEPTH}"
            )
        if node.end > len(source):
            raise ValueError(f"{node.segment_id} lies outside source")
        if node.segment_id in seen:
            raise ValueError(f"duplicate segment identity {node.segment_id!r}")
        seen.add(node.segment_id)
        if node.children:
            cursor = node.start
            for child in node.children:
                if child.start != cursor:
                    raise ValueError(
                        f"children of {node.segment_id} do not exactly partition source"
                    )
                if child.end > node.end:
                    raise ValueError(f"child {child.segment_id} exceeds its parent")
                cursor = child.end
            if cursor != node.end:
                raise ValueError(
                    f"children of {node.segment_id} do not exactly partition source"
                )
            for child in reversed(node.children):
                stack.append((child, depth + 1))


@dataclass(frozen=True, slots=True)
class DocumentHierarchy:
    source: str
    root: Segment
    manifest: HierarchyManifest = _CALLER_TREE_MANIFEST

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if not isinstance(self.root, Segment):
            raise TypeError("root must be a Segment")
        if not isinstance(self.manifest, HierarchyManifest):
            raise TypeError("manifest must be a HierarchyManifest")
        _validate_hierarchy(self.source, self.root)
        realised = _tree_sha256(self.root)
        if self.manifest.tree_sha256 is None:
            object.__setattr__(
                self, "manifest", replace(self.manifest, tree_sha256=realised)
            )
        elif self.manifest.tree_sha256 != realised:
            raise ValueError("hierarchy tree does not match manifest tree_sha256")
        if self.reconstruct() != self.source:
            raise ValueError("hierarchy does not reconstruct the exact source")

    @classmethod
    def from_segments(
        cls,
        source: str,
        children: Iterable[Segment],
        *,
        manifest: HierarchyManifest = _CALLER_TREE_MANIFEST,
    ) -> "DocumentHierarchy":
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        materialised = tuple(children)
        if source and not materialised:
            raise ValueError("non-empty source requires at least one child segment")
        root = Segment(SegmentKind.DOCUMENT, 0, len(source), materialised, level=0)
        return cls(source, root, manifest)

    def text(self, segment: Segment | None = None) -> str:
        return (self.root if segment is None else segment).text(self.source)

    def segments(self, kind: SegmentKind | str | None = None) -> Iterator[Segment]:
        return self.root.walk(kind)

    def leaves(self) -> Iterator[Segment]:
        return self.root.leaves()

    def reconstruct(self) -> str:
        return self.root.reconstruct(self.source)


@dataclass(frozen=True, slots=True)
class _Line:
    index: int
    start: int
    content_end: int
    end: int
    content: str
    stripped: str


@dataclass(slots=True)
class _Heading:
    line_index: int
    start: int
    heading_end: int
    label: str
    heading_type: str
    number: str | None
    parts: tuple[str, ...]
    level: int = 1
    end: int = 0


@dataclass(frozen=True, slots=True)
class _NumericCandidate:
    line_index: int
    parts: tuple[str, ...]
    title: str


@dataclass(slots=True)
class _SpanNode:
    span: StructuralSpan
    children: list["_SpanNode"]


_NEWLINE_AT_END_RE = re.compile(r"(?:\r\n|\n\r|\r|\n)$")
_BLANK_LINE_RE = re.compile(
    r"(?:(?:[ \t]*)(?:\r\n|\n\r|\r(?!\n)|\n(?!\r))){2,}"
)
_PAGE_MARKER_RE = re.compile(
    r"^[ \t]*(?:<PAGE>(?:[ \t]+\d+)?|PAGE[ \t]+\d+(?:[ \t]+OF[ \t]+\d+)?)[ \t]*(?:\r\n|\n\r|\r|\n|$)",
    re.IGNORECASE | re.MULTILINE,
)
_EXPLICIT_HEADING_RE = re.compile(
    r"^\s*(?P<kind>SCHEDULE|EXHIBIT|APPENDIX|PART|TITLE|SUBTITLE|CHAPTER|ARTICLE|SECTION)"
    r"\s+(?P<number>[A-Z0-9IVXLCDM]+(?:[-.][A-Z0-9IVXLCDM]+)*(?:\([A-Z0-9IVXLCDM]+\))*)"
    r"(?:(?:\s*[—–:-]\s*|\s+)(?P<title>.*?))?\s*$",
    re.IGNORECASE,
)
_NUMERIC_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+)*(?:\([A-Za-z0-9ivxlcdm]+\))*)"
    r"(?P<terminal>\.)?\s+(?P<title>\S.*?)\s*$"
)
_CLAUSE_RE = re.compile(
    r"^\s*(?P<label>(?:[A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)+|\d+\.)"
    r"(?:\([A-Za-z0-9ivxlcdm]+\))*)\s+",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"^\s*(?P<label>(?:\([A-Za-z0-9ivxlcdm]+\)|\d+\)|•))\s+",
    re.IGNORECASE,
)


def _split_lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    offset = 0
    for index, raw in enumerate(text.splitlines(keepends=True)):
        newline = _NEWLINE_AT_END_RE.search(raw)
        content = raw[: newline.start()] if newline else raw
        end = offset + len(raw)
        result.append(
            _Line(index, offset, offset + len(content), end, content, content.strip())
        )
        offset = end
    if offset < len(text):
        content = text[offset:]
        result.append(
            _Line(len(result), offset, len(text), len(text), content, content.strip())
        )
    return result


def _delimited_row_shape(line: _Line) -> tuple[str, int] | None:
    """Return delimiter kind and non-empty semantic cell count for one row."""
    pipe_count = line.content.count("|")
    tab_count = line.content.count("\t")
    if pipe_count and not tab_count:
        delimiter = "|"
    elif tab_count and not pipe_count:
        delimiter = "\t"
    else:
        return None
    cells = [cell.strip() for cell in line.content.split(delimiter)]
    if cells and not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    if len(cells) < 2 or not all(cells):
        return None
    return delimiter, len(cells)


def _delimited_blocks(
    lines: Sequence[_Line],
) -> tuple[list[tuple[int, int]], set[int]]:
    """Classify table rows once, before any competing structural detector.

    Rows with the historical strong evidence (at least two pipe or tab
    delimiters) remain tables on their own.  A one-delimiter/two-cell row is
    weaker evidence and is promoted only by an adjacent row with the same
    delimiter and semantic cell count, including a strong Markdown-style row.
    """
    table_lines = {
        line.index
        for line in lines
        if line.content.count("|") >= 2 or line.content.count("\t") >= 2
    }
    row_shapes = tuple(_delimited_row_shape(line) for line in lines)
    index = 0
    while index < len(lines):
        shape = row_shapes[index]
        if shape is None:
            index += 1
            continue
        run_start = index
        index += 1
        while index < len(lines) and row_shapes[index] == shape:
            index += 1
        if index - run_start >= 2:
            table_lines.update(
                lines[position].index for position in range(run_start, index)
            )

    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if lines[index].index not in table_lines:
            index += 1
            continue
        start_index = index
        index += 1
        while index < len(lines) and lines[index].index in table_lines:
            index += 1
        blocks.append((lines[start_index].start, lines[index - 1].end))
    return blocks, table_lines


def _number_parts(number: str) -> tuple[str, ...]:
    return tuple(
        part.casefold()
        for part in re.findall(r"\d+|[A-Za-z]+", number)
    )


def _is_upper_heading(title: str) -> bool:
    letters = [character for character in title if character.isalpha()]
    return bool(letters) and all(not character.islower() for character in letters)


def _is_initial_document_title(text: str) -> bool:
    """Conservatively recognize a standalone all-caps legal document title."""
    if len(text) > 200 or text.endswith((".", ":", ";", "?", "!")):
        return False
    words = re.findall(r"[^\W\d_]+", text, re.UNICODE)
    return len(words) >= 2 and _is_upper_heading(text)


def _ordinal(value: str) -> tuple[str, int] | None:
    if value.isdigit():
        return ("number", int(value))
    if len(value) == 1 and value.isalpha():
        return ("letter", ord(value.casefold()))
    return None


def _sequence_numbered_heading_indices(
    candidates: Sequence[_NumericCandidate],
) -> set[int]:
    """Return statute-sequence candidates in linear time.

    Sibling evidence is indexed by the exact parent numbering tuple.  An
    adjacency graph then promotes descendants of evidenced parents without
    rescanning the candidate list.
    """
    promoted: set[int] = set()
    last_by_parent: dict[tuple[str, ...], tuple[tuple[str, int], int]] = {}
    index_by_parts: dict[tuple[str, ...], int] = {}
    children_by_index: dict[int, list[int]] = {}

    for candidate in candidates:
        if candidate.parts:
            parent_index = index_by_parts.get(candidate.parts[:-1])
            if parent_index is not None:
                children_by_index.setdefault(parent_index, []).append(
                    candidate.line_index
                )
        index_by_parts[candidate.parts] = candidate.line_index
        if not candidate.parts:
            continue
        current = _ordinal(candidate.parts[-1])
        if current is None:
            continue
        parent = candidate.parts[:-1]
        previous = last_by_parent.get(parent)
        if (
            previous is not None
            and previous[0][0] == current[0]
            and current[1] == previous[0][1] + 1
        ):
            promoted.add(previous[1])
            promoted.add(candidate.line_index)
        last_by_parent[parent] = (current, candidate.line_index)

    queue = list(promoted)
    cursor = 0
    while cursor < len(queue):
        parent_index = queue[cursor]
        cursor += 1
        for child_index in children_by_index.get(parent_index, ()):
            if child_index not in promoted:
                promoted.add(child_index)
                queue.append(child_index)
    return promoted


def _heading_candidates(
    lines: Sequence[_Line],
    profile: StructureProfile,
    table_lines: set[int],
) -> tuple[list[_Heading], set[int]]:
    numeric: list[_NumericCandidate] = []
    numeric_matches: dict[int, re.Match[str]] = {}
    explicit_matches: dict[int, re.Match[str]] = {}
    first_content_index = next(
        (line.index for line in lines if line.stripped),
        None,
    )

    for line in lines:
        if not line.stripped:
            continue
        # Table evidence is classified before headings so a leading cell such
        # as "2." or "SECTION 2" cannot acquire a competing structural scope.
        if line.index in table_lines:
            continue
        explicit = _EXPLICIT_HEADING_RE.match(line.content)
        if explicit:
            explicit_matches[line.index] = explicit
            continue
        match = _NUMERIC_HEADING_RE.match(line.content)
        if match:
            numeric_matches[line.index] = match
            numeric.append(
                _NumericCandidate(
                    line.index,
                    _number_parts(match.group("number")),
                    match.group("title"),
                )
            )

    promoted = (
        _sequence_numbered_heading_indices(numeric)
        if profile is StructureProfile.STATUTE
        else set()
    )
    headings: list[_Heading] = []
    heading_lines: set[int] = set()

    for line in lines:
        explicit = explicit_matches.get(line.index)
        if explicit is not None:
            heading_type = explicit.group("kind").upper()
            number = explicit.group("number")
            headings.append(
                _Heading(
                    line.index,
                    line.start,
                    line.content_end,
                    line.stripped,
                    heading_type,
                    number,
                    _number_parts(number) if heading_type == "SECTION" else (),
                )
            )
            heading_lines.add(line.index)
            continue

        match = numeric_matches.get(line.index)
        if match is None:
            if (
                line.index == first_content_index
                and line.index not in table_lines
                and _is_initial_document_title(line.stripped)
            ):
                headings.append(
                    _Heading(
                        line.index,
                        line.start,
                        line.content_end,
                        line.stripped,
                        "DOCUMENT_TITLE",
                        None,
                        (),
                    )
                )
                heading_lines.add(line.index)
            continue
        title = match.group("title")
        if not _is_upper_heading(title) and line.index not in promoted:
            continue
        number = match.group("number")
        headings.append(
            _Heading(
                line.index,
                line.start,
                line.content_end,
                line.stripped,
                "NUMBERED",
                number,
                _number_parts(number),
            )
        )
        heading_lines.add(line.index)

    return headings, heading_lines


_COMPATIBLE_PARENTS: dict[str, tuple[str, ...]] = {
    "PART": ("SCHEDULE", "EXHIBIT", "APPENDIX"),
    "SUBTITLE": ("TITLE",),
    "CHAPTER": ("SUBTITLE", "TITLE", "PART"),
    "ARTICLE": ("PART", "TITLE", "CHAPTER", "SCHEDULE"),
    "SECTION": (
        "ARTICLE",
        "CHAPTER",
        "SUBTITLE",
        "TITLE",
        "PART",
        "SCHEDULE",
        "EXHIBIT",
        "APPENDIX",
    ),
    "NUMBERED": (
        "SECTION",
        "ARTICLE",
        "CHAPTER",
        "SUBTITLE",
        "TITLE",
        "PART",
        "SCHEDULE",
        "EXHIBIT",
        "APPENDIX",
    ),
}


def _assign_heading_levels(headings: list[_Heading], source_end: int) -> None:
    active: list[_Heading] = []
    for heading in headings:
        level: int | None = None
        if heading.parts and heading.heading_type in {"SECTION", "NUMBERED"}:
            for parent in reversed(active):
                if (
                    parent.parts
                    and len(parent.parts) < len(heading.parts)
                    and heading.parts[: len(parent.parts)] == parent.parts
                ):
                    level = parent.level + len(heading.parts) - len(parent.parts)
                    break

        if level is None:
            allowed = _COMPATIBLE_PARENTS.get(heading.heading_type, ())
            for parent in reversed(active):
                if parent.heading_type in allowed:
                    level = parent.level + 1
                    break

        if level is None:
            level = 1

        heading.level = level
        while active and active[-1].level >= level:
            active.pop()
        active.append(heading)

    open_indices: list[int] = []
    for index, heading in enumerate(headings):
        while open_indices and headings[open_indices[-1]].level >= heading.level:
            headings[open_indices.pop()].end = heading.start
        open_indices.append(index)
    while open_indices:
        headings[open_indices.pop()].end = source_end


def _section_spans(
    text: str,
    lines: Sequence[_Line],
    profile: StructureProfile,
    table_lines: set[int],
) -> tuple[list[StructuralSpan], set[int]]:
    headings, heading_lines = _heading_candidates(lines, profile, table_lines)
    _assign_heading_levels(headings, len(text))
    spans = [
        StructuralSpan(
            SegmentKind.SECTION,
            heading.start,
            heading.end,
            heading.label,
            heading.level,
            (
                ("heading_type", heading.heading_type),
                ("heading_end", str(heading.heading_end)),
                *((("number", heading.number),) if heading.number is not None else ()),
            ),
        )
        for heading in headings
    ]
    return spans, heading_lines


def _outline_spans(
    text: str,
    lines: Sequence[_Line],
    sections: Sequence[StructuralSpan],
    heading_lines: set[int],
    table_lines: set[int],
) -> list[StructuralSpan]:
    sorted_sections = sorted(sections, key=lambda span: (span.start, -span.end))
    next_section = 0
    active_sections: list[StructuralSpan] = []
    grouped: dict[
        tuple[int, int] | None,
        list[tuple[int, _Line, SegmentKind, str, int]],
    ] = {}

    for line in lines:
        while active_sections and active_sections[-1].end <= line.start:
            active_sections.pop()
        while (
            next_section < len(sorted_sections)
            and sorted_sections[next_section].start <= line.start
        ):
            section = sorted_sections[next_section]
            next_section += 1
            while active_sections and active_sections[-1].end <= section.start:
                active_sections.pop()
            active_sections.append(section)

        if line.index in heading_lines or not line.stripped:
            continue
        # Delimited table rows own their line-level structural evidence.
        if line.index in table_lines:
            continue
        list_match = _LIST_RE.match(line.content)
        clause_match = _CLAUSE_RE.match(line.content)
        parent = active_sections[-1] if active_sections else None
        parent_key = (parent.start, parent.end) if parent else None
        base_level = (parent.level or 0) + 1 if parent else 1

        if list_match:
            marker_offset = list_match.start("label")
            marker_start = line.start + marker_offset
            grouped.setdefault(parent_key, []).append(
                (
                    marker_start,
                    line,
                    SegmentKind.LIST_ITEM,
                    list_match.group("label"),
                    base_level + 1 + marker_offset,
                )
            )
        elif clause_match:
            label = clause_match.group("label")
            marker_start = line.start + clause_match.start("label")
            parts = _number_parts(label)
            extra_depth = max(0, len(parts) - 2)
            grouped.setdefault(parent_key, []).append(
                (
                    marker_start,
                    line,
                    SegmentKind.CLAUSE,
                    label,
                    base_level + extra_depth,
                )
            )

    result: list[StructuralSpan] = []
    parent_ends = {
        (span.start, span.end): span.end for span in sections
    }
    for parent_key, markers in grouped.items():
        limit = len(text) if parent_key is None else parent_ends[parent_key]
        ends = [limit] * len(markers)
        stack: list[int] = []
        for index, (_marker_start, line, _kind, _label_value, level) in enumerate(markers):
            while stack and markers[stack[-1]][4] >= level:
                # Structural spans begin at their marker, but the prior sibling
                # must stop before indentation belonging to this source line.
                ends[stack.pop()] = line.start
            stack.append(index)
        while stack:
            ends[stack.pop()] = limit
        for marker, end in zip(markers, ends):
            marker_start, line, kind, label_value, level = marker
            if end > marker_start:
                result.append(
                    StructuralSpan(
                        kind,
                        marker_start,
                        end,
                        label_value,
                        level,
                        (("heading_end", str(line.content_end)),),
                    )
                )
    # A builtin outline marker never owns a later section heading.  Cap every
    # open clause/list scope at the nearest such heading without rescanning the
    # section list for each marker.
    section_starts = tuple(sorted({section.start for section in sections}))
    if section_starts:
        for index, span in enumerate(result):
            position = bisect.bisect_right(section_starts, span.start)
            if (
                position < len(section_starts)
                and section_starts[position] < span.end
            ):
                result[index] = replace(
                    span,
                    end=section_starts[position],
                )
    return result


def _table_spans(
    blocks: Sequence[tuple[int, int]],
    containers: Sequence[StructuralSpan],
) -> list[StructuralSpan]:
    # Both inputs are source ordered and the structural containers are
    # laminar.  Sweep each container once rather than rescanning C containers
    # for every one of T tables.
    ordered_containers = sorted(
        containers,
        key=lambda span: (span.start, -span.end, span.kind.value),
    )
    active: list[StructuralSpan] = []
    container_index = 0
    result: list[StructuralSpan] = []
    for start, end in blocks:
        while active and active[-1].end <= start:
            active.pop()
        while (
            container_index < len(ordered_containers)
            and ordered_containers[container_index].start <= start
        ):
            candidate = ordered_containers[container_index]
            container_index += 1
            while active and candidate.start >= active[-1].end:
                active.pop()
            if candidate.end <= start:
                continue
            active.append(candidate)
        while active and active[-1].end < end:
            active.pop()
        parent = (
            active[-1]
            if active and active[-1].start <= start and end <= active[-1].end
            else None
        )
        level = ((parent.level or 0) + 1) if parent else 1
        result.append(
            StructuralSpan(
                SegmentKind.TABLE,
                start,
                end,
                None,
                level,
                (("detector", "delimited_lines"),),
            )
        )
    return result


def _builtin_structural_spans(
    text: str,
    profile: StructureProfile,
) -> tuple[StructuralSpan, ...]:
    lines = _split_lines(text)
    table_blocks, table_lines = _delimited_blocks(lines)
    sections, heading_lines = _section_spans(
        text,
        lines,
        profile,
        table_lines,
    )
    outlines = _outline_spans(
        text,
        lines,
        sections,
        heading_lines,
        table_lines,
    )
    tables = _table_spans(table_blocks, (*sections, *outlines))
    spans = (*sections, *outlines, *tables)
    return tuple(sorted(spans, key=lambda span: (span.start, -span.end, span.kind.value)))


def _validate_structural_spans(
    spans: Iterable[StructuralSpan],
    source_length: int,
) -> tuple[StructuralSpan, ...]:
    materialised = tuple(spans)
    if any(not isinstance(span, StructuralSpan) for span in materialised):
        raise TypeError("structural_spans must contain only StructuralSpan objects")
    for span in materialised:
        if span.end > source_length:
            raise ValueError("structural span lies outside source")

    ordered = tuple(
        sorted(materialised, key=lambda span: (span.start, -span.end, span.kind.value))
    )
    stack: list[StructuralSpan] = []
    seen_ranges: set[tuple[int, int]] = set()
    for span in ordered:
        key = (span.start, span.end)
        if key in seen_ranges:
            raise ValueError(f"identical structural ranges conflict: {key}")
        seen_ranges.add(key)
        while stack and span.start >= stack[-1].end:
            stack.pop()
        if stack and span.end > stack[-1].end:
            raise ValueError(
                f"crossing structural spans are not laminar: "
                f"{stack[-1].start}:{stack[-1].end} and {span.start}:{span.end}"
            )
        if len(stack) + 1 > MAX_HIERARCHY_DEPTH:
            raise ValueError(
                f"structural spans exceed MAX_HIERARCHY_DEPTH={MAX_HIERARCHY_DEPTH}"
            )
        stack.append(span)
    return ordered


def _span_forest(spans: Sequence[StructuralSpan]) -> list[_SpanNode]:
    roots: list[_SpanNode] = []
    stack: list[_SpanNode] = []
    for span in spans:
        while stack and span.start >= stack[-1].span.end:
            stack.pop()
        node = _SpanNode(span, [])
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _span_result(
    local_text: str,
    result: object,
    *,
    backend_name: str,
) -> tuple[int, int]:
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise TypeError(f"{backend_name} must yield 2- or 3-tuples")
    start, end = result[0], result[1]
    _integer(start, f"{backend_name} span start", minimum=0)
    _integer(end, f"{backend_name} span end", minimum=0)
    if end <= start or end > len(local_text):
        raise ValueError(f"{backend_name} yielded an invalid span")
    if len(result) == 3:
        returned_text = result[2]
        if not isinstance(returned_text, str):
            raise TypeError(f"{backend_name} third tuple item must be a string")
        if returned_text != local_text[start:end]:
            raise ValueError(f"{backend_name} text does not match its exact span")
    return start, end


def _prediction_spans(
    local_text: str,
    predictions: Iterable[object],
    *,
    backend_name: str,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    cursor = 0
    for prediction in predictions:
        start, end = _span_result(local_text, prediction, backend_name=backend_name)
        if start < cursor:
            raise ValueError(f"{backend_name} spans must be ordered and non-overlapping")
        result.append((start, end))
        cursor = end
    return tuple(result)


def _separator_intervals(text: str) -> tuple[tuple[int, int], ...]:
    raw = [
        match.span() for expression in (_BLANK_LINE_RE, _PAGE_MARKER_RE)
        for match in expression.finditer(text)
    ]
    if not raw:
        return ()
    raw.sort()
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _default_paragraph_spans(text: str) -> Iterator[ParagraphSpan]:
    cursor = 0
    for start, end in _separator_intervals(text):
        if cursor < start and text[cursor:start].strip():
            yield cursor, start, text[cursor:start]
        cursor = end
    if cursor < len(text) and text[cursor:].strip():
        yield cursor, len(text), text[cursor:]


def _legacy_sentence_segmenter(text: str) -> Iterable[SentenceSpan]:
    from lexnlp.nlp.en.segments.sentences import get_sentence_span

    return get_sentence_span(text)


def _generic_segment(kind: SegmentKind, start: int, end: int) -> Segment:
    return Segment(kind, start, end)


def _gap_segment(source: str, start: int, end: int) -> Segment:
    kind = SegmentKind.SEPARATOR if not source[start:end].strip() else SegmentKind.TEXT
    return _generic_segment(kind, start, end)


def _sentence_children(
    source: str,
    start: int,
    end: int,
    sentence_segmenter: SentenceSegmenter,
) -> tuple[Segment, ...]:
    local = source[start:end]
    spans = _prediction_spans(
        local,
        sentence_segmenter(local),
        backend_name="sentence_segmenter",
    )
    children: list[Segment] = []
    cursor = 0
    for local_start, local_end in spans:
        if cursor < local_start:
            children.append(_gap_segment(source, start + cursor, start + local_start))
        children.append(
            Segment(
                SegmentKind.SENTENCE,
                start + local_start,
                start + local_end,
            )
        )
        cursor = local_end
    if cursor < len(local):
        children.append(_gap_segment(source, start + cursor, end))
    return tuple(children)


def _plain_segments(
    source: str,
    start: int,
    end: int,
    paragraph_segmenter: ParagraphSegmenter,
    sentence_segmenter: SentenceSegmenter,
) -> tuple[Segment, ...]:
    if start >= end:
        return ()
    local = source[start:end]
    spans = _prediction_spans(
        local,
        paragraph_segmenter(local),
        backend_name="paragraph_segmenter",
    )
    children: list[Segment] = []
    cursor = 0
    for local_start, local_end in spans:
        if cursor < local_start:
            children.append(_gap_segment(source, start + cursor, start + local_start))
        absolute_start = start + local_start
        absolute_end = start + local_end
        if source[absolute_start:absolute_end].strip():
            sentence_children = _sentence_children(
                source,
                absolute_start,
                absolute_end,
                sentence_segmenter,
            )
            children.append(
                Segment(
                    SegmentKind.PARAGRAPH,
                    absolute_start,
                    absolute_end,
                    sentence_children,
                )
            )
        else:
            children.append(_gap_segment(source, absolute_start, absolute_end))
        cursor = local_end
    if cursor < len(local):
        children.append(_gap_segment(source, start + cursor, end))
    return tuple(children)


def _build_children(
    source: str,
    start: int,
    end: int,
    structural_children: Sequence[_SpanNode],
    paragraph_segmenter: ParagraphSegmenter,
    sentence_segmenter: SentenceSegmenter,
) -> tuple[Segment, ...]:
    children: list[Segment] = []
    cursor = start
    for node in structural_children:
        if cursor < node.span.start:
            children.extend(
                _plain_segments(
                    source,
                    cursor,
                    node.span.start,
                    paragraph_segmenter,
                    sentence_segmenter,
                )
            )
        nested = _build_children(
            source,
            node.span.start,
            node.span.end,
            node.children,
            paragraph_segmenter,
            sentence_segmenter,
        )
        children.append(
            Segment(
                node.span.kind,
                node.span.start,
                node.span.end,
                nested,
                node.span.label,
                node.span.level,
                node.span.attributes,
            )
        )
        cursor = node.span.end
    if cursor < end:
        children.extend(
            _plain_segments(
                source,
                cursor,
                end,
                paragraph_segmenter,
                sentence_segmenter,
            )
        )
    return tuple(children)


def _callable_id(
    backend: object | None,
    explicit: str | None,
    default: str,
    *,
    name: str,
) -> str:
    if backend is None:
        if explicit is None or explicit == default:
            return default
        raise ValueError(f"{name} requires its corresponding backend callable")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return explicit
    attribute = getattr(backend, "backend_id", None)
    if isinstance(attribute, str) and attribute.strip():
        return attribute
    module = getattr(backend, "__module__", type(backend).__module__)
    qualified = getattr(backend, "__qualname__", type(backend).__qualname__)
    return f"derived:{module}.{qualified}"


def segment_document(
    text: str,
    *,
    sentence_segmenter: SentenceSegmenter | None = None,
    paragraph_segmenter: ParagraphSegmenter | None = None,
    structural_spans: Iterable[StructuralSpan] | None = None,
    structural_mode: StructuralMode | str = StructuralMode.REPLACE,
    structure_profile: StructureProfile | str = StructureProfile.CONSERVATIVE,
    sentence_backend_id: str | None = None,
    paragraph_backend_id: str | None = None,
    structural_backend_id: str | None = None,
) -> DocumentHierarchy:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    mode = _enum(structural_mode, StructuralMode, "structural_mode")
    profile = _enum(structure_profile, StructureProfile, "structure_profile")
    supplied = structural_spans is not None

    if not supplied:
        if structural_backend_id is not None:
            raise ValueError(
                "structural_backend_id is only valid when structural_spans are supplied"
            )
        realised_spans = _builtin_structural_spans(text, profile)
        effective_mode = StructuralMode.REPLACE
        structural_id = f"builtin.legal_structure.v1:{profile.value}"
    else:
        external = _validate_structural_spans(tuple(structural_spans), len(text))
        external_id = _callable_id(
            structural_spans,
            structural_backend_id,
            "caller.structural_spans:derived",
            name="structural_backend_id",
        )
        effective_mode = mode
        if mode is StructuralMode.AUGMENT:
            builtins = _builtin_structural_spans(text, profile)
            realised_spans = (*builtins, *external)
            structural_id = (
                f"builtin.legal_structure.v1:{profile.value}+{external_id}"
            )
        else:
            realised_spans = external
            structural_id = external_id

    validated = _validate_structural_spans(realised_spans, len(text))
    forest = _span_forest(validated)
    paragraph_backend = (
        _default_paragraph_spans
        if paragraph_segmenter is None
        else paragraph_segmenter
    )
    sentence_backend = (
        _legacy_sentence_segmenter
        if sentence_segmenter is None
        else sentence_segmenter
    )
    paragraph_id = _callable_id(
        paragraph_segmenter,
        paragraph_backend_id,
        "builtin.blank_lines.v1",
        name="paragraph_backend_id",
    )
    sentence_id = _callable_id(
        sentence_segmenter,
        sentence_backend_id,
        "lexnlp.sentences.get_sentence_span:v1",
        name="sentence_backend_id",
    )
    children = _build_children(
        text,
        0,
        len(text),
        forest,
        paragraph_backend,
        sentence_backend,
    )
    manifest = HierarchyManifest(
        structural_id,
        effective_mode,
        paragraph_id,
        sentence_id,
        profile,
    )
    return DocumentHierarchy.from_segments(text, children, manifest=manifest)


def iter_document_segments(
    text: str,
    *,
    kind: SegmentKind | str | None = None,
    sentence_segmenter: SentenceSegmenter | None = None,
    paragraph_segmenter: ParagraphSegmenter | None = None,
    structural_spans: Iterable[StructuralSpan] | None = None,
    structural_mode: StructuralMode | str = StructuralMode.REPLACE,
    structure_profile: StructureProfile | str = StructureProfile.CONSERVATIVE,
    sentence_backend_id: str | None = None,
    paragraph_backend_id: str | None = None,
    structural_backend_id: str | None = None,
) -> Iterator[Segment]:
    hierarchy = segment_document(
        text,
        sentence_segmenter=sentence_segmenter,
        paragraph_segmenter=paragraph_segmenter,
        structural_spans=structural_spans,
        structural_mode=structural_mode,
        structure_profile=structure_profile,
        sentence_backend_id=sentence_backend_id,
        paragraph_backend_id=paragraph_backend_id,
        structural_backend_id=structural_backend_id,
    )
    yield from hierarchy.segments(kind)


__all__ = [
    "DocumentHierarchy",
    "HIERARCHY_SCHEMA_VERSION",
    "HierarchyManifest",
    "MAX_HIERARCHY_DEPTH",
    "ParagraphSegmenter",
    "ParagraphSpan",
    "Segment",
    "SegmentAttribute",
    "SegmentKind",
    "SentenceSegmenter",
    "SentenceSpan",
    "StructuralSpan",
    "StructuralMode",
    "StructureProfile",
    "iter_document_segments",
    "segment_document",
]
