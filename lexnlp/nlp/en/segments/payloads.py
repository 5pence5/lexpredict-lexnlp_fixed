"""Versioned, budget-checked payloads for embedding document chunks.

Chunk packing budgets the exact source slice. Retrieval systems commonly add
headings or repeated table headers before embedding that slice, so this module
provides a separate final-serialization contract: the caller's actual token
counter is run over the exact payload text and overflow is never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Iterable, TypeAlias

from lexnlp.nlp.en.segments.chunks import DocumentChunk
from lexnlp.nlp.en.segments.hierarchy import SegmentKind


EMBEDDING_PAYLOAD_SERIALIZATION_VERSION = "lexnlp.embedding_payload.v1"
TokenCounter: TypeAlias = Callable[[str], int]
_CONTEXT_ROLES = frozenset({"heading", "table_header"})
_ANCESTRY_KINDS = frozenset(
    {SegmentKind.SECTION, SegmentKind.CLAUSE, SegmentKind.TABLE}
)


def _non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ContextFragment:
    """A source-traceable heading or table header prepended to a chunk."""

    role: str
    text: str
    segment_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _CONTEXT_ROLES:
            raise ValueError("role must be 'heading' or 'table_header'")
        _non_empty_string("text", self.text)
        _non_empty_string("segment_id", self.segment_id)


@dataclass(frozen=True)
class EmbeddingPayload:
    """The exact, versioned string and tokenizer evidence sent for embedding."""

    chunk_id: str
    serialization_version: str
    tokenizer_id: str
    max_tokens: int
    token_count: int
    context_fragments: tuple[ContextFragment, ...]
    source_text: str
    text: str
    text_sha256: str

    def __post_init__(self) -> None:
        _non_empty_string("chunk_id", self.chunk_id)
        if self.serialization_version != EMBEDDING_PAYLOAD_SERIALIZATION_VERSION:
            raise ValueError(
                "serialization_version must be "
                f"{EMBEDDING_PAYLOAD_SERIALIZATION_VERSION!r}"
            )
        _non_empty_string("tokenizer_id", self.tokenizer_id)
        _positive_integer("max_tokens", self.max_tokens)
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")
        if self.token_count > self.max_tokens:
            raise ValueError("token_count cannot exceed max_tokens")
        object.__setattr__(self, "context_fragments", tuple(self.context_fragments))
        if not all(
            isinstance(fragment, ContextFragment)
            for fragment in self.context_fragments
        ):
            raise TypeError("context_fragments must contain ContextFragment values")
        if not isinstance(self.source_text, str) or not isinstance(self.text, str):
            raise TypeError("source_text and text must be strings")
        if not isinstance(self.text_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.text_sha256
        ):
            raise ValueError("text_sha256 must be a lowercase SHA-256 digest")

        if self.context_fragments:
            expected = (
                "\n".join(fragment.text for fragment in self.context_fragments)
                + "\n\n"
                + self.source_text
            )
        else:
            expected = self.source_text
        if self.text != expected:
            raise ValueError("text does not match the versioned payload serialization")
        actual_digest = hashlib.sha256(
            self.text.encode("utf-8", "surrogatepass")
        ).hexdigest()
        if self.text_sha256 != actual_digest:
            raise ValueError("text_sha256 does not match the exact payload text")

    @property
    def payload_metadata_sha256(self) -> str:
        """Canonical identity for exact text, context, tokenizer and budget evidence."""

        value = {
            "chunk_id": self.chunk_id,
            "context_fragments": [
                {
                    "role": fragment.role,
                    "segment_id": fragment.segment_id,
                    "text": fragment.text,
                }
                for fragment in self.context_fragments
            ],
            "max_tokens": self.max_tokens,
            "serialization_version": self.serialization_version,
            "source_text_sha256": hashlib.sha256(
                self.source_text.encode("utf-8", "surrogatepass")
            ).hexdigest(),
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
            "tokenizer_id": self.tokenizer_id,
        }
        serialised = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", "surrogatepass")
        return hashlib.sha256(serialised).hexdigest()

    @property
    def payload_id(self) -> str:
        """Deterministic identity for the complete validated payload evidence."""

        return f"{self.chunk_id}:payload:sha256:{self.payload_metadata_sha256}"


class PayloadBudgetExceeded(ValueError):
    """Raised when final serialized embedding text exceeds its token budget."""

    def __init__(self, actual: int, maximum: int, chunk_id: str):
        self.actual = actual
        self.maximum = maximum
        self.chunk_id = chunk_id
        super().__init__(
            f"embedding payload {chunk_id} uses {actual} tokens; budget is {maximum}"
        )


def _ancestry_fragments(chunk: DocumentChunk) -> tuple[ContextFragment, ...]:
    fragments: list[ContextFragment] = []
    for reference in chunk.content_provenance.segments:
        if (
            reference.kind not in _ANCESTRY_KINDS
            or not reference.label
            or reference.start >= chunk.new_content_start
            or reference.end < chunk.end
        ):
            continue
        fragments.append(
            ContextFragment(
                role="heading",
                text=reference.label,
                segment_id=reference.segment_id,
            )
        )
    return tuple(fragments)


def _ordered_fragments(
    chunk: DocumentChunk,
    fragments: Iterable[ContextFragment],
) -> tuple[ContextFragment, ...]:
    content_references = {
        reference.segment_id: reference
        for reference in chunk.content_provenance.segments
    }
    content_order = {
        reference.segment_id: index
        for index, reference in enumerate(chunk.content_provenance.segments)
    }
    candidates: list[ContextFragment] = []
    for fragment in fragments:
        if not isinstance(fragment, ContextFragment):
            raise TypeError("context_fragments must contain ContextFragment values")
        if fragment.segment_id not in content_order:
            raise ValueError(
                "context fragment segment_id must occur in content_provenance; "
                "overlap-only provenance cannot own embedding context"
            )
        owner = content_references[fragment.segment_id]
        if fragment.role == "table_header" and owner.kind is not SegmentKind.TABLE:
            raise ValueError("table_header context must be owned by a TABLE segment")
        if fragment.role == "heading" and owner.kind not in _ANCESTRY_KINDS:
            raise ValueError(
                "heading context must be owned by a SECTION, CLAUSE, or TABLE segment"
            )
        candidates.append(fragment)

    role_order = {"heading": 0, "table_header": 1}
    ordered = sorted(
        candidates,
        key=lambda fragment: (
            content_order[fragment.segment_id],
            role_order[fragment.role],
            fragment.text,
            fragment.segment_id,
        ),
    )
    selected: dict[tuple[str, str], ContextFragment] = {}
    for fragment in ordered:
        selected[(fragment.role, fragment.text)] = fragment
    return tuple(
        sorted(
            selected.values(),
            key=lambda fragment: (
                content_order[fragment.segment_id],
                role_order[fragment.role],
                fragment.text,
                fragment.segment_id,
            ),
        )
    )


def render_embedding_payload(
    chunk: DocumentChunk,
    *,
    token_counter: TokenCounter,
    tokenizer_id: str,
    max_tokens: int,
    context_fragments: Iterable[ContextFragment] = (),
    include_ancestry_labels: bool = True,
) -> EmbeddingPayload:
    """Render and strictly budget the final text sent to an embedding model.

    The serializer never trims context or source text. Callers that exceed the
    budget must reserve more headroom or rechunk and retry.
    """

    if not isinstance(chunk, DocumentChunk):
        raise TypeError("chunk must be a DocumentChunk")
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")
    selected_tokenizer_id = _non_empty_string("tokenizer_id", tokenizer_id)
    selected_max_tokens = _positive_integer("max_tokens", max_tokens)
    if not isinstance(include_ancestry_labels, bool):
        raise TypeError("include_ancestry_labels must be a boolean")

    supplied = tuple(context_fragments)
    candidates = (
        _ancestry_fragments(chunk) + supplied
        if include_ancestry_labels
        else supplied
    )
    ordered = _ordered_fragments(chunk, candidates)
    if ordered:
        text = "\n".join(fragment.text for fragment in ordered) + "\n\n" + chunk.text
    else:
        text = chunk.text

    token_count = token_counter(text)
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("token_counter must return an integer")
    if token_count < 0:
        raise ValueError("token_counter cannot return a negative value")
    if token_count > selected_max_tokens:
        raise PayloadBudgetExceeded(token_count, selected_max_tokens, chunk.chunk_id)

    return EmbeddingPayload(
        chunk_id=chunk.chunk_id,
        serialization_version=EMBEDDING_PAYLOAD_SERIALIZATION_VERSION,
        tokenizer_id=selected_tokenizer_id,
        max_tokens=selected_max_tokens,
        token_count=token_count,
        context_fragments=ordered,
        source_text=chunk.text,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(),
    )


__all__ = [
    "EMBEDDING_PAYLOAD_SERIALIZATION_VERSION",
    "ContextFragment",
    "EmbeddingPayload",
    "PayloadBudgetExceeded",
    "render_embedding_payload",
]
