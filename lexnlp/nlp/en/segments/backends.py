"""Optional sentence-boundary backends for lossless document segmentation.

The hierarchy and chunking APIs do not depend on a particular neural model.
This module adapts models with the wtpsplit SaT.split interface without
importing wtpsplit or downloading model weights. Applications construct and
pin the model themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable


SentenceSpan: TypeAlias = tuple[int, int, str]


@runtime_checkable
class SplitModel(Protocol):
    """Structural protocol implemented by Segment Any Text models."""

    def split(self, text: str, **kwargs: Any) -> Iterable[str]:
        """Return consecutive pieces of text without changing characters."""


def parts_to_spans(text: str, parts: Iterable[str]) -> tuple[SentenceSpan, ...]:
    """Convert consecutive, lossless text pieces into source-coordinate spans."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    cursor = 0
    spans: list[SentenceSpan] = []
    for index, part in enumerate(parts):
        if not isinstance(part, str):
            raise TypeError(f"segment {index} is not a string")
        if not part:
            raise ValueError(f"segment {index} is empty")
        end = cursor + len(part)
        if text[cursor:end] != part:
            raise ValueError(
                "sentence backend did not preserve the input exactly at "
                f"segment {index}, offset {cursor}"
            )
        spans.append((cursor, end, part))
        cursor = end

    if cursor != len(text):
        raise ValueError(
            "sentence backend did not cover the complete input: "
            f"covered {cursor} of {len(text)} characters"
        )
    return tuple(spans)


@dataclass(slots=True)
class SaTSentenceSegmenter:
    """Adapt an instantiated Segment Any Text model to exact LexNLP spans.

    backend_id must identify the caller-pinned model revision and split
    configuration. LexNLP records it in hierarchy and chunk manifests.
    """

    model: SplitModel
    backend_id: str
    split_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(getattr(self.model, "split", None)):
            raise TypeError("model must provide a callable split(text, **kwargs) method")
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError(
                "backend_id must be a non-empty caller-pinned model and config identity"
            )
        self.split_kwargs = dict(self.split_kwargs)

    def __call__(self, text: str) -> Iterator[SentenceSpan]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return

        kwargs = {"split_on_input_newlines": False}
        kwargs.update(self.split_kwargs)
        parts = self.model.split(text, **kwargs)
        if isinstance(parts, str):
            raise TypeError("model.split must return an iterable of strings, not a string")
        yield from parts_to_spans(text, parts)


def legacy_sentence_segmenter(text: str) -> Iterator[SentenceSpan]:
    """Expose LexNLP's existing legal-tuned Punkt model through the new seam."""

    from lexnlp.nlp.en.segments.sentences import get_sentence_span

    yield from get_sentence_span(text)


__all__ = [
    "SaTSentenceSegmenter",
    "SentenceSpan",
    "SplitModel",
    "legacy_sentence_segmenter",
    "parts_to_spans",
]
