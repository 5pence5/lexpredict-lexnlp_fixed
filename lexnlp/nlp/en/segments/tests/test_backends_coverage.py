"""Coverage tests for sentence-boundary segmentation backends."""

from __future__ import annotations

from typing import Any

import pytest

from lexnlp.nlp.en.segments.backends import (
    SaTSentenceSegmenter,
    legacy_sentence_segmenter,
    parts_to_spans,
)
from lexnlp.nlp.en.segments.sentences import get_sentence_span


class _FakeSplitModel:
    def __init__(self, parts: Any) -> None:
        self._parts = parts

    def split(self, text: str, **kwargs: Any) -> Any:
        return self._parts


def _make_segmenter(parts: Any) -> SaTSentenceSegmenter:
    return SaTSentenceSegmenter(model=_FakeSplitModel(parts), backend_id="sat-test-v1")


def test_parts_to_spans_rejects_non_string_text() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        parts_to_spans(123, ["hello"])  # type: ignore[arg-type]


def test_parts_to_spans_rejects_empty_segment() -> None:
    with pytest.raises(ValueError, match=r"segment 1 is empty"):
        parts_to_spans("abc", ["ab", ""])


def test_parts_to_spans_rejects_incomplete_coverage() -> None:
    with pytest.raises(ValueError, match="covered 3 of 6 characters"):
        parts_to_spans("abcdef", ["abc"])


def test_segmenter_rejects_model_without_split() -> None:
    with pytest.raises(TypeError, match="model must provide a callable split"):
        SaTSentenceSegmenter(model=object(), backend_id="sat-test-v1")  # type: ignore[arg-type]


def test_segmenter_rejects_non_string_text() -> None:
    segmenter = _make_segmenter(["hi"])
    assert list(segmenter("hi")) == [(0, 2, "hi")]
    with pytest.raises(TypeError, match="text must be a string"):
        list(segmenter(123))  # type: ignore[arg-type]


def test_segmenter_rejects_string_split_result() -> None:
    segmenter = _make_segmenter("hello")
    with pytest.raises(TypeError, match="must return an iterable of strings, not a string"):
        list(segmenter("hello"))


def test_legacy_sentence_segmenter_delegates_to_punkt() -> None:
    text = "Hello world. This is a test."
    spans = list(legacy_sentence_segmenter(text))
    assert spans == list(get_sentence_span(text))
    assert len(spans) >= 2
    for start, end, part in spans:
        assert 0 <= start <= end <= len(text)
        assert text[start:end] == part
    assert spans[0][0] == 0
    assert "Hello" in spans[0][2]
