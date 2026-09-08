"""Coverage tests for :mod:`lexnlp.nlp.en.segments.paragraphs` missing lines."""

import pytest

import lexnlp.nlp.en.segments.paragraphs as paragraphs
from lexnlp.nlp.en.segments.paragraphs import get_paragraph_spans, splitlines_with_spans


def test_splitlines_with_spans_none_returns_empty() -> None:
    lines, spans = splitlines_with_spans(None)  # type: ignore[arg-type]
    assert lines == []
    assert spans == []


def test_splitlines_with_spans_pins_offsets() -> None:
    lines, spans = splitlines_with_spans("ab\ncd")
    assert lines == ["ab", "cd"]
    assert spans == [(0, 3), (3, 5)]


def test_empty_text_reraises_value_error() -> None:
    with pytest.raises(ValueError, match="0 sample"):
        list(get_paragraph_spans(""))


class _FeatureMismatchModel:
    def predict_proba(self, features):  # noqa: ANN001, ANN202
        raise ValueError("Number of features of the model must match the input. Model n_features is 361.")


def test_feature_mismatch_falls_back_to_whole_text(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "First paragraph line one.\nSecond line here.\n\nSecond paragraph here.\n"
    monkeypatch.setattr(paragraphs, "PARAGRAPH_SEGMENTER_MODEL", _FeatureMismatchModel())
    assert list(get_paragraph_spans(text)) == [(0, len(text), text)]


class _OtherErrorModel:
    def predict_proba(self, features):  # noqa: ANN001, ANN202
        raise ValueError("some unrelated model failure")


def test_other_value_error_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paragraphs, "PARAGRAPH_SEGMENTER_MODEL", _OtherErrorModel())
    with pytest.raises(ValueError, match="unrelated model failure"):
        list(get_paragraph_spans("Some text here.\n"))
