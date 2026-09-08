"""Coverage tests for parsed_text_quality_estimator missing lines."""

from __future__ import annotations

from lexnlp.utils.lines_processing.line_processor import LineOrPhrase
from lexnlp.utils.lines_processing.parsed_text_quality_estimator import (
    LineType,
    ParsedTextQualityEstimator,
    TypedLineOrPhrase,
)


class TestTypedLineRepr:
    def test_repr_contains_type_text_and_ending(self) -> None:
        line = LineOrPhrase("hello", 0)
        line.ending = "\n\n"
        typed = TypedLineOrPhrase.wrap_line(line)
        assert typed.text == "hello"
        assert typed.start == 0
        assert typed.ending == "\n\n"
        assert typed.type == LineType.regular
        assert repr(typed) == "[" + str(LineType.regular) + "] hello->\n\n"

    def test_repr_with_header_type(self) -> None:
        typed = TypedLineOrPhrase()
        typed.text = "Section One"
        typed.ending = "\n"
        typed.type = LineType.header
        assert repr(typed) == "[" + str(LineType.header) + "] Section One->\n"


class TestEmptyTextEstimate:
    def test_estimate_text_empty_returns_zero_probs(self) -> None:
        est = ParsedTextQualityEstimator()
        result = est.estimate_text("")
        assert est.lines == []
        assert result.extra_line_breaks_prob == 0
        assert result.corrupted_prob == 0
        assert result.avg_line_length == est.proc.line_length

    def test_estimate_extra_line_breaks_no_lines_is_noop(self) -> None:
        est = ParsedTextQualityEstimator()
        est.lines = []
        est.estimate.extra_line_breaks_prob = 0
        est.estimate_extra_line_breaks()
        assert est.estimate.extra_line_breaks_prob == 0


class TestHeaderProbEmptyLine:
    def test_empty_string_returns_zero(self) -> None:
        est = ParsedTextQualityEstimator()
        assert est.estimate_line_is_header_prob("") == 0

    def test_whitespace_only_returns_zero(self) -> None:
        est = ParsedTextQualityEstimator()
        assert est.estimate_line_is_header_prob("   ") == 0
        assert est.estimate_line_is_header_prob(" \t ") == 0

    def test_non_empty_short_line_is_header(self) -> None:
        est = ParsedTextQualityEstimator()
        est.estimate.avg_line_length = 100
        assert est.estimate_line_is_header_prob("Section One") == 65
