"""Coverage tests for DurationAnnotation.get_dictionary_values."""

from __future__ import annotations

from decimal import Decimal

from lexnlp.extract.common.annotations.duration_annotation import DurationAnnotation


class TestDurationAnnotationDictionaryValues:
    def test_values_with_amount_and_text(self) -> None:
        ann = DurationAnnotation(
            coords=(0, 8),
            text="3 years",
            amount=Decimal("3"),
            duration_type="years",
        )
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Value": "3", "Extracted Entity Text": "3 years"}
        }

    def test_values_without_amount_uses_empty_string(self) -> None:
        ann = DurationAnnotation(coords=(0, 5), text="a while")
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Value": "", "Extracted Entity Text": "a while"}
        }

    def test_to_dictionary_merges_values(self) -> None:
        ann = DurationAnnotation(
            coords=(0, 8),
            text="3 years",
            amount=Decimal("3"),
            duration_type="years",
        )
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 8}
        assert result["tags"]["Extracted Entity Type"] == "duration"
        assert result["tags"]["Extracted Entity Value"] == "3"
        assert result["tags"]["Extracted Entity Text"] == "3 years"

    def test_cite_value_parts(self) -> None:
        ann = DurationAnnotation(
            coords=(0, 8),
            text="3 years",
            amount=Decimal("3"),
            duration_type="years",
        )
        assert ann.get_cite_value_parts() == ["3", "years"]
