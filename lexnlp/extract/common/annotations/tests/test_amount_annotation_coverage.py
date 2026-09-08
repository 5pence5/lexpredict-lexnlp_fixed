"""Coverage tests for AmountAnnotation.get_dictionary_values."""

from __future__ import annotations

from decimal import Decimal

from lexnlp.extract.common.annotations.amount_annotation import AmountAnnotation


class TestAmountAnnotationDictionaryValues:
    def test_values_with_amount_and_text(self) -> None:
        ann = AmountAnnotation(coords=(0, 8), text="100 dollars", value=Decimal("100"))
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Value": "100", "Extracted Entity Text": "100 dollars"}
        }

    def test_zero_value_falls_back_to_empty_string(self) -> None:
        ann = AmountAnnotation(coords=(0, 4), text="none")
        assert ann.value == Decimal("0.0")
        assert ann.get_dictionary_values() == {"tags": {"Extracted Entity Value": "", "Extracted Entity Text": "none"}}

    def test_to_dictionary_merges_values(self) -> None:
        ann = AmountAnnotation(coords=(0, 8), text="100 dollars", value=Decimal("100"))
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 8}
        assert result["tags"]["Extracted Entity Type"] == "amount"
        assert result["tags"]["Extracted Entity Value"] == "100"
        assert result["tags"]["Extracted Entity Text"] == "100 dollars"

    def test_cite_value_parts(self) -> None:
        ann = AmountAnnotation(coords=(0, 8), text="100 dollars", value=Decimal("100"))
        assert ann.get_cite_value_parts() == ["100"]
        empty = AmountAnnotation(coords=(0, 4), text="none")
        assert empty.get_cite_value_parts() == []
