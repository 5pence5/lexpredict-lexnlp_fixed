"""Coverage tests for PercentAnnotation.get_dictionary_values."""

from __future__ import annotations

from decimal import Decimal

from lexnlp.extract.common.annotations.percent_annotation import PercentAnnotation


class TestPercentAnnotationDictionaryValues:
    def test_with_sign_includes_sign_tag(self) -> None:
        ann = PercentAnnotation(coords=(0, 5), text="5%", amount=Decimal("5"), sign="+")
        result = ann.get_dictionary_values()
        assert result["tags"]["Extracted Entity Value"] == "5"
        assert result["tags"]["Extracted Entity Text"] == "5%"
        assert result["tags"]["sign"] == "+"
        assert dict(result["tags"]) == {
            "Extracted Entity Value": "5",
            "Extracted Entity Text": "5%",
            "sign": "+",
        }

    def test_without_sign_omits_sign_tag(self) -> None:
        ann = PercentAnnotation(coords=(0, 5), text="5%", amount=Decimal("5"))
        result = ann.get_dictionary_values()
        assert dict(result["tags"]) == {
            "Extracted Entity Value": "5",
            "Extracted Entity Text": "5%",
        }
        assert "sign" not in result["tags"]

    def test_missing_amount_renders_empty_value(self) -> None:
        ann = PercentAnnotation(coords=(0, 5), text="n/a")
        result = ann.get_dictionary_values()
        assert result["tags"]["Extracted Entity Value"] == ""
        assert result["tags"]["Extracted Entity Text"] == "n/a"
        assert "sign" not in result["tags"]
