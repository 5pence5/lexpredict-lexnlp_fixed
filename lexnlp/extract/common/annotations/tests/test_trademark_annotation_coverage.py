"""Coverage tests for TrademarkAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.trademark_annotation import TrademarkAnnotation


class TestTrademarkAnnotationDictionaryValues:
    def test_values_with_trademark_and_text(self) -> None:
        ann = TrademarkAnnotation(coords=(0, 5), text="Acme™", trademark="Acme")
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Trademark": "Acme", "Extracted Entity Text": "Acme™"}
        }

    def test_values_with_default_trademark(self) -> None:
        ann = TrademarkAnnotation(coords=(0, 5), text="Acme™")
        assert ann.trademark == ""
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Trademark": "", "Extracted Entity Text": "Acme™"}
        }

    def test_to_dictionary_merges_values(self) -> None:
        ann = TrademarkAnnotation(coords=(0, 5), text="Acme™", trademark="Acme")
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 5}
        assert result["tags"]["Extracted Entity Type"] == "trademark"
        assert result["tags"]["Extracted Entity Trademark"] == "Acme"
        assert result["tags"]["Extracted Entity Text"] == "Acme™"

    def test_cite_value_parts(self) -> None:
        ann = TrademarkAnnotation(coords=(0, 5), text="Acme™", trademark="Acme")
        assert ann.get_cite_value_parts() == ["Acme"]
