"""Coverage tests for LawAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.law_annotation import LawAnnotation


class TestLawAnnotationDictionaryValues:
    def test_values_with_text(self) -> None:
        ann = LawAnnotation(coords=(0, 10), name="Clean Air Act", text="Clean Air Act text")
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Name": "Clean Air Act",
                "Extracted Entity Text": "Clean Air Act text",
            }
        }

    def test_values_falls_back_to_name_when_text_empty(self) -> None:
        ann = LawAnnotation(coords=(0, 10), name="Clean Air Act")
        assert ann.text == ""
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Name": "Clean Air Act",
                "Extracted Entity Text": "Clean Air Act",
            }
        }

    def test_to_dictionary_merges_values(self) -> None:
        ann = LawAnnotation(coords=(0, 10), name="Clean Air Act", text="Clean Air Act text")
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 10}
        assert result["tags"]["Extracted Entity Type"] == "law"
        assert result["tags"]["Extracted Entity Name"] == "Clean Air Act"
        assert result["tags"]["Extracted Entity Text"] == "Clean Air Act text"

    def test_cite_value_parts(self) -> None:
        ann = LawAnnotation(coords=(0, 10), name="Clean Air Act", text="Clean Air Act text")
        assert ann.get_cite_value_parts() == ["Clean Air Act"]
