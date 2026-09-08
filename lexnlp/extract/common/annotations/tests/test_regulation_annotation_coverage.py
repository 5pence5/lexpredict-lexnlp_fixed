"""Coverage tests for RegulationAnnotation.get_dictionary_values source branch."""

from __future__ import annotations

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation


class TestRegulationAnnotationCoverage:
    def test_get_dictionary_values_with_source(self) -> None:
        ann = RegulationAnnotation(
            coords=(0, 30),
            locale="en",
            name="17 CFR 240.10b-5",
            text="17 CFR 240.10b-5",
            source="SEC",
            country="US",
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "External Reference Issuing Country": "US",
                "External Reference Text": "17 CFR 240.10b-5",
                "Extracted Entity Text": "17 CFR 240.10b-5",
                "External Reference Source": "SEC",
            }
        }

    def test_get_dictionary_values_without_source(self) -> None:
        ann = RegulationAnnotation(coords=(0, 10), name="S 12", country="DE")
        assert ann.get_dictionary_values() == {
            "tags": {
                "External Reference Issuing Country": "DE",
                "External Reference Text": "S 12",
                "Extracted Entity Text": "S 12",
            }
        }

    def test_get_dictionary_values_text_falls_back_to_name(self) -> None:
        ann = RegulationAnnotation(coords=(0, 10), name="S 12", source="BaFin", country="DE")
        values = ann.get_dictionary_values()
        assert values["tags"]["Extracted Entity Text"] == "S 12"
        assert values["tags"]["External Reference Source"] == "BaFin"
