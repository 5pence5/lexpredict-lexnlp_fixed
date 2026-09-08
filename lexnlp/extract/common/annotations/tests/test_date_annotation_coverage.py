"""Coverage tests for DateAnnotation.get_dictionary_values."""

from __future__ import annotations

from datetime import date

from lexnlp.extract.common.annotations.date_annotation import DateAnnotation


class TestDateAnnotationCoverage:
    def test_get_dictionary_values_with_date(self) -> None:
        ann = DateAnnotation(coords=(0, 12), text="June 1, 2024", date=date(2024, 6, 1))
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Date": "2024-06-01",
                "Extracted Entity Text": "June 1, 2024",
            }
        }

    def test_get_dictionary_values_no_date(self) -> None:
        ann = DateAnnotation(coords=(0, 5), text="sometime")
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Date": "",
                "Extracted Entity Text": "sometime",
            }
        }

    def test_get_dictionary_values_defaults(self) -> None:
        ann = DateAnnotation(coords=(3, 9))
        assert ann.get_dictionary_values() == {"tags": {"Extracted Entity Date": "", "Extracted Entity Text": None}}

    def test_to_dictionary_merges_tags(self) -> None:
        ann = DateAnnotation(coords=(4, 16), text="2024-06-01", date=date(2024, 6, 1))
        d = ann.to_dictionary()
        assert d["attrs"] == {"start": 4, "end": 16}
        assert d["tags"]["Extracted Entity Type"] == "date"
        assert d["tags"]["Extracted Entity Date"] == "2024-06-01"
        assert d["tags"]["Extracted Entity Text"] == "2024-06-01"
