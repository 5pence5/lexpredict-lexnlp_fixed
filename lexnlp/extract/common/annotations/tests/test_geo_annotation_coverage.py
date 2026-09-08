"""Coverage tests for GeoAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.geo_annotation import GeoAnnotation


class TestGeoAnnotationDictionaryValues:
    def test_with_year_includes_year_tag(self) -> None:
        ann = GeoAnnotation(coords=(0, 10), text="Berlin", name="Berlin", year=2020)
        result = ann.get_dictionary_values()
        assert result["tags"]["Extracted Entity Name"] == "Berlin"
        assert result["tags"]["Extracted Entity Text"] == "Berlin"
        assert result["tags"]["year"] == 2020
        assert dict(result["tags"]) == {
            "Extracted Entity Name": "Berlin",
            "Extracted Entity Text": "Berlin",
            "year": 2020,
        }

    def test_without_year_omits_year_tag(self) -> None:
        ann = GeoAnnotation(coords=(0, 10), text="Berlin", name="Berlin")
        result = ann.get_dictionary_values()
        assert dict(result["tags"]) == {
            "Extracted Entity Name": "Berlin",
            "Extracted Entity Text": "Berlin",
        }
        assert "year" not in result["tags"]

    def test_falsy_year_omits_year_tag(self) -> None:
        ann = GeoAnnotation(coords=(0, 10), text="Berlin", name="Berlin", year=0)
        result = ann.get_dictionary_values()
        assert "year" not in result["tags"]
        assert result["tags"]["Extracted Entity Name"] == "Berlin"
