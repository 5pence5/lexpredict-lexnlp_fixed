"""Coverage tests for DistanceAnnotation.get_dictionary_values."""

from __future__ import annotations

from decimal import Decimal

from lexnlp.extract.common.annotations.distance_annotation import DistanceAnnotation


class TestDistanceAnnotationCoverage:
    def test_get_dictionary_values_full(self) -> None:
        ann = DistanceAnnotation(
            coords=(0, 6),
            text="101 km",
            amount=Decimal("101"),
            distance_type="km",
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Value": "101",
                "Extracted Entity Text": "101 km",
            }
        }

    def test_get_dictionary_values_defaults(self) -> None:
        ann = DistanceAnnotation(coords=(0, 3))
        assert ann.get_dictionary_values() == {"tags": {"Extracted Entity Value": "", "Extracted Entity Text": None}}

    def test_to_dictionary_merges_tags(self) -> None:
        ann = DistanceAnnotation(
            coords=(7, 15),
            text="500 miles",
            amount=Decimal("500"),
            distance_type="miles",
        )
        d = ann.to_dictionary()
        assert d["attrs"] == {"start": 7, "end": 15}
        assert d["tags"]["Extracted Entity Type"] == "distance"
        assert d["tags"]["Extracted Entity Value"] == "500"
        assert d["tags"]["Extracted Entity Text"] == "500 miles"

    def test_get_cite_value_parts_full(self) -> None:
        ann = DistanceAnnotation(coords=(0, 6), amount=Decimal("101"), distance_type="km")
        assert ann.get_cite_value_parts() == ["101", "km"]
