"""Coverage tests for :class:`RatioAnnotation.get_dictionary_values`."""

from __future__ import annotations

from decimal import Decimal

from lexnlp.extract.common.annotations.ratio_annotation import RatioAnnotation


class TestGetDictionaryValues:
    def test_all_fields(self) -> None:
        ann = RatioAnnotation(
            coords=(0, 3),
            text="1:2",
            left=Decimal("1"),
            right=Decimal("2"),
            ratio=Decimal("0.5"),
        )
        values = ann.get_dictionary_values()
        assert values["tags"]["Extracted Entity Ratio"] == "0.5"
        assert values["tags"]["Extracted Entity Text"] == "1:2"
        assert values["tags"]["left"] == "1"
        assert values["tags"]["right"] == "2"

    def test_left_only(self) -> None:
        ann = RatioAnnotation(coords=(0, 3), text="1:?", left=Decimal("1"), right=None, ratio=None)
        values = ann.get_dictionary_values()
        assert values["tags"]["left"] == "1"
        assert "right" not in values["tags"]
        assert values["tags"]["Extracted Entity Ratio"] == ""

    def test_right_only(self) -> None:
        ann = RatioAnnotation(coords=(0, 3), text="?:2", left=None, right=Decimal("2"), ratio=None)
        values = ann.get_dictionary_values()
        assert "left" not in values["tags"]
        assert values["tags"]["right"] == "2"

    def test_no_optionals(self) -> None:
        ann = RatioAnnotation(coords=(0, 1), text="x")
        values = ann.get_dictionary_values()
        assert values["tags"] == {
            "Extracted Entity Ratio": "",
            "Extracted Entity Text": "x",
        }

    def test_to_dictionary_merges_ratio_tags(self) -> None:
        ann = RatioAnnotation(
            coords=(0, 3),
            text="1:2",
            left=Decimal("1"),
            right=Decimal("2"),
            ratio=Decimal("0.5"),
        )
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 3}
        assert result["tags"]["Extracted Entity Type"] == "ratio"
        assert result["tags"]["left"] == "1"
        assert result["tags"]["right"] == "2"
        assert result["tags"]["Extracted Entity Ratio"] == "0.5"
