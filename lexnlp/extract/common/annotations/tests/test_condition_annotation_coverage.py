"""Coverage tests for ConditionAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.condition_annotation import ConditionAnnotation


class TestConditionAnnotationCoverage:
    def test_get_dictionary_values_full(self) -> None:
        ann = ConditionAnnotation(
            coords=(0, 20),
            text="if party defaults",
            condition="if party defaults",
            pre="In the event that",
            post="then penalties apply",
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Condition": "if party defaults",
                "Extracted Entity Pre": "In the event that",
                "Extracted Entity Post": "then penalties apply",
            }
        }

    def test_get_dictionary_values_defaults_none(self) -> None:
        ann = ConditionAnnotation(coords=(0, 5))
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Condition": None,
                "Extracted Entity Pre": None,
                "Extracted Entity Post": None,
            }
        }

    def test_to_dictionary_merges_tags(self) -> None:
        ann = ConditionAnnotation(
            coords=(5, 25),
            text="unless cured",
            condition="unless cured",
            pre="prior to termination",
            post="remedy applies",
        )
        d = ann.to_dictionary()
        assert d["attrs"] == {"start": 5, "end": 25}
        assert d["tags"]["Extracted Entity Type"] == "condition"
        assert d["tags"]["Extracted Entity Condition"] == "unless cured"
        assert d["tags"]["Extracted Entity Pre"] == "prior to termination"
        assert d["tags"]["Extracted Entity Post"] == "remedy applies"

    def test_get_cite_value_parts_full(self) -> None:
        ann = ConditionAnnotation(coords=(0, 10), condition="if", pre="when", post="then")
        assert ann.get_cite_value_parts() == ["if", "when", "then"]
