"""Coverage tests for ConstraintAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.constraint_annotation import ConstraintAnnotation


class TestConstraintAnnotationCoverage:
    def test_get_dictionary_values_full(self) -> None:
        ann = ConstraintAnnotation(
            coords=(0, 30),
            text="The party must not exceed the limit",
            constraint="must not exceed",
            pre="The party",
            post="the limit",
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Constraint": "must not exceed",
                "Extracted Entity Pre": "The party",
                "Extracted Entity Post": "the limit",
            }
        }

    def test_get_dictionary_values_defaults_none(self) -> None:
        ann = ConstraintAnnotation(coords=(0, 5))
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Constraint": None,
                "Extracted Entity Pre": None,
                "Extracted Entity Post": None,
            }
        }

    def test_to_dictionary_merges_tags(self) -> None:
        ann = ConstraintAnnotation(
            coords=(2, 18),
            text="shall not assign",
            constraint="shall not assign",
            pre="Licensee",
            post="without consent",
        )
        d = ann.to_dictionary()
        assert d["attrs"] == {"start": 2, "end": 18}
        assert d["tags"]["Extracted Entity Type"] == "constraint"
        assert d["tags"]["Extracted Entity Constraint"] == "shall not assign"
        assert d["tags"]["Extracted Entity Pre"] == "Licensee"
        assert d["tags"]["Extracted Entity Post"] == "without consent"

    def test_get_cite_value_parts_full(self) -> None:
        ann = ConstraintAnnotation(coords=(0, 10), constraint="must", pre="party", post="limit")
        assert ann.get_cite_value_parts() == ["must", "party", "limit"]
