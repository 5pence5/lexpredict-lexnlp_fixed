"""Coverage tests for ActAnnotation.get_dictionary_values."""

from lexnlp.extract.common.annotations.act_annotation import ActAnnotation


class TestActAnnotationDictionaryCoverage:
    def test_minimal_omits_optional_tags(self) -> None:
        ann = ActAnnotation(
            coords=(0, 10),
            act_name="Clean Air Act",
            text="Clean Air Act",
            ambiguous=None,
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Name": "Clean Air Act",
                "Extracted Entity Text": "Clean Air Act",
            }
        }

    def test_full_includes_all_tags(self) -> None:
        ann = ActAnnotation(
            coords=(0, 30),
            act_name="Clean Air Act",
            section="Section 12",
            year=1970,
            ambiguous=True,
            text="Clean Air Act, Section 12 (1970)",
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Name": "Clean Air Act",
                "Extracted Entity Text": "Clean Air Act, Section 12 (1970)",
                "Extracted Entity Section": "Section 12",
                "Extracted Entity Year": "1970",
                "Extracted Entity Ambiguous": "True",
            }
        }

    def test_ambiguous_false_recorded_as_string(self) -> None:
        ann = ActAnnotation(
            coords=(0, 10),
            act_name="Clean Air Act",
            text="Clean Air Act",
            ambiguous=False,
        )
        assert ann.get_dictionary_values()["tags"]["Extracted Entity Ambiguous"] == "False"

    def test_falsy_year_and_empty_section_omitted(self) -> None:
        ann = ActAnnotation(
            coords=(0, 10),
            act_name="Clean Air Act",
            section="",
            year=None,
            text="Clean Air Act",
            ambiguous=None,
        )
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Name": "Clean Air Act",
                "Extracted Entity Text": "Clean Air Act",
            }
        }
