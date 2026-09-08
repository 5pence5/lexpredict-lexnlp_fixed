"""Coverage tests for ``CopyrightAnnotation.__repr__`` and year-end tagging."""

from __future__ import annotations

from lexnlp.extract.common.annotations.copyright_annotation import CopyrightAnnotation


class TestCopyrightAnnotationRepr:
    def test_repr_prefers_company(self) -> None:
        ant = CopyrightAnnotation(coords=(0, 42), company="Acme", name="nm", text="tx")
        assert repr(ant) == "Acme, (0, 42)"

    def test_repr_falls_back_to_name(self) -> None:
        ant = CopyrightAnnotation(coords=(0, 10), name="nm", text="tx")
        assert repr(ant) == "nm, (0, 10)"

    def test_repr_falls_back_to_text(self) -> None:
        ant = CopyrightAnnotation(coords=(1, 5), text="some text")
        assert repr(ant) == "some text, (1, 5)"

    def test_repr_empty_fields(self) -> None:
        ant = CopyrightAnnotation(coords=(2, 20), year_start=1998, year_end=2001)
        assert repr(ant) == ", (2, 20)"


class TestCopyrightAnnotationDictionaryValues:
    def test_year_end_tag_present(self) -> None:
        ant = CopyrightAnnotation(
            coords=(0, 42),
            company="Acme",
            name="nm",
            text="tx",
            year_start=1998,
            year_end=2001,
        )
        tags = ant.get_dictionary_values()["tags"]
        assert tags["Extracted Entity End"] == 2001
        assert tags == {
            "Extracted Entity Name": "nm",
            "Extracted Entity Text": "tx",
            "Extracted Entity Company": "Acme",
            "Extracted Entity Start": 1998,
            "Extracted Entity End": 2001,
        }

    def test_year_end_tag_absent_without_year_end(self) -> None:
        ant = CopyrightAnnotation(coords=(0, 10), name="nm", text="tx", year_start=1998)
        tags = ant.get_dictionary_values()["tags"]
        assert "Extracted Entity End" not in tags
        assert tags["Extracted Entity Start"] == 1998
