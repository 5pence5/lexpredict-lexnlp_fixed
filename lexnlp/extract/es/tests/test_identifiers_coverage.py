"""Coverage tests for :mod:`lexnlp.extract.es.identifiers` validator branches."""

from __future__ import annotations

from unittest import TestCase

from lexnlp.extract.es.identifiers import (
    _cif_is_valid,
    _nie_is_valid,
    get_cif_annotations,
)


class TestNieValidatorEdges(TestCase):
    def test_short_canonical_is_invalid(self) -> None:
        self.assertFalse(_nie_is_valid("X123"))
        self.assertFalse(_nie_is_valid(""))

    def test_non_digit_body_is_invalid(self) -> None:
        # Prefix X is valid and length is 9, but the folded body
        # "0" + "123A567" contains a letter.
        self.assertFalse(_nie_is_valid("X123A567L"))


class TestCifValidatorEdges(TestCase):
    def test_short_canonical_is_invalid(self) -> None:
        self.assertFalse(_cif_is_valid("A1234"))
        self.assertFalse(_cif_is_valid(""))

    def test_alpha_only_head(self) -> None:
        # Heads P/Q/R/S/N/W must use the alpha control character.
        self.assertTrue(_cif_is_valid("P1234567D"))
        self.assertFalse(_cif_is_valid("P12345674"))

    def test_either_form_head_accepts_both(self) -> None:
        self.assertTrue(_cif_is_valid("C12345674"))
        self.assertTrue(_cif_is_valid("C1234567D"))

    def test_either_form_head_rejects_wrong_controls(self) -> None:
        self.assertFalse(_cif_is_valid("C12345670"))
        self.assertFalse(_cif_is_valid("C1234567A"))


class TestCifAnnotationsBranches(TestCase):
    def test_alpha_only_head_extracted_from_text(self) -> None:
        results = list(get_cif_annotations("La sociedad P1234567D firmó."))
        self.assertEqual(1, len(results))
        self.assertEqual("P1234567D", results[0].value)
        self.assertEqual("cif", results[0].kind)
        self.assertEqual((12, 21), results[0].coords)

    def test_alpha_only_head_rejects_numeric_control_in_text(self) -> None:
        self.assertEqual([], list(get_cif_annotations("La sociedad P12345674 firmó.")))

    def test_either_form_head_both_controls_extracted(self) -> None:
        results = list(get_cif_annotations("Sociedad C12345674 y C1234567D."))
        self.assertEqual(["C12345674", "C1234567D"], [r.value for r in results])
        self.assertTrue(all(r.kind == "cif" for r in results))

    def test_wrong_controls_yield_nothing(self) -> None:
        self.assertEqual([], list(get_cif_annotations("Sociedad C12345670 firmó.")))
        self.assertEqual([], list(get_cif_annotations("Sociedad C1234567A firmó.")))
