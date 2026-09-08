"""Tests for :mod:`lexnlp.extract.es.identifiers`."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import dataclasses
from unittest import TestCase

from lexnlp.extract.es.identifiers import (
    EsIdentifierMatch,
    _cif_is_valid,
    _dni_is_valid,
    _nie_is_valid,
    get_cif_annotations,
    get_dni_annotations,
    get_identifier_annotations,
    get_nie_annotations,
    get_nif_annotations,
)


class TestDniValidator(TestCase):
    def test_valid_dni(self):
        self.assertTrue(_dni_is_valid("12345678Z"))

    def test_invalid_check_letter(self):
        self.assertFalse(_dni_is_valid("12345678A"))

    def test_invalid_length(self):
        self.assertFalse(_dni_is_valid("1234567Z"))

    def test_lowercase_letter_rejected(self):
        # Canonical form is uppercase; the public extractors normalise
        # before calling this validator.
        self.assertFalse(_dni_is_valid("12345678z"))


class TestNieValidator(TestCase):
    def test_valid_x_prefix(self):
        self.assertTrue(_nie_is_valid("X1234567L"))

    def test_valid_y_prefix(self):
        self.assertTrue(_nie_is_valid("Y1234567X"))

    def test_invalid_prefix(self):
        # ``A`` is not a valid NIE prefix.
        self.assertFalse(_nie_is_valid("A1234567L"))

    def test_invalid_check_letter(self):
        self.assertFalse(_nie_is_valid("X1234567A"))


class TestCifValidator(TestCase):
    def test_valid_cif_alpha_control(self):
        # ``A`` letter forces numeric control character.
        self.assertTrue(_cif_is_valid("A12345674"))

    def test_invalid_cif(self):
        self.assertFalse(_cif_is_valid("A12345670"))

    def test_invalid_letter(self):
        # ``T`` is not a CIF leading letter.
        self.assertFalse(_cif_is_valid("T12345674"))


class TestGetDniAnnotations(TestCase):
    def test_extracts_dni(self):
        text = "El propietario presenta el DNI 12345678Z para el contrato."
        results = list(get_dni_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual("12345678Z", results[0].value)
        self.assertEqual("dni", results[0].kind)
        self.assertEqual("es", results[0].locale)

    def test_does_not_extract_invalid_dni(self):
        text = "Este DNI 12345678A es inválido."
        results = list(get_dni_annotations(text))
        self.assertEqual([], results)

    def test_to_dictionary(self):
        text = "DNI: 12345678Z."
        match = next(get_dni_annotations(text))
        d = match.to_dictionary()
        self.assertEqual("dni", d["record_type"])
        self.assertEqual("12345678Z", d["value"])
        self.assertEqual("es", d["locale"])


class TestGetNieAnnotations(TestCase):
    def test_extracts_nie(self):
        text = "El residente extranjero presenta NIE X1234567L hoy."
        results = list(get_nie_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual("X1234567L", results[0].value)
        self.assertEqual("nie", results[0].kind)


class TestGetCifAnnotations(TestCase):
    def test_extracts_cif(self):
        text = "La sociedad con CIF A12345674 firmó el acuerdo."
        results = list(get_cif_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual("A12345674", results[0].value)
        self.assertEqual("cif", results[0].kind)


class TestGetNifAnnotations(TestCase):
    def test_nif_includes_dni_and_nie(self):
        text = "DNI 12345678Z y NIE X1234567L."
        results = list(get_nif_annotations(text))
        kinds = {r.kind for r in results}
        self.assertEqual({"nif"}, kinds)
        values = {r.value for r in results}
        self.assertEqual({"12345678Z", "X1234567L"}, values)


class TestGetIdentifierAnnotations(TestCase):
    def test_extracts_all_kinds(self):
        text = (
            "El consumidor con DNI 12345678Z y la sociedad CIF A12345674. El residente con NIE X1234567L también firma."
        )
        results = list(get_identifier_annotations(text))
        kinds = sorted(r.kind for r in results)
        self.assertEqual(["cif", "dni", "nie"], kinds)

    def test_empty_text_yields_nothing(self):
        self.assertEqual([], list(get_identifier_annotations("")))

    def test_dataclass_is_frozen(self):
        m = EsIdentifierMatch(kind="dni", value="12345678Z", surface="12345678Z", coords=(0, 9))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            m.value = "00000000T"


class TestEsIdentifiersPackageReexports(TestCase):
    """Lock the public surface of ``lexnlp.extract.es`` against drift."""

    def test_reexports_are_identical_objects(self):
        from lexnlp.extract import es as es_pkg
        from lexnlp.extract.es import identifiers as identifiers_mod

        for name in (
            "EsIdentifierMatch",
            "get_cif_annotations",
            "get_dni_annotations",
            "get_identifier_annotations",
            "get_nie_annotations",
            "get_nif_annotations",
        ):
            self.assertIs(getattr(es_pkg, name), getattr(identifiers_mod, name))
            self.assertIn(name, es_pkg.__all__)
