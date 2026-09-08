"""Tests for :mod:`lexnlp.extract.de.identifiers`."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import dataclasses
from unittest import TestCase

from lexnlp.extract.de.identifiers import (
    DeIdentifierMatch,
    _steuer_idnr_is_valid,
    _ust_idnr_is_valid,
    get_hrb_annotations,
    get_identifier_annotations,
    get_steuer_idnr_annotations,
    get_ust_idnr_annotations,
)

# ``47036892816`` is a valid Steuer-IdNr published as a BZSt test vector
# (digit ``8`` repeats once, ISO 7064 MOD 11,10 check digit is 6).
VALID_IDNR = "47036892816"
# ``DE136695976`` is the published USt-IdNr of the Bundesfinanzhof.
VALID_UST = "DE136695976"


class TestSteuerIdnrValidator(TestCase):
    def test_valid_idnr(self):
        self.assertTrue(_steuer_idnr_is_valid(VALID_IDNR))

    def test_invalid_check_digit(self):
        # Flip the check digit from 6 -> 5.
        self.assertFalse(_steuer_idnr_is_valid(VALID_IDNR[:-1] + "5"))

    def test_all_zeros_rejected(self):
        self.assertFalse(_steuer_idnr_is_valid("00000000000"))

    def test_wrong_length(self):
        self.assertFalse(_steuer_idnr_is_valid("1234567890"))

    def test_repeat_digit_invariant_violated(self):
        # Three different digits repeating - violates the BZSt invariant.
        # "1122334560" has 1, 2 and 3 each appearing twice.
        self.assertFalse(_steuer_idnr_is_valid("11223345607"))

    def test_leading_zero_rejected(self):
        # BZSt rule: the Steuer-IdNr cannot start with a zero.
        self.assertFalse(_steuer_idnr_is_valid("0" + VALID_IDNR[1:]))


class TestUstIdnrValidator(TestCase):
    def test_valid_ust(self):
        self.assertTrue(_ust_idnr_is_valid(VALID_UST[2:]))  # strip ``DE``

    def test_invalid_check_digit(self):
        digits = VALID_UST[2:]
        bad = digits[:-1] + str((int(digits[-1]) + 1) % 10)
        self.assertFalse(_ust_idnr_is_valid(bad))

    def test_wrong_length(self):
        self.assertFalse(_ust_idnr_is_valid("12345678"))


class TestSteuerIdnrAnnotations(TestCase):
    def test_extracts_idnr(self):
        text = f"Meine Steueridentifikationsnummer ist {VALID_IDNR}."
        results = list(get_steuer_idnr_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual(VALID_IDNR, results[0].value)
        self.assertEqual("steuer_idnr", results[0].kind)
        self.assertEqual("de", results[0].locale)

    def test_skips_invalid_idnr(self):
        text = "IdNr 12345678901 ist falsch."
        self.assertEqual([], list(get_steuer_idnr_annotations(text)))


class TestUstIdnrAnnotations(TestCase):
    def test_extracts_ust(self):
        text = f"USt-IdNr.: {VALID_UST}"
        results = list(get_ust_idnr_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual(VALID_UST, results[0].value)
        self.assertEqual("ust_idnr", results[0].kind)

    def test_to_dictionary(self):
        match = next(get_ust_idnr_annotations(f"VAT-No {VALID_UST}"))
        d = match.to_dictionary()
        self.assertEqual("ust_idnr", d["record_type"])
        self.assertEqual(VALID_UST, d["value"])


class TestHrbAnnotations(TestCase):
    def test_extracts_hrb(self):
        text = "Eingetragen im Handelsregister, HRB 12345 Berlin."
        results = list(get_hrb_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual("hrb", results[0].kind)
        self.assertEqual("HRB 12345", results[0].value)

    def test_extracts_hra(self):
        text = "HRA 99876 Amtsgericht München."
        results = list(get_hrb_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual("hra", results[0].kind)
        self.assertEqual("HRA 99876", results[0].value)


class TestGetIdentifierAnnotations(TestCase):
    def test_extracts_all_kinds(self):
        text = f"Steuer-IdNr {VALID_IDNR}, USt-IdNr {VALID_UST}, HRB 12345."
        results = list(get_identifier_annotations(text))
        kinds = sorted(r.kind for r in results)
        self.assertEqual(["hrb", "steuer_idnr", "ust_idnr"], kinds)

    def test_empty_text_yields_nothing(self):
        self.assertEqual([], list(get_identifier_annotations("")))

    def test_dataclass_is_frozen(self):
        m = DeIdentifierMatch(kind="ust_idnr", value=VALID_UST, surface=VALID_UST, coords=(0, 11))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            m.value = "DE000000000"


class TestDeIdentifiersPackageReexports(TestCase):
    """Lock the public surface of ``lexnlp.extract.de`` against drift."""

    def test_reexports_are_identical_objects(self):
        from lexnlp.extract import de as de_pkg
        from lexnlp.extract.de import identifiers as identifiers_mod

        for name in (
            "DeIdentifierMatch",
            "get_hrb_annotations",
            "get_identifier_annotations",
            "get_steuer_idnr_annotations",
            "get_ust_idnr_annotations",
        ):
            self.assertIs(getattr(de_pkg, name), getattr(identifiers_mod, name))
            self.assertIn(name, de_pkg.__all__)
