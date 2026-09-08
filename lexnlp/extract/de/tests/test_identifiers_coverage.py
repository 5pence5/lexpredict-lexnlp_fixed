"""Coverage tests for the ``sum_mod == 0`` branch in ``_ust_idnr_is_valid``."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.de.identifiers import _ust_idnr_is_valid, get_ust_idnr_annotations

# Leading-zero body: the first loop iteration computes
# ``(0 + 10) % 10 == 0``, forcing the ``sum_mod = 10`` branch.
ZERO_LEAD_BODY = "037297049"
ZERO_LEAD_UST = "DE037297049"


class TestUstIdnrZeroSumBranch(TestCase):
    def test_valid_leading_zero_id_passes_checksum(self):
        self.assertTrue(_ust_idnr_is_valid(ZERO_LEAD_BODY))

    def test_leading_zero_id_with_bad_check_digit_fails(self):
        bad = ZERO_LEAD_BODY[:-1] + str((int(ZERO_LEAD_BODY[-1]) + 1) % 10)
        self.assertNotEqual(ZERO_LEAD_BODY, bad)
        self.assertFalse(_ust_idnr_is_valid(bad))

    def test_extracts_leading_zero_ust_id(self):
        text = f"USt-IdNr.: {ZERO_LEAD_UST}"
        results = list(get_ust_idnr_annotations(text))
        self.assertEqual(1, len(results))
        match = results[0]
        self.assertEqual("ust_idnr", match.kind)
        self.assertEqual(ZERO_LEAD_UST, match.value)
        self.assertEqual(ZERO_LEAD_UST, match.surface)
        self.assertEqual((11, 22), match.coords)
        self.assertEqual("de", match.locale)

    def test_extracts_spaced_leading_zero_ust_id(self):
        text = "USt-IdNr DE 037 297 049"
        results = list(get_ust_idnr_annotations(text))
        self.assertEqual(1, len(results))
        self.assertEqual(ZERO_LEAD_UST, results[0].value)
