__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.en.trademarks import (
    get_trademark_annotation_list,
    get_trademark_annotations,
    get_trademark_list,
    get_trademarks,
)


class TestTrademarkListWrappers(TestCase):
    def test_get_trademark_list_matches_generator(self):
        text = (
            "1.11.SCADA System means a supervisory control and data "
            + "acquisition system such as the S/3 Software or Licensee's OASyS(R) product."
        )
        result = get_trademark_list(text)
        self.assertEqual(["OASyS (R)"], result)
        self.assertEqual(list(get_trademarks(text)), result)

    def test_get_trademark_list_empty(self):
        self.assertEqual([], get_trademark_list("no marks here, just plain words."))
        self.assertEqual([], get_trademark_list(""))

    def test_get_trademark_annotation_list_matches_generator(self):
        text = (
            "1.11.SCADA System means a supervisory control and data "
            + "acquisition system such as the S/3 Software or Licensee's OASyS(R) product."
        )
        result = get_trademark_annotation_list(text)
        expected = list(get_trademark_annotations(text))
        self.assertEqual(1, len(result))
        self.assertEqual("OASyS (R)", result[0].trademark)
        self.assertEqual("en", result[0].locale)
        self.assertEqual(expected[0].coords, result[0].coords)
        self.assertEqual(expected[0].trademark, result[0].trademark)

    def test_get_trademark_annotation_list_empty(self):
        self.assertEqual([], get_trademark_annotation_list("no marks here."))
        self.assertEqual([], get_trademark_annotation_list(""))


class TestTrademarkEndOfTextClipping(TestCase):
    def test_trailing_trademark_is_clipped_to_text_end(self):
        # NPExtractor normalizes "OASyS(R)" to "OASyS (R)" (with a space),
        # so the in-phrase match end overshoots the source text length and
        # the coords are clipped to len(text) - 1.
        text = "Licensee's OASyS(R)"
        ants = list(get_trademark_annotations(text))
        self.assertEqual(1, len(ants))
        self.assertEqual("OASyS (R)", ants[0].trademark)
        self.assertEqual((11, len(text) - 1), ants[0].coords)

    def test_single_token_trailing_trademark_is_clipped(self):
        text = "BetLUCK(TM)"
        ants = list(get_trademark_annotations(text))
        self.assertEqual(1, len(ants))
        self.assertEqual("BetLUCK (TM)", ants[0].trademark)
        self.assertEqual((0, len(text) - 1), ants[0].coords)

    def test_unicode_trailing_trademark_is_clipped(self):
        text = "Buy Acme™"
        ants = list(get_trademark_annotations(text))
        self.assertEqual(1, len(ants))
        self.assertEqual("Buy Acme™", ants[0].trademark)
        self.assertEqual((0, len(text) - 1), ants[0].coords)

    def test_mid_text_trademark_is_not_clipped(self):
        text = "Licensee's OASyS(R) product."
        ants = list(get_trademark_annotations(text))
        self.assertEqual(1, len(ants))
        self.assertEqual((11, 20), ants[0].coords)
        self.assertLess(ants[0].coords[1], len(text))

    def test_clipped_lists_agree(self):
        text = "Licensee's OASyS(R)"
        self.assertEqual(["OASyS (R)"], get_trademark_list(text))
        ants = get_trademark_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual((11, len(text) - 1), ants[0].coords)
