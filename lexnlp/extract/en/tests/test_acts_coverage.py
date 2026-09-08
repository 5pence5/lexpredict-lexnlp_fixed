__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.common.annotations.act_annotation import ActAnnotation
from lexnlp.extract.en.acts import get_acts_annotations, get_acts_annotations_list


class TestGetActsAnnotationsList(TestCase):
    def test_returns_act_annotations(self):
        text = "test section 12 of the VERY Important Act of 1954."
        result = get_acts_annotations_list(text)
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        act = result[0]
        self.assertIsInstance(act, ActAnnotation)
        self.assertEqual("VERY Important Act", act.act_name)
        self.assertEqual("12", act.section)
        self.assertEqual(1954, act.year)
        self.assertFalse(act.ambiguous)
        self.assertEqual("section 12 of the VERY Important Act of 1954", act.text)
        self.assertEqual((5, 49), act.coords)
        self.assertEqual("en", act.locale)

    def test_matches_generator_output(self):
        text = "test section 12 of the VERY Important Act of 1954."
        from_generator = list(get_acts_annotations(text))
        from_list = get_acts_annotations_list(text)
        self.assertEqual(len(from_generator), len(from_list))
        self.assertEqual(from_generator[0].act_name, from_list[0].act_name)
        self.assertEqual(from_generator[0].coords, from_list[0].coords)

    def test_no_acts_returns_empty_list(self):
        self.assertEqual([], get_acts_annotations_list("no acts here, just a contract."))
        self.assertEqual([], get_acts_annotations_list(""))
