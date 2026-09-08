__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.common.annotations.court_annotation import CourtAnnotation
from lexnlp.extract.en.courts import get_court_annotation_list, get_court_annotations


class TestGetCourtAnnotationList(TestCase):
    def test_returns_court_annotations(self):
        text = "The Supreme Court of the United States decided the case."
        result = get_court_annotation_list(text)
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        annotation = result[0]
        self.assertIsInstance(annotation, CourtAnnotation)
        payload = annotation.to_dictionary()
        self.assertEqual("court", payload["tags"]["Extracted Entity Type"])
        self.assertEqual("Supreme Court", payload["tags"]["Extracted Entity Court Name"])
        self.assertIn("Supreme Court", payload["tags"]["Extracted Entity Text"])

    def test_matches_generator_output(self):
        text = "The Supreme Court of the United States decided the case."
        from_generator = list(get_court_annotations(text))
        from_list = get_court_annotation_list(text)
        self.assertEqual(len(from_generator), len(from_list))
        self.assertEqual(
            from_generator[0].to_dictionary()["tags"],
            from_list[0].to_dictionary()["tags"],
        )

    def test_no_courts_returns_empty_list(self):
        self.assertEqual([], get_court_annotation_list("No court mentioned here, just a sunny day."))
        self.assertEqual([], get_court_annotation_list(""))
