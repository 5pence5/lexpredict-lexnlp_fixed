__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.common.annotations.url_annotation import UrlAnnotation


class TestUrlAnnotationCoverage(TestCase):
    def test_get_dictionary_values_returns_url_and_text(self):
        ant = UrlAnnotation(coords=(0, 10), locale="en", text="google", url="www.google.com")
        self.assertEqual(
            ant.get_dictionary_values(),
            {"tags": {"Extracted Entity URL": "www.google.com", "Extracted Entity Text": "google"}},
        )

    def test_get_dictionary_values_preserves_none(self):
        ant = UrlAnnotation(coords=(5, 15), text="display")
        self.assertIsNone(ant.url)
        self.assertEqual(
            ant.get_dictionary_values(),
            {"tags": {"Extracted Entity URL": None, "Extracted Entity Text": "display"}},
        )

    def test_to_dictionary_merges_url_values(self):
        ant = UrlAnnotation(coords=(0, 10), locale="en", text="google", url="www.google.com")
        result = ant.to_dictionary()
        self.assertEqual(result["attrs"]["start"], 0)
        self.assertEqual(result["attrs"]["end"], 10)
        self.assertEqual(result["tags"]["Extracted Entity Type"], "url")
        self.assertEqual(result["tags"]["Extracted Entity URL"], "www.google.com")
        self.assertEqual(result["tags"]["Extracted Entity Text"], "google")

    def test_get_cite_value_parts_returns_url(self):
        ant = UrlAnnotation(coords=(0, 10), url="www.google.com")
        self.assertEqual(ant.get_cite_value_parts(), ["www.google.com"])
