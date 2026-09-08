__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import types
from unittest import TestCase

from lexnlp.extract.de.copyrights import get_copyright_list, get_copyrights


class TestGetCopyrightsGenerator(TestCase):
    def test_get_copyrights_yields_dictionaries(self):
        text = "Copyright 2019, Siemens"
        result = get_copyrights(text)
        self.assertIsInstance(result, types.GeneratorType)
        items = list(result)
        self.assertEqual(1, len(items))
        item = items[0]
        self.assertIsInstance(item, dict)
        self.assertEqual("Siemens", item["tags"]["Extracted Entity Company"])
        self.assertEqual(2019, item["tags"]["Extracted Entity Start"])
        self.assertEqual("copyright", item["tags"]["Extracted Entity Type"])

    def test_get_copyrights_empty_text_yields_nothing(self):
        self.assertEqual([], list(get_copyrights("")))
        self.assertEqual([], list(get_copyrights("no rights mentioned here")))

    def test_get_copyright_list_matches_generator(self):
        text = "Copyright 2019, Siemens"
        self.assertEqual(list(get_copyrights(text)), get_copyright_list(text))
        self.assertEqual(1, len(get_copyright_list(text)))

    def test_get_copyrights_return_sources(self):
        text = "Copyright 2019, Siemens"
        items = list(get_copyrights(text, return_sources=True))
        self.assertEqual(1, len(items))
        self.assertEqual("Siemens", items[0]["tags"]["Extracted Entity Company"])
