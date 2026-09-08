__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator
from unittest import TestCase

from lexnlp.extract.pt.definitions import get_definition_list, get_definitions


class TestPtGetDefinitionsWrappers(TestCase):
    TEXT = (
        'Neste acordo, o termo "Software" refere-se a: '
        "(i) o programa de computador e todos os seus componentes;"
    )

    def test_get_definitions_yields_dicts(self):
        result = get_definitions(self.TEXT)
        self.assertIsInstance(result, Generator)
        items = list(result)
        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], dict)
        self.assertEqual('"Software"', items[0]["tags"]["Extracted Entity Definition Name"])
        self.assertEqual("definition", items[0]["tags"]["Extracted Entity Type"])

    def test_get_definition_list_matches_generator(self):
        result = get_definition_list(self.TEXT)
        self.assertIsInstance(result, list)
        self.assertEqual(1, len(result))
        self.assertEqual(list(get_definitions(self.TEXT)), result)
        self.assertEqual('"Software"', result[0]["tags"]["Extracted Entity Definition Name"])

    def test_get_definitions_empty(self):
        self.assertEqual([], list(get_definitions("Olá.")))
        self.assertEqual([], list(get_definitions("")))

    def test_get_definition_list_empty(self):
        self.assertEqual([], get_definition_list("Olá."))
        self.assertEqual([], get_definition_list(""))
        self.assertEqual([], get_definition_list("Sem rótulo aqui."))
