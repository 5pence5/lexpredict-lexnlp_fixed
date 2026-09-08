__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.common.annotations.text_annotation import TextAnnotation


class ExtraValuesAnnotation(TextAnnotation):
    record_type = "extra"

    def __init__(self, name, locale, coords, text="", extras=None):
        super().__init__(name, locale, coords, text)
        self._extras = extras or {}

    def get_dictionary_values(self):
        return self._extras


class TestTextAnnotationCoverage(TestCase):
    def test_base_cite_value_parts(self):
        ant = TextAnnotation(name="Siemens", locale="en", coords=(0, 7), text="Siemens")
        self.assertEqual(["Siemens"], ant.get_cite_value_parts())
        self.assertEqual("/en/Siemens", ant.get_cite())

    def test_get_extracted_text(self):
        full = "hello world"
        ant = TextAnnotation(name="x", locale="en", coords=(0, 5), text="")
        self.assertEqual("hello", ant.get_extracted_text(full))
        ant2 = TextAnnotation(name="x", locale="en", coords=(6, 11), text="")
        self.assertEqual("world", ant2.get_extracted_text(full))

    def test_base_dictionary_values_empty(self):
        ant = TextAnnotation(name="x", locale="en", coords=(0, 1), text="x")
        self.assertEqual({}, ant.get_dictionary_values())

    def test_to_dictionary_new_key(self):
        ant = ExtraValuesAnnotation(
            name="x", locale="en", coords=(0, 1), text="x", extras={"custom_field": "custom_value"}
        )
        dic = ant.to_dictionary()
        self.assertEqual("custom_value", dic["custom_field"])
        self.assertEqual(0, dic["attrs"]["start"])
        self.assertEqual(1, dic["attrs"]["end"])

    def test_to_dictionary_merges_existing_key(self):
        ant = ExtraValuesAnnotation(
            name="x", locale="en", coords=(0, 1), text="x", extras={"tags": {"Extra Tag": "yes"}}
        )
        dic = ant.to_dictionary()
        self.assertEqual("yes", dic["tags"]["Extra Tag"])
        self.assertIn("Extracted Entity Type", dic["tags"])

    def test_get_int_value_none(self):
        self.assertIsNone(TextAnnotation.get_int_value(None))
        self.assertEqual(7, TextAnnotation.get_int_value(None, 7))

    def test_get_int_value_types(self):
        self.assertEqual(5, TextAnnotation.get_int_value(5))
        self.assertEqual(5, TextAnnotation.get_int_value("5"))
        self.assertEqual(9, TextAnnotation.get_int_value("bad", 9))
        self.assertEqual(3, TextAnnotation.get_int_value([1], 3))

    def test_safe_cast(self):
        self.assertEqual(42, TextAnnotation.safe_cast("42", int))
        self.assertIsNone(TextAnnotation.safe_cast("bad", int))
        self.assertEqual("dflt", TextAnnotation.safe_cast(None, int, "dflt"))
