__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.de.citations import (
    get_citation_annotation_list,
    get_citation_annotations,
    get_citation_list,
)


class TestDeCitationsCoverage(TestCase):
    def test_single_number_sets_volume(self):
        text = "Artikel 2 Nr. 1 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"
        ants = list(get_citation_annotations(text))
        self.assertEqual(len(ants), 1)
        ant = ants[0]
        self.assertEqual(ant.article, 2)
        self.assertEqual(ant.volume, 1)
        self.assertIsNone(ant.volume_str)
        self.assertEqual(ant.page, 2477)
        self.assertEqual(ant.part, "I")

    def test_single_number_visible_in_citation_dict(self):
        text = "Artikel 2 Nr. 1 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"
        rows = get_citation_list(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["number"], "1")
        self.assertEqual(rows[0]["article"], "2")
        self.assertEqual(rows[0]["page"], "2477")

    def test_nummer_word_form_sets_volume(self):
        text = "Artikel 2 Nummer 5 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"
        ants = list(get_citation_annotations(text))
        self.assertEqual(len(ants), 1)
        self.assertEqual(ants[0].volume, 5)
        self.assertIsNone(ants[0].volume_str)

    def test_get_citation_annotation_list_matches_generator(self):
        text = "Artikel 2 Nr. 1 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"
        from_list = get_citation_annotation_list(text)
        from_gen = list(get_citation_annotations(text))
        self.assertEqual(len(from_list), 1)
        self.assertEqual(len(from_gen), 1)
        self.assertEqual(from_list[0].coords, from_gen[0].coords)
        self.assertEqual(from_list[0].text, from_gen[0].text)
        self.assertEqual(from_list[0].article, 2)
        self.assertEqual(from_list[0].volume, 1)

    def test_get_citation_annotation_list_empty(self):
        self.assertEqual(get_citation_annotation_list("kein Zitat hier"), [])
        self.assertEqual(get_citation_annotation_list(""), [])
