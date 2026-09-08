__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase
from unittest.mock import patch

from lexnlp.extract.de import court_citations
from lexnlp.extract.de.court_citations import (
    CourtCitationsParser,
    PossibleToken,
    get_court_citation_annotation_list,
)


class TestCourtCitationsCoverage(TestCase):
    def test_possible_token_repr(self):
        tok = PossibleToken("registry", "BFH", (0, 3), 100)
        text = repr(tok)
        self.assertEqual("BFH [registry] at (0, 3), prob: 100", text)

    def test_get_dates_type_error_falls_back_to_years(self):
        parser = CourtCitationsParser()
        with patch.object(court_citations, "get_dates", side_effect=TypeError("boom")):
            tokens = parser.get_dates_from_text("something in 2015 happened")
        self.assertEqual(1, len(tokens))
        self.assertEqual("date", tokens[0].token_type)
        self.assertEqual("2015", tokens[0].value)
        self.assertEqual(50, tokens[0].prob)

    def test_get_dates_type_error_no_year(self):
        parser = CourtCitationsParser()
        with patch.object(court_citations, "get_dates", side_effect=TypeError("boom")):
            tokens = parser.get_dates_from_text("no numbers here at all")
        self.assertEqual([], tokens)

    def test_get_court_citation_annotation_list(self):
        text = (
            "Der IV. Senat des BFH hat im Urteil in BFHE 238, 518, BStBl II 2013, 505 ebenfalls "
            "einen Anspruch auf eine Billigkeitsmassnahme verneint."
        )
        items = get_court_citation_annotation_list(text)
        self.assertGreater(len(items), 0)
        self.assertTrue(all(item.locale == "de" for item in items))
        names = [item.name for item in items]
        self.assertIn("Deutschland, Rechtswesen: Bundesfinanzhof", names)
