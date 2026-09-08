"""A single word quoted as a defined term is not a person.

Contracts introduce parties as ``and GERALD GREENWALD (the "Employee")``. The
quoted word defines a role; it does not name a second human. nltk 3.10's NER
model tags several of these, so the person extractor drops any single-word
candidate that only ever appears inside quotes.
"""

from unittest import TestCase

from lexnlp.extract.en.entities.company_detector import _is_quoted_defined_term
from lexnlp.extract.en.entities.nltk_maxent import get_persons


class TestQuotedDefinedTermsAreNotPersons(TestCase):
    def test_role_words_are_not_returned_as_people(self):
        text = "a Delaware corporation (the 'Employer'), and GERALD GREENWALD (the 'Employee')."

        found = list(get_persons(text))

        self.assertIn("GERALD GREENWALD", found)
        self.assertNotIn("Employer", found)
        self.assertNotIn("Employee", found)

    def test_a_quoted_multi_word_party_name_is_kept(self):
        """A real party name is usually quoted too, so only single words are dropped."""
        self.assertFalse(_is_quoted_defined_term("Acme Holdings", 'between "Acme Holdings" and others'))

    def test_single_word_only_ever_quoted_is_a_defined_term(self):
        self.assertTrue(_is_quoted_defined_term("Employee", 'GERALD GREENWALD (the "Employee").'))

    def test_single_word_also_used_bare_is_kept(self):
        """If the word also appears unquoted it is being used as a name, not defined."""
        self.assertFalse(_is_quoted_defined_term("Greenwald", 'Greenwald signed. See "Greenwald" above.'))

    def test_curly_and_straight_quotes_are_both_recognised(self):
        for opening, closing in (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’")):
            with self.subTest(quotes=f"{opening}{closing}"):
                self.assertTrue(_is_quoted_defined_term("Employer", f"the {opening}Employer{closing} shall pay"))
