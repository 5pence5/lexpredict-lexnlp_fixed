"""Coverage tests for lexnlp.extract.en.entities.company_detector."""

from lexnlp.config.en.company_types import COMPANY_DESCRIPTIONS, COMPANY_TYPES
from lexnlp.extract.en.entities.company_detector import CompanyDetector, get_noun_phrases

detector = CompanyDetector(COMPANY_TYPES, COMPANY_DESCRIPTIONS)


def test_get_company_annotations_uppercase_text_yields_nothing():
    assert list(detector.get_company_annotations("ACME LLC IS GREAT")) == []


def test_get_company_annotations_basic_match():
    annotations = list(detector.get_company_annotations("Acme LLC is great."))
    assert [(a.name, a.company_type) for a in annotations] == [("Acme", "LLC")]
    assert annotations[0].coords[0] >= 0


def test_get_persons_cc_joined_name():
    # "and" (CC) immediately follows the PERSON chunk "Tom", so the
    # CC/punctuation join branch merges everything into one person.
    assert list(detector.get_persons("Tom and Jerry Adams went home.")) == ["Tom and Jerry Adams"]


def test_get_companies_re_without_sentence_splitter():
    annotations = list(detector.get_companies_re("Acme LLC is great.", use_sentence_splitter=False))
    assert [(a.name, a.company_type) for a in annotations] == [("Acme", "LLC")]


def test_get_companies_re_description_only_name_filtered():
    # "Bank" matches the pattern via "Bank LLC", but the surviving
    # company_name is itself a company description, so it is skipped.
    assert list(detector.get_companies_re("Bank LLC", use_sentence_splitter=False)) == []
    assert list(detector.get_companies_re("Trust Company", use_sentence_splitter=False)) == []


def test_get_noun_phrases_merges_adjacent_nnp():
    assert list(get_noun_phrases("John Smith went home.")) == ["John Smith"]


def test_get_noun_phrases_joins_over_cc():
    assert list(get_noun_phrases("John and Smith went home.")) == ["John and Smith"]
