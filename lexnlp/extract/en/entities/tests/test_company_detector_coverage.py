"""Coverage tests for lexnlp.extract.en.entities.company_detector."""

from lexnlp.config.en.company_types import COMPANY_DESCRIPTIONS, COMPANY_TYPES
from lexnlp.extract.common.entities.entity_banlist import BanListUsage, EntityBanListItem
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


def test_get_company_annotations_use_gnp_path():
    annotations = list(detector.get_company_annotations("Acme LLC is great.", use_gnp=True))
    assert [(a.name, a.company_type) for a in annotations] == [("Acme", "LLC")]
    assert annotations[0].coords[0] >= 0


def test_get_company_annotations_skips_degenerate_match():
    degenerate = list(detector.get_companies_re("LLC LLC", use_sentence_splitter=False))
    assert [(a.name, a.company_type) for a in degenerate] == [("LLC", "LLC")]
    assert list(detector.get_company_annotations("LLC LLC is great.")) == []


def test_get_company_annotations_banlist_skip():
    banned = BanListUsage(banlist=[EntityBanListItem("Acme")], use_default_banlist=False)
    assert list(detector.get_company_annotations("Acme LLC is great.", banlist_usage=banned)) == []
    control = list(detector.get_company_annotations("Acme LLC is great."))
    assert [(a.name, a.company_type) for a in control] == [("Acme", "LLC")]


def test_get_persons_filters_short_name():
    assert list(detector.get_persons("Al went home.")) == []
    assert list(detector.get_persons("Albert went home.")) == ["Albert"]


def test_get_companies_re_skips_backtrack_catastrophe():
    text = "A" * 85 + " LLC"
    assert detector.check_backtrack_catastrophy(text) is not None
    assert list(detector.get_companies_re(text, use_sentence_splitter=False)) == []


def test_get_companies_re_skips_empty_company_name():
    assert detector.re_company.search("2020 LLC") is not None
    assert list(detector.get_companies_re("2020 LLC", use_sentence_splitter=False)) == []


def test_get_companies_re_skips_lowercase_article():
    assert list(detector.get_companies_re("a Acme LLC is here.", use_sentence_splitter=False, use_article=True)) == []
    assert [
        (a.name, a.company_type)
        for a in detector.get_companies_re("A Acme LLC is here.", use_sentence_splitter=False, use_article=True)
    ] == [("Acme", "LLC")]
    assert [
        (a.name, a.company_type) for a in detector.get_companies_re("a Acme LLC is here.", use_sentence_splitter=False)
    ] == [("Acme", "LLC")]


def test_contains_companies_skips_degenerate_match():
    assert detector.contains_companies("LLC LLC", []) is False
    companies = list(detector.get_company_annotations("John Smith, LLC is great."))
    assert [(a.name, a.company_type) for a in companies] == [("John Smith", "LLC")]
    assert detector.contains_companies("John Smith", companies) is True


def test_get_persons_filters_company_overlap():
    assert list(detector.get_persons("John Smith met Jane Doe near John Smith, LLC headquarters.")) == ["Jane Doe"]


def test_get_noun_phrases_strips_trailing_ampersand():
    assert list(get_noun_phrases("John & went home.")) == ["John"]


def test_get_noun_phrases_return_source():
    assert list(get_noun_phrases("John Smith went home.", return_source=True)) == [
        ("John Smith", "John Smith went home.")
    ]
