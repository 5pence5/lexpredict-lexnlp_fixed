__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import collections.abc

from lexnlp.extract.all_locales import citations
from lexnlp.extract.all_locales.citations import ROUTINE_BY_LOCALE, get_citation_annotations
from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN
from lexnlp.extract.de.citations import get_citation_annotations as get_citation_annotations_de
from lexnlp.extract.en.citations import get_citation_annotations as get_citation_annotations_en

EN_TEXT = "See 410 U.S. 113 (1973) for details."
DE_TEXT = "Artikel 2 Nr. 1 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"


def test_module_metadata():
    assert citations.__version__ == "2.3.0"
    assert citations.__maintainer__ == "LexPredict, LLC"


def test_routine_by_locale_mapping():
    assert set(ROUTINE_BY_LOCALE) == {LANG_EN.code, LANG_DE.code}
    assert ROUTINE_BY_LOCALE["en"] is get_citation_annotations_en
    assert ROUTINE_BY_LOCALE["de"] is get_citation_annotations_de


def test_returns_generator():
    assert isinstance(get_citation_annotations("en", EN_TEXT), collections.abc.Generator)


def test_english_locale_routes_to_english():
    for locale in ("en", "en-US"):
        found = list(get_citation_annotations(locale, EN_TEXT))
        expected = list(get_citation_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert found[0].locale == "en"
        assert found[0].volume == 410
        assert found[0].page == 113
        assert found[0].reporter == "U.S."
        assert found[0].coords == (3, 24)


def test_german_locale_routes_to_german():
    for locale in ("de", "de-DE"):
        found = list(get_citation_annotations(locale, DE_TEXT))
        expected = list(get_citation_annotations_de(DE_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert found[0].locale == "de"
        assert found[0].article == 2
        assert found[0].page == 2477


def test_unknown_locale_falls_back_to_default_english():
    assert ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code] is get_citation_annotations_en
    for locale in ("fr-FR", "es-ES", "pt-PT"):
        found = list(get_citation_annotations(locale, EN_TEXT))
        expected = list(get_citation_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]


def test_empty_and_citation_free_text_yield_nothing():
    assert list(get_citation_annotations("en", "")) == []
    assert list(get_citation_annotations("de", "")) == []
    assert list(get_citation_annotations("en", "no citations here at all")) == []
    assert list(get_citation_annotations("de", "kein Zitat hier")) == []
