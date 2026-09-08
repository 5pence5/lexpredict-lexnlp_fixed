__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import collections.abc
from decimal import Decimal

from lexnlp.extract.all_locales import money
from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN
from lexnlp.extract.all_locales.money import ROUTINE_BY_LOCALE, get_money_annotations
from lexnlp.extract.de.money import get_money_annotations as get_money_annotations_de
from lexnlp.extract.en.money import get_money_annotations as get_money_annotations_en

EN_TEXT = "The fee is 1000.56789 dollars now."
DE_TEXT = "Der Preis beträgt 1.000,50 Euro fällig jetzt."


def test_module_metadata():
    assert money.__version__ == "2.3.0"
    assert money.__maintainer__ == "LexPredict, LLC"


def test_routine_by_locale_mapping():
    assert set(ROUTINE_BY_LOCALE) == {LANG_EN.code, LANG_DE.code}
    assert ROUTINE_BY_LOCALE["en"] is get_money_annotations_en
    assert ROUTINE_BY_LOCALE["de"] is get_money_annotations_de


def test_returns_generator():
    assert isinstance(get_money_annotations("en", EN_TEXT), collections.abc.Generator)


def test_english_locale_routes_to_english():
    for locale in ("en", "en-US"):
        found = list(get_money_annotations(locale, EN_TEXT))
        expected = list(get_money_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert found[0].locale == "en"
        assert found[0].text == "1000.56789 dollars"
        assert found[0].currency == "USD"


def test_german_locale_routes_to_german():
    for locale in ("de", "de-DE"):
        found = list(get_money_annotations(locale, DE_TEXT))
        expected = list(get_money_annotations_de(DE_TEXT))
        assert len(found) >= 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert all(a.locale == "de" for a in found)


def test_unknown_locale_falls_back_to_default_english():
    assert ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code] is get_money_annotations_en
    for locale in ("fr-FR", "es-ES"):
        found = list(get_money_annotations(locale, EN_TEXT))
        expected = list(get_money_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]


def test_float_digits_default_and_forwarding():
    defaulted = list(get_money_annotations("en", EN_TEXT))
    explicit = list(get_money_annotations("en", EN_TEXT, 4))
    assert [a.amount for a in defaulted] == [a.amount for a in explicit] == [Decimal("1000.5679")]
    rounded = list(get_money_annotations("en", EN_TEXT, 2))
    assert [a.amount for a in rounded] == [Decimal("1000.57")]
    assert rounded[0].amount != defaulted[0].amount


def test_float_digits_forwarded_to_german_routine():
    defaulted = list(get_money_annotations("de", DE_TEXT))
    expected = list(get_money_annotations_de(DE_TEXT, 4))
    assert [(a.text, a.amount) for a in defaulted] == [(a.text, a.amount) for a in expected]


def test_empty_and_money_free_text_yield_nothing():
    assert list(get_money_annotations("en", "")) == []
    assert list(get_money_annotations("de", "")) == []
    assert list(get_money_annotations("en", "no money here")) == []
