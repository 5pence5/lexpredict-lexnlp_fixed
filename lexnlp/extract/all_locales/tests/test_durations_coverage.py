__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import collections.abc
from decimal import Decimal

from lexnlp.extract.all_locales import durations
from lexnlp.extract.all_locales.durations import ROUTINE_BY_LOCALE, get_duration_annotations
from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN, LANG_PT
from lexnlp.extract.de.durations import get_duration_annotations as get_duration_annotations_de
from lexnlp.extract.en.durations import get_duration_annotations as get_duration_annotations_en
from lexnlp.extract.pt.durations import get_duration_annotations as get_duration_annotations_pt

EN_TEXT = "The term is 30 days after signing."
DE_TEXT = "Die Frist beträgt 30 Tage nach Unterzeichnung."
EN_ROUNDING_TEXT = "The term is 1.56789 years long here."


def test_module_metadata():
    assert durations.__version__ == "2.3.0"
    assert durations.__maintainer__ == "LexPredict, LLC"


def test_routine_by_locale_mapping():
    # Portuguese joined the dispatcher: lexnlp.extract.pt ships a native
    # routine, and routing pt through the English one reported Brazilian
    # reais as USD.
    assert set(ROUTINE_BY_LOCALE) == {LANG_EN.code, LANG_DE.code, LANG_PT.code}
    assert ROUTINE_BY_LOCALE["en"] is get_duration_annotations_en
    assert ROUTINE_BY_LOCALE["de"] is get_duration_annotations_de
    assert ROUTINE_BY_LOCALE["pt"] is get_duration_annotations_pt


def test_returns_generator():
    assert isinstance(get_duration_annotations("en", EN_TEXT), collections.abc.Generator)


def test_english_locale_routes_to_english():
    for locale in ("en", "en-US"):
        found = list(get_duration_annotations(locale, EN_TEXT))
        expected = list(get_duration_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert found[0].locale == "en"
        assert found[0].text == "30 days"


def test_german_locale_routes_to_german():
    for locale in ("de", "de-DE"):
        found = list(get_duration_annotations(locale, DE_TEXT))
        expected = list(get_duration_annotations_de(DE_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]
        assert found[0].locale == "de"
        assert found[0].text == "30 Tage"


def test_unknown_locale_falls_back_to_default_english():
    assert ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code] is get_duration_annotations_en
    for locale in ("fr-FR", "es-ES"):
        found = list(get_duration_annotations(locale, EN_TEXT))
        expected = list(get_duration_annotations_en(EN_TEXT))
        assert len(found) == 1
        assert [(a.text, a.coords) for a in found] == [(a.text, a.coords) for a in expected]


def test_float_digits_default_and_forwarding():
    defaulted = list(get_duration_annotations("en", EN_ROUNDING_TEXT))
    explicit = list(get_duration_annotations("en", EN_ROUNDING_TEXT, 4))
    assert [a.amount for a in defaulted] == [a.amount for a in explicit] == [Decimal("1.5679")]
    rounded = list(get_duration_annotations("en", EN_ROUNDING_TEXT, 2))
    assert [a.amount for a in rounded] == [Decimal("1.57")]
    assert rounded[0].amount != defaulted[0].amount


def test_float_digits_forwarded_to_german_routine():
    defaulted = list(get_duration_annotations("de", DE_TEXT))
    expected = list(get_duration_annotations_de(DE_TEXT, 4))
    assert [(a.text, a.amount) for a in defaulted] == [(a.text, a.amount) for a in expected]


def test_empty_and_duration_free_text_yield_nothing():
    assert list(get_duration_annotations("en", "")) == []
    assert list(get_duration_annotations("de", "")) == []
    assert list(get_duration_annotations("en", "no durations here")) == []
