from datetime import datetime
from unittest.mock import patch

from lexnlp.extract.all_locales import amounts, court_citations, dates
from lexnlp.extract.all_locales.languages import Locale


def test_german_amount_dispatch_uses_named_arguments_in_the_right_order():
    calls = []

    def parse_amounts(*, text, float_digits, return_sources):
        calls.append((text, float_digits, return_sources))
        yield 'amount'

    with patch.dict(amounts.ROUTINE_BY_LOCALE, {'de': parse_amounts}):
        result = list(
            amounts.get_amount_annotations(
                'de-DE',
                'zehn',
                extended_sources=False,
                float_digits=2,
            )
        )

    assert result == ['amount']
    assert calls == [('zehn', 2, False)]


def test_german_date_dispatch_uses_the_supported_signature():
    calls = []
    base_date = datetime(2026, 7, 29)

    def parse_dates(*, text, strict, locale, base_date, threshold):
        calls.append((text, strict, locale, base_date, threshold))
        yield 'date'

    with patch.dict(dates.ROUTINE_BY_LOCALE, {'de': parse_dates}):
        result = list(
            dates.get_date_annotations(
                'de-DE',
                '29. Juli 2026',
                strict=False,
                base_date=base_date,
                threshold=0.75,
            )
        )

    assert result == ['date']
    assert calls[0][:2] == ('29. Juli 2026', False)
    assert isinstance(calls[0][2], Locale)
    assert calls[0][2].get_locale() == 'de-DE'
    assert calls[0][3:] == (base_date, 0.75)


def test_german_date_dispatch_runs_with_default_options():
    text = '5. Oktober 2011'
    annotations = list(dates.get_date_annotations('de-DE', text))

    assert len(annotations) == 1
    assert annotations[0].coords == (0, len(text))
    assert annotations[0].text == text


def test_all_locale_entry_point_preserves_english_fallback():
    calls = []

    def parse_amounts(*, text, extended_sources, float_digits):
        calls.append((text, extended_sources, float_digits))
        yield 'amount'

    with patch.dict(
        amounts.ROUTINE_BY_LOCALE,
        {'en': parse_amounts},
        clear=True,
    ):
        result = list(
            amounts.get_amount_annotations(
                'fr-FR',
                'dix',
                extended_sources=False,
                float_digits=2,
            )
        )

    assert result == ['amount']
    assert calls == [('dix', False, 2)]


def test_court_citation_dispatch_defaults_language_from_locale():
    calls = []

    def parse_citations(text, language):
        calls.append((text, language))
        yield 'citation'

    with patch.dict(
        court_citations.ROUTINE_BY_LOCALE,
        {'de': parse_citations},
    ):
        result = list(
            court_citations.get_court_citation_annotations('de-DE', 'BStBl')
        )

    assert result == ['citation']
    assert calls == [('BStBl', 'de')]


def test_court_citation_dispatch_preserves_german_fallback():
    calls = []

    def parse_citations(text, language):
        calls.append((text, language))
        yield 'citation'

    with patch.dict(
        court_citations.ROUTINE_BY_LOCALE,
        {'de': parse_citations},
        clear=True,
    ):
        result = list(
            court_citations.get_court_citation_annotations('fr-FR', 'BStBl')
        )

    assert result == ['citation']
    assert calls == [('BStBl', 'de')]
