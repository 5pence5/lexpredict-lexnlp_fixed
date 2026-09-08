from datetime import datetime
from unittest.mock import patch

from lexnlp.extract.all_locales import amounts, court_citations, dates
from lexnlp.extract.all_locales.languages import Locale


def test_german_amount_dispatch_uses_named_arguments_in_the_right_order():
    calls = []

    def parse_amounts(*, text, float_digits, return_sources):
        calls.append((text, float_digits, return_sources))
        yield "amount"

    with patch.dict(amounts.ROUTINE_BY_LOCALE, {"de": parse_amounts}):
        result = list(
            amounts.get_amount_annotations(
                "de-DE",
                "zehn",
                extended_sources=False,
                float_digits=2,
            )
        )

    assert result == ["amount"]
    assert calls == [("zehn", 2, False)]


def test_german_date_dispatch_uses_the_supported_signature():
    """German, Spanish and Portuguese parsers take (text, locale, strict) only.

    They do not accept ``base_date`` or ``threshold``, so the dispatcher must
    not forward them. Passing all five arguments -- positionally or by name --
    raises TypeError against the real parsers.
    """
    calls = []
    base_date = datetime(2026, 7, 29)

    def parse_dates(*, text, locale, strict):
        calls.append((text, locale, strict))
        yield "date"

    with patch.dict(dates.ROUTINE_BY_LOCALE, {"de": parse_dates}):
        result = list(
            dates.get_date_annotations(
                "de-DE",
                "29. Juli 2026",
                strict=False,
                base_date=base_date,
                threshold=0.75,
            )
        )

    assert result == ["date"]
    assert calls[0][0] == "29. Juli 2026"
    assert isinstance(calls[0][1], Locale)
    assert calls[0][1].get_locale() == "de-DE"
    assert calls[0][2] is False


def test_german_date_dispatch_omits_strict_when_unset():
    """The parsers declare ``strict: bool = True``.

    Forwarding the dispatcher's ``None`` default reaches dateparser as
    STRICT_PARSING=None, which it rejects, so ``strict`` must be left out
    entirely when the caller does not set it.
    """
    seen = []

    def parse_dates(**kwargs):
        seen.append(kwargs)
        yield "date"

    with patch.dict(dates.ROUTINE_BY_LOCALE, {"de": parse_dates}):
        assert list(dates.get_date_annotations("de-DE", "29. Juli 2026")) == ["date"]

    assert "strict" not in seen[0]


def test_every_non_english_date_locale_parses_through_the_dispatcher():
    """Regression: the dispatcher used to raise TypeError for de, es and pt."""
    samples = {
        "de": "Das Datum ist der 1. Januar 2020.",
        "es": "La fecha es 1 de enero de 2020.",
        "pt": "A data e 1 de janeiro de 2020.",
    }

    for locale, text in samples.items():
        found = [a.date for a in dates.get_date_annotations(locale, text)]
        assert found, f"{locale} produced no annotation"
        assert found[0].year == 2020


def test_german_date_dispatch_runs_with_default_options():
    text = "5. Oktober 2011"
    annotations = list(dates.get_date_annotations("de-DE", text))

    assert len(annotations) == 1
    assert annotations[0].coords == (0, len(text))
    assert annotations[0].text == text


def test_all_locale_entry_point_preserves_english_fallback():
    calls = []

    def parse_amounts(*, text, extended_sources, float_digits):
        calls.append((text, extended_sources, float_digits))
        yield "amount"

    with patch.dict(
        amounts.ROUTINE_BY_LOCALE,
        {"en": parse_amounts},
        clear=True,
    ):
        result = list(
            amounts.get_amount_annotations(
                "fr-FR",
                "dix",
                extended_sources=False,
                float_digits=2,
            )
        )

    assert result == ["amount"]
    assert calls == [("dix", False, 2)]


def test_court_citation_dispatch_defaults_language_from_locale():
    calls = []

    def parse_citations(text, language):
        calls.append((text, language))
        yield "citation"

    with patch.dict(
        court_citations.ROUTINE_BY_LOCALE,
        {"de": parse_citations},
    ):
        result = list(court_citations.get_court_citation_annotations("de-DE", "BStBl"))

    assert result == ["citation"]
    assert calls == [("BStBl", "de")]


def test_court_citation_dispatch_preserves_german_fallback():
    calls = []

    def parse_citations(text, language):
        calls.append((text, language))
        yield "citation"

    with patch.dict(
        court_citations.ROUTINE_BY_LOCALE,
        {"de": parse_citations},
        clear=True,
    ):
        result = list(court_citations.get_court_citation_annotations("fr-FR", "BStBl"))

    assert result == ["citation"]
    assert calls == [("BStBl", "de")]


def test_german_amount_dispatch_does_not_swap_float_digits_and_sources():
    """English takes (text, extended_sources, float_digits); German takes
    (text, float_digits, return_sources).

    Dispatching positionally passed ``extended_sources=True`` in as
    ``float_digits``, so German amounts were rounded to a single decimal place
    while English kept four.
    """
    from decimal import Decimal

    from lexnlp.extract.all_locales import amounts

    german = [a.value for a in amounts.get_amount_annotations("de", "Der Betrag betraegt 1.000,5678 Euro.")]
    english = [a.value for a in amounts.get_amount_annotations("en", "The amount is 1,000.5678 dollars.")]

    assert german == [Decimal("1000.5678")]
    assert english == [Decimal("1000.5678")]


def test_german_amount_dispatch_passes_arguments_by_name():
    seen = []

    from lexnlp.extract.all_locales import amounts

    def parse_amounts(*, text, float_digits, return_sources):
        seen.append((text, float_digits, return_sources))
        yield "amount"

    with patch.dict(amounts.ROUTINE_BY_LOCALE, {"de": parse_amounts}):
        result = list(amounts.get_amount_annotations("de", "1.000,50 Euro", True, 4))

    assert result == ["amount"]
    assert seen == [("1.000,50 Euro", 4, True)]
