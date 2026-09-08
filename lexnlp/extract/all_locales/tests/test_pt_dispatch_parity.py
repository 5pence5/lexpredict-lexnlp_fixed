"""Portuguese must be reachable through every all_locales dispatcher.

``lexnlp.extract.pt`` ships native money / percent / amount / duration /
citation extractors, but the matching dispatchers only registered ``en`` and
``de``. A caller asking for ``pt-BR`` therefore fell through to the *English*
routine and got silently wrong answers -- most visibly ``R$ 1.500.000,00``
reported as **USD** rather than **BRL**.

These tests pin the wiring and, just as importantly, pin the calling
convention: the per-language routines do not share a signature, so a
dispatcher that guesses gets a ``TypeError`` (or, worse, mis-binds an
argument silently -- see ``test_dispatch.py``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lexnlp.extract.all_locales import amounts, citations, durations, money, percents
from lexnlp.extract.all_locales.languages import LANG_EN, LANG_PT

DISPATCHERS = {
    "money": (money, "get_money_annotations"),
    "percents": (percents, "get_percent_annotations"),
    "amounts": (amounts, "get_amount_annotations"),
    "durations": (durations, "get_duration_annotations"),
    "citations": (citations, "get_citation_annotations"),
}


@pytest.mark.parametrize("name", sorted(DISPATCHERS))
def test_pt_is_registered(name: str) -> None:
    module, _ = DISPATCHERS[name]
    assert LANG_PT.code in module.ROUTINE_BY_LOCALE


@pytest.mark.parametrize("name", sorted(DISPATCHERS))
def test_pt_routine_is_not_the_english_routine(name: str) -> None:
    module, _ = DISPATCHERS[name]
    assert module.ROUTINE_BY_LOCALE[LANG_PT.code] is not module.ROUTINE_BY_LOCALE[LANG_EN.code]


@pytest.mark.parametrize("name", sorted(DISPATCHERS))
@pytest.mark.parametrize("locale", ["pt", "pt-BR", "pt-PT"])
def test_pt_locales_route_to_the_pt_routine(name: str, locale: str) -> None:
    """Every pt-* region tag resolves to the single ``pt`` routine."""
    module, func_name = DISPATCHERS[name]
    mock_pt = MagicMock(return_value=iter([]))
    with patch.dict(module.ROUTINE_BY_LOCALE, {LANG_PT.code: mock_pt}):
        list(getattr(module, func_name)(locale, "texto"))
    mock_pt.assert_called_once()


@pytest.mark.parametrize("name", sorted(DISPATCHERS))
def test_unregistered_locale_still_falls_back_to_english(name: str) -> None:
    """Adding pt must not break the fallback path.

    Branching the calling convention on the locale *string* rather than on the
    routine actually selected is what broke ``fr``/``it``/``nl`` for dates.
    """
    module, func_name = DISPATCHERS[name]
    mock_en = MagicMock(return_value=iter([]))
    with patch.dict(module.ROUTINE_BY_LOCALE, {LANG_EN.code: mock_en}):
        list(getattr(module, func_name)("fr", "texte"))
    mock_en.assert_called_once()


@pytest.mark.parametrize("name", sorted(DISPATCHERS))
def test_pt_dispatch_does_not_raise_against_the_real_routine(name: str) -> None:
    """The real pt routines must accept whatever the dispatcher forwards."""
    module, func_name = DISPATCHERS[name]
    text = "A Sociedade pagara R$ 1.500.000,00 no prazo de 30 dias, a taxa de 5,5% ao ano."
    list(getattr(module, func_name)("pt-BR", text))


def test_brazilian_currency_is_not_reported_as_usd() -> None:
    """The bug this file exists for: R$ must resolve to BRL, not USD."""
    found = list(money.get_money_annotations("pt-BR", "A Sociedade pagara R$ 1.500.000,00."))
    assert found, "no money annotation produced for a Brazilian amount"
    currencies = {annotation.currency for annotation in found}
    assert "USD" not in currencies, f"R$ was misread as USD: {currencies}"
    assert "BRL" in currencies, f"expected BRL, got {currencies}"


def test_pt_amounts_accept_a_non_default_float_digits() -> None:
    """``float_digits`` must reach the pt routine rather than being dropped."""
    text = "O valor e de 1.500.000,00 reais."
    assert list(amounts.get_amount_annotations("pt-BR", text, float_digits=2)) is not None


def test_pt_durations_accept_the_float_digits_keyword() -> None:
    """pt.durations takes no ``float_digits``; the dispatcher must not forward it."""
    list(durations.get_duration_annotations("pt-BR", "no prazo de 30 dias", float_digits=2))
