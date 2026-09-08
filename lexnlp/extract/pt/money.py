"""Money extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.de.money` and :mod:`lexnlp.extract.en.money`,
tuned to the Brazilian Real and to Brazilian numeric formatting (``.`` as
thousands separator, ``,`` as the decimal mark). Recognises:

- ``R$ 1.234,56`` / ``R$1.234,56`` — Brazilian Real with the
  ``R$`` prefix (with or without a thin space).
- ``1.234,56 reais`` — Real-suffixed amounts (``real`` / ``reais``).
- ``1.234,56 BRL`` / ``BRL 1.234,56`` — ISO-4217 prefix or suffix.
- ``USD 100,00`` / ``EUR 100,00`` / ``GBP 100,00`` — common foreign
  currencies in Brazilian contracts (with both pre/post placement).
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Iterator
from decimal import Decimal

import regex as re

from lexnlp.extract.common.annotations.money_annotation import MoneyAnnotation

# Brazilian Real and common foreign currencies seen in Brazilian contracts.
_CURRENCY_SYMBOL_TO_CODE: dict[str, str] = {
    "r$": "BRL",
    "us$": "USD",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}

_CURRENCY_NAME_TO_CODE: dict[str, str] = {
    "real": "BRL",
    "reais": "BRL",
    "dólar": "USD",
    "dolar": "USD",
    "dólares": "USD",
    "dolares": "USD",
    "euro": "EUR",
    "euros": "EUR",
    "libra": "GBP",
    "libras": "GBP",
    "iene": "JPY",
    "ienes": "JPY",
    "yen": "JPY",
    "iuan": "CNY",
    "yuan": "CNY",
}

_CURRENCY_CODE_RE_PART = "|".join(sorted({"BRL", "USD", "EUR", "GBP", "JPY", "CNY"}))
_CURRENCY_NAME_RE_PART = "|".join(sorted(_CURRENCY_NAME_TO_CODE.keys(), key=len, reverse=True))
_CURRENCY_SYMBOL_RE_PART = "R\\$|US\\$|\\$|€|£|¥"

# Brazilian numeric pattern: optional thousands "." groups, optional decimal ","
# fraction (e.g. "1.234,56" / "12,5" / "100").
_PT_NUMBER_PTN = r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?"

# Symbol/code prefix: "R$ 100,00", "USD 1.234,56". The leading
# ``(?<!\w)`` lookbehind prevents matches inside larger tokens like
# ``XBRL100`` (which would otherwise yield a spurious ``BRL100`` match).
_MONEY_PREFIX_RE = re.compile(
    rf"(?<!\w)(?P<text>(?P<currency>{_CURRENCY_SYMBOL_RE_PART}|{_CURRENCY_CODE_RE_PART})\s*"
    rf"(?P<num>{_PT_NUMBER_PTN}))(?!\d)",
    re.IGNORECASE | re.UNICODE,
)
# Suffix form: "100,00 reais", "1.234,56 BRL".
_MONEY_SUFFIX_RE = re.compile(
    rf"(?<!\w)(?P<text>(?P<num>{_PT_NUMBER_PTN})\s+"
    rf"(?P<currency>{_CURRENCY_NAME_RE_PART}|{_CURRENCY_CODE_RE_PART}))",
    re.IGNORECASE | re.UNICODE,
)


def _parse_pt_number(raw: str) -> Decimal:
    """Convert a Brazilian-formatted numeric string to :class:`Decimal`."""
    return Decimal(raw.replace(".", "").replace(",", "."))


def _normalize_currency(token: str) -> str:
    """Map a currency symbol/name/code to its ISO 4217 code."""
    lowered = token.lower()
    if lowered in _CURRENCY_SYMBOL_TO_CODE:
        return _CURRENCY_SYMBOL_TO_CODE[lowered]
    if lowered in _CURRENCY_NAME_TO_CODE:
        return _CURRENCY_NAME_TO_CODE[lowered]
    return token.upper()


def get_money_annotations(text: str, float_digits: int = 4) -> Iterator[MoneyAnnotation]:
    """Yield :class:`MoneyAnnotation` for every monetary expression in *text*.

    Order: prefix-form matches first (``R$ 100``), then suffix-form
    (``100 reais``). Spans returned as ``(start, end)`` of the full
    surface text including the currency token.
    """
    seen_spans: set[tuple[int, int]] = set()
    for match in _MONEY_PREFIX_RE.finditer(text):
        span = match.span("text")
        if span in seen_spans:
            continue
        seen_spans.add(span)
        amount = _parse_pt_number(match.group("num"))
        if float_digits is not None:
            amount = round(amount, float_digits)
        yield MoneyAnnotation(
            coords=span,
            text=match.group("text"),
            amount=amount,
            currency=_normalize_currency(match.group("currency")),
            locale="pt",
        )
    for match in _MONEY_SUFFIX_RE.finditer(text):
        span = match.span("text")
        # Skip a suffix match whose span overlaps any already-yielded
        # prefix match. ``R$ 100 reais`` would otherwise yield two
        # annotations: prefix (``R$ 100``) and suffix (``100 reais``)
        # — overlapping but neither contains the other.
        if any(not (span[1] <= s or e <= span[0]) for s, e in seen_spans):
            continue
        amount = _parse_pt_number(match.group("num"))
        if float_digits is not None:
            amount = round(amount, float_digits)
        yield MoneyAnnotation(
            coords=span,
            text=match.group("text"),
            amount=amount,
            currency=_normalize_currency(match.group("currency")),
            locale="pt",
        )


def get_money(text: str, float_digits: int = 4) -> Iterator[dict]:
    """Yield dictionaries describing each monetary expression in *text*."""
    for ant in get_money_annotations(text, float_digits):
        yield {
            "location_start": ant.coords[0],
            "location_end": ant.coords[1],
            "source_text": ant.text,
            "amount": ant.amount,
            "currency": ant.currency,
        }


def get_money_annotation_list(text: str, float_digits: int = 4) -> list[MoneyAnnotation]:
    """Return all money annotations in *text* as a list."""
    return list(get_money_annotations(text, float_digits))


def get_money_list(text: str, float_digits: int = 4) -> list[dict]:
    """Return all money annotations as a list of plain dictionaries."""
    return list(get_money(text, float_digits))


__all__ = [
    "get_money",
    "get_money_annotation_list",
    "get_money_annotations",
    "get_money_list",
]
