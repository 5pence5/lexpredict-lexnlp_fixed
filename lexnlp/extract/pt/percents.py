"""Percent extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.de.percents` and :mod:`lexnlp.extract.en.percents`
but tuned to Brazilian number formatting (``.`` as thousands separator,
``,`` as the decimal mark) and to the Portuguese-specific lexicon
(``por cento`` / ``%``).
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

from lexnlp.extract.common.annotations.percent_annotation import PercentAnnotation

# Brazilian numeric pattern: optional thousands "." groups, optional decimal ","
# fraction (e.g. "1.234,56" / "12,5" / "100"). Plain integers are also matched.
_PT_NUMBER_PTN = r"(?P<num>\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?)"

# Surface forms: "12,5%", "12.5 %", "doze por cento" (left out — see notes),
# "12,5 por cento". We also accept the abbreviation "p.c." commonly used in
# Brazilian financial reports.
PERCENT_PTN_RE = re.compile(
    rf"(?P<text>{_PT_NUMBER_PTN}\s*(?P<unit>%|por\s+cento|p\.c\.))(?:\W|$)",
    re.IGNORECASE | re.UNICODE,
)


def _parse_pt_number(raw: str) -> Decimal:
    """Convert a Brazilian-formatted numeric string to :class:`Decimal`."""
    return Decimal(raw.replace(".", "").replace(",", "."))


def get_percent_annotations(text: str, float_digits: int = 4) -> Iterator[PercentAnnotation]:
    """Yield :class:`PercentAnnotation` for every percent expression in *text*."""
    for match in PERCENT_PTN_RE.finditer(text):
        amount = _parse_pt_number(match.group("num"))
        unit = match.group("unit")
        sign = "%" if unit == "%" else "por cento"
        fraction = amount * Decimal("0.01")
        if float_digits is not None:
            fraction = round(fraction, float_digits)
        yield PercentAnnotation(
            coords=match.span("text"),
            text=match.group("text"),
            sign=sign,
            amount=amount,
            fraction=fraction,
            locale="pt",
        )


def get_percents(text: str, float_digits: int = 4) -> Iterator[dict]:
    """Yield dictionaries describing each percent expression in *text*."""
    for ant in get_percent_annotations(text, float_digits):
        yield {
            "location_start": ant.coords[0],
            "location_end": ant.coords[1],
            "source_text": ant.text,
            "unit_name": ant.sign,
            "amount": ant.amount,
            "real_amount": ant.fraction,
        }


def get_percent_annotation_list(text: str, float_digits: int = 4) -> list[PercentAnnotation]:
    """Return all percent annotations in *text* as a list."""
    return list(get_percent_annotations(text, float_digits))


def get_percent_list(text: str, float_digits: int = 4) -> list[dict]:
    """Return all percent annotations as a list of plain dictionaries."""
    return list(get_percents(text, float_digits))


__all__ = [
    "PERCENT_PTN_RE",
    "get_percent_annotation_list",
    "get_percent_annotations",
    "get_percent_list",
    "get_percents",
]
