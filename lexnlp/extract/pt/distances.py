"""Distance extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.distances`, swapping the unit dictionary
to the Brazilian Portuguese lexicon and reusing the canonical pt-BR
number pattern from :mod:`lexnlp.extract.pt.amounts` (``NUM_PTN``).

Recognised units: kilometres, metres, centimetres, millimetres, miles
(``milha`` / ``milhas``), nautical miles, yards, feet and inches — the
last three are common in technical specifications imported from English
sources.
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

from lexnlp.extract.common.annotations.distance_annotation import DistanceAnnotation
from lexnlp.extract.pt.amounts import NUM_PTN, _parse_pt_number  # noqa: PLC2701 - reuse pt number parser

# Symbol -> canonical English unit name (matches the EN convention so the
# annotation's ``distance_type`` stays comparable across locales).
DISTANCE_SYMBOL_MAP: dict[str, str] = {
    "km": "kilometer",
    "m": "meter",
    "cm": "centimeter",
    "mm": "millimeter",
    "mi": "mile",
    "yd": "yard",
    "ft": "foot",
    "in": "inch",
    "nmi": "nautical_mile",
}

# pt-BR word -> canonical English unit name. Both singular and plural
# forms are accepted.
DISTANCE_TOKEN_MAP: dict[str, str] = {
    "quilômetros": "kilometer",
    "quilometros": "kilometer",
    "quilômetro": "kilometer",
    "quilometro": "kilometer",
    "metros": "meter",
    "metro": "meter",
    "centímetros": "centimeter",
    "centimetros": "centimeter",
    "centímetro": "centimeter",
    "centimetro": "centimeter",
    "milímetros": "millimeter",
    "milimetros": "millimeter",
    "milímetro": "millimeter",
    "milimetro": "millimeter",
    "milhas": "mile",
    "milha": "mile",
    "milhas náuticas": "nautical_mile",
    "milha náutica": "nautical_mile",
    "jardas": "yard",
    "jarda": "yard",
    "pés": "foot",
    "pes": "foot",
    "pé": "foot",
    "polegadas": "inch",
    "polegada": "inch",
}

_TOKEN_PART = "|".join(re.escape(t) for t in sorted(DISTANCE_TOKEN_MAP, key=len, reverse=True))
_SYMBOL_PART = "|".join(re.escape(s) for s in DISTANCE_SYMBOL_MAP)

DISTANCE_PTN_RE = re.compile(
    rf"(?P<text>(?P<num>{NUM_PTN})\s*"
    rf"(?P<unit>{_TOKEN_PART}|{_SYMBOL_PART}))(?:\W|$)",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


def _resolve_unit(token: str) -> str:
    """Map a unit surface form to its canonical English name."""
    lowered = token.lower()
    return DISTANCE_TOKEN_MAP.get(lowered) or DISTANCE_SYMBOL_MAP.get(lowered, lowered)


def get_distance_annotations(text: str, float_digits: int = 4) -> Iterator[DistanceAnnotation]:
    """Yield :class:`DistanceAnnotation` for every distance in *text*."""
    for match in DISTANCE_PTN_RE.finditer(text):
        amount = _parse_pt_number(match.group("num"))
        if float_digits:
            amount = round(amount, float_digits)
        yield DistanceAnnotation(
            coords=match.span("text"),
            text=match.group("text"),
            amount=amount,
            distance_type=_resolve_unit(match.group("unit")),
            locale="pt",
        )


def get_distances(
    text: str, return_sources: bool = False, float_digits: int = 4
) -> Iterator[tuple[Decimal, str] | tuple[Decimal, str, str]]:
    """Yield ``(amount, distance_type)`` tuples (optionally with source text)."""
    for ant in get_distance_annotations(text, float_digits):
        if return_sources:
            yield ant.amount, ant.distance_type, ant.text
        else:
            yield ant.amount, ant.distance_type


def get_distance_annotation_list(text: str, float_digits: int = 4) -> list[DistanceAnnotation]:
    """Return all distance annotations in *text* as a list."""
    return list(get_distance_annotations(text, float_digits))


def get_distance_list(
    text: str, return_sources: bool = False, float_digits: int = 4
) -> list[tuple[Decimal, str] | tuple[Decimal, str, str]]:
    """Return all distance tuples in *text* as a list."""
    return list(get_distances(text, return_sources, float_digits))


__all__ = [
    "DISTANCE_PTN_RE",
    "DISTANCE_SYMBOL_MAP",
    "DISTANCE_TOKEN_MAP",
    "get_distance_annotation_list",
    "get_distance_annotations",
    "get_distance_list",
    "get_distances",
]
