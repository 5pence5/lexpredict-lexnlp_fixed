"""Ratio extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.ratios`, swapping in the canonical pt-BR
number pattern. The pattern (:data:`RATIO_PTN_RE`) accepts the Brazilian
connector words ``para`` / ``por`` and the universal separators ``:`` /
``/`` / ``-`` between **numeric** operands (``2 para 1``, ``2 por 1``,
``3:1``, ``3/1``, ``3-1``). Word-form numbers (e.g. ``três para um``)
are NOT matched: only :data:`NUM_PTN`-shaped digit forms are accepted.
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

from lexnlp.extract.common.annotations.ratio_annotation import RatioAnnotation
from lexnlp.extract.pt.amounts import NUM_PTN, _parse_pt_number  # noqa: PLC2701 - reuse pt number parser

RATIO_PTN_RE = re.compile(
    rf"(?P<text>(?P<left>{NUM_PTN})\s*"
    rf"(?:para|por|:|/|-)\s*"
    # ``(?!\d)`` stops the engine backtracking the right operand to a shorter
    # number purely to escape the clock-time guard below. Without it,
    # "10:30 a.m." matches as 10/3, because dropping the "0" moves "a.m."
    # out of the lookahead's reach.
    rf"(?P<right>{NUM_PTN})(?!\d))(?!\s*[ap]\.?m(?:\W|$))",
    re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


def get_ratio_annotations(text: str, float_digits: int = 4) -> Iterator[RatioAnnotation]:
    """Yield :class:`RatioAnnotation` for every ratio expression in *text*.

    The ratio is computed as ``left / right`` and stored alongside the
    individual operands. Matches followed by a time-of-day suffix
    (``a.m.`` / ``p.m.``) are ignored to avoid false positives on clock
    expressions.
    """
    for match in RATIO_PTN_RE.finditer(text):
        left = _parse_pt_number(match.group("left"))
        right = _parse_pt_number(match.group("right"))
        if right == 0:
            continue
        ratio = left / right
        if float_digits:
            left = round(left, float_digits)
            right = round(right, float_digits)
            ratio = round(ratio, float_digits)
        yield RatioAnnotation(
            coords=match.span("text"),
            text=match.group("text"),
            left=left,
            right=right,
            ratio=ratio,
            locale="pt",
        )


def get_ratios(
    text: str, return_sources: bool = False, float_digits: int = 4
) -> Iterator[tuple[Decimal, Decimal, Decimal] | tuple[Decimal, Decimal, Decimal, str]]:
    """Yield ``(left, right, ratio)`` tuples (optionally with source text)."""
    for ant in get_ratio_annotations(text, float_digits=float_digits):
        if return_sources:
            yield ant.left, ant.right, ant.ratio, ant.text
        else:
            yield ant.left, ant.right, ant.ratio


def get_ratio_annotation_list(text: str, float_digits: int = 4) -> list[RatioAnnotation]:
    """Return all ratio annotations in *text* as a list."""
    return list(get_ratio_annotations(text, float_digits))


def get_ratio_list(
    text: str, return_sources: bool = False, float_digits: int = 4
) -> list[tuple[Decimal, Decimal, Decimal] | tuple[Decimal, Decimal, Decimal, str]]:
    """Return all ratio tuples in *text* as a list."""
    return list(get_ratios(text, return_sources, float_digits))


__all__ = [
    "RATIO_PTN_RE",
    "get_ratio_annotation_list",
    "get_ratio_annotations",
    "get_ratio_list",
    "get_ratios",
]
