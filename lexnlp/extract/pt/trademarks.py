"""Trademark and entity-suffix extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.trademarks` for the universal trademark
markers (``™``, ``®``, ``(R)``, ``TM``) and additionally surfaces
Brazilian commercial-entity suffixes that commonly mark a company name in
contracts (``Ltda.``, ``S.A.``, ``S/A``, ``EIRELI``, ``MEI``, ``LTDA.``).

The trademark regex is locale-agnostic, so we re-use the same shape as
the English module. The named-phrase extraction (``NPExtractor``) used in
:mod:`lexnlp.extract.en.trademarks` is English-only (NLTK pre-trained
chunker), so we instead anchor the trademark regex to the surrounding
context and emit one annotation per direct hit. Callers who need
phrase-level grouping should fall back to :mod:`lexnlp.extract.ner`.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Iterator

import regex as re

from lexnlp.extract.common.annotations.trademark_annotation import TrademarkAnnotation

# Universal trademark markers — same shape as the EN regex but matched
# against the full source text rather than NPExtractor phrases.
TRADEMARK_PTN_RE = re.compile(
    r"[\p{Lu}\p{N}][^\)]{0,80}?(?:[a-z]TM|[\s\(]TM(?:\W|$)|™|\s*\(R\)|Ⓡ|®)",
    re.UNICODE,
)

# Brazilian commercial-entity suffixes. Captures the immediately preceding
# capitalised token sequence as the company name. Matches ``XYZ Ltda.``,
# ``Foo Bar S.A.``, ``Empresa S/A``, ``Foo EIRELI``.
ENTITY_SUFFIX_RE = re.compile(
    r"(?P<full>(?:[\p{Lu}][\p{L}\.&'-]*\s+){1,5}"
    r"(?P<suffix>Ltda\.?|S\.?A\.?|S/A|EIRELI|ME|MEI|EPP|LTDA\.?))"
    r"(?!\w)",
    re.UNICODE,
)


def get_trademark_annotations(text: str) -> Iterator[TrademarkAnnotation]:
    """Yield :class:`TrademarkAnnotation` for every ™/® mark in *text*.

    The match is anchored on the trademark symbol itself and includes up
    to 80 chars of preceding context (capped to keep matches local). Only
    matches whose span lies fully within ``text`` are yielded.
    """
    if not TRADEMARK_PTN_RE.search(text):
        return
    for match in TRADEMARK_PTN_RE.finditer(text):
        coords = match.span()
        if coords[1] > len(text):
            # finditer spans always lie within [0, len(text)], so clamping is impossible.
            coords = (coords[0], len(text))  # pragma: no cover
        yield TrademarkAnnotation(coords=coords, trademark=match.group(), text=match.group(), locale="pt")


def get_trademarks(text: str) -> Iterator[str]:
    """Yield the surface form of every trademark match in *text*."""
    for ant in get_trademark_annotations(text):
        yield ant.trademark


def get_trademark_list(text: str) -> list[str]:
    """Return all trademark strings in *text* as a list."""
    return list(get_trademarks(text))


def get_trademark_annotation_list(text: str) -> list[TrademarkAnnotation]:
    """Return all trademark annotations in *text* as a list."""
    return list(get_trademark_annotations(text))


def get_entity_suffix_annotations(text: str) -> Iterator[TrademarkAnnotation]:
    """Yield Brazilian commercial-entity suffixes (``Ltda.`` / ``S.A.`` …).

    These are not technically trademarks but live in the same recognition
    layer in legal contracts: they mark a company name and are commonly
    extracted alongside ``®``/``™`` matches. The :class:`TrademarkAnnotation`
    container is used so callers can iterate a single typed stream.
    """
    for match in ENTITY_SUFFIX_RE.finditer(text):
        yield TrademarkAnnotation(
            coords=match.span("full"),
            trademark=match.group("full"),
            text=match.group("full"),
            locale="pt",
        )


def get_entity_suffix_list(text: str) -> list[TrademarkAnnotation]:
    """Return all Brazilian commercial-entity suffix annotations as a list."""
    return list(get_entity_suffix_annotations(text))


__all__ = [
    "ENTITY_SUFFIX_RE",
    "TRADEMARK_PTN_RE",
    "get_entity_suffix_annotations",
    "get_entity_suffix_list",
    "get_trademark_annotation_list",
    "get_trademark_annotations",
    "get_trademark_list",
    "get_trademarks",
]
