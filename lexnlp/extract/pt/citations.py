"""Case-citation extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.citations`, tuned to the Brazilian
appellate-court citation forms most commonly seen in legal writing:

- Supremo Tribunal Federal: ``ADI 1.234``, ``ADC 56``, ``ADPF 132``,
  ``RE 123.456``, ``ARE 654.321``, ``HC 999``, ``MS 12345``.
- Superior Tribunal de Justiça: ``REsp 12.345``, ``AgRg no AREsp 234.567``,
  ``EREsp 99.999``, ``HC 88.888``, ``RHC 77.777``.
- Tribunais Superiores: ``CC 12345`` (conflito de competência),
  ``IUJur 1234`` (incidente de uniformização), ``MI 99``.
- Justiça do Trabalho: ``RR 1234567-56.2020.5.04.0001``,
  ``AIRR 1234567-67.2019.5.10.0009`` (CNJ-format process numbers — the
  sequencial component is always 7 digits per Resolução CNJ 65/2008).

Each match is reported as a :class:`CitationAnnotation`. The
``reporter`` field carries the Brazilian abbreviation (e.g. ``REsp``)
and ``source`` holds the canonical case identifier (digits-and-slashes
form). For CNJ-format process numbers we additionally split out the
year, instance ("5"), regional segment and unit number into
``volume_str`` for downstream consumers.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Iterator

import regex as re

from lexnlp.extract.common.annotations.citation_annotation import CitationAnnotation

# Brazilian court reporter abbreviations recognised by the regex. Order
# matters: longest tokens come first so e.g. ``AREsp`` is consumed before
# ``RE``.
_REPORTERS = [
    # STJ family
    "AgRg no AREsp",
    "AgInt no AREsp",
    "AgRg no REsp",
    "AgInt no REsp",
    "EDcl no REsp",
    "EREsp",
    "AREsp",
    "REsp",
    "RHC",
    "RMS",
    # STF family
    "ADPF",
    "ADI",
    "ADC",
    "ADO",
    "ARE",
    "RE",
    "HC",
    "MS",
    "MI",
    # Tribunais Superiores
    "IUJur",
    "CC",
    "ED",
    "QO",
]
_REPORTERS_SORTED = sorted(_REPORTERS, key=len, reverse=True)
_REPORTER_PART = "|".join(re.escape(r) for r in _REPORTERS_SORTED)

# Numeric body: ``12.345``, ``123.456``, ``999``, ``MS 12345``,
# ``CC 12345`` — accept either grouped form (with thousands dots) OR
# ungrouped runs of 4+ digits common in older case-law citations.
_NUMBER_PART = r"(?:\d{1,3}(?:\.\d{3})*|\d{4,})"
_UF_LIST = "AC|AL|AM|AP|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO"
_YEAR_PART = r"\d{2,4}"

# "REsp 12.345/SP", "RE 123.456", "ADI 1.234"
CASE_CITATION_RE = re.compile(
    rf"(?<![\w./])(?P<full>(?P<reporter>{_REPORTER_PART})\s+"
    rf"(?P<source>(?P<number>{_NUMBER_PART})"
    rf"(?:/(?P<uf>{_UF_LIST}))?"
    rf"(?:[/\s](?P<year>{_YEAR_PART}))?"
    rf"))(?!\d)",
    re.UNICODE,
)

# CNJ-format process number, mandatory since 2010 across all Brazilian
# courts: NNNNNNN-DD.AAAA.J.TR.OOOO (7-2-4-1-2-4 digits with separators).
CNJ_PROCESS_RE = re.compile(
    r"(?<!\d)"
    r"(?P<full>(?P<sequencial>\d{7})-(?P<digito>\d{2})\."
    r"(?P<ano>\d{4})\.(?P<segmento>\d)\.(?P<tribunal>\d{2})\.(?P<origem>\d{4}))"
    r"(?!\d)"
)


def _parse_int(value: str | None) -> int | None:
    """Convert ``value`` to ``int`` after stripping thousands separators."""
    if value is None:
        return None
    digits = value.replace(".", "")
    return int(digits) if digits.isdigit() else None


def _normalise_year(value: str | None) -> int | None:
    """Coerce a 2- or 4-digit year string to the canonical 4-digit form."""
    if value is None:
        return None
    if len(value) == 2 and value.isdigit():
        # Treat ``/85`` as 1985 and ``/20`` as 2020 — Brazilian case-law
        # citations span the 20th and 21st century, so we pivot at 50.
        n = int(value)
        return 1900 + n if n >= 50 else 2000 + n
    return _parse_int(value)


def get_case_citation_annotations(text: str) -> Iterator[CitationAnnotation]:
    """Yield :class:`CitationAnnotation` for short-form Brazilian case cites."""
    for match in CASE_CITATION_RE.finditer(text):
        yield CitationAnnotation(
            coords=match.span("full"),
            text=match.group("full"),
            reporter=match.group("reporter"),
            source=match.group("source"),
            volume=_parse_int(match.group("number")),
            year=_normalise_year(match.group("year")),
            court=match.group("uf"),
            locale="pt",
        )


def get_cnj_process_annotations(text: str) -> Iterator[CitationAnnotation]:
    """Yield :class:`CitationAnnotation` for CNJ-format process numbers.

    The CNJ process number embeds the year, judicial segment, regional
    tribunal and unit of origin; we parse them out into the annotation's
    ``year`` / ``volume_str`` fields so they're inspectable without
    re-parsing the source string.
    """
    for match in CNJ_PROCESS_RE.finditer(text):
        cnj = match.groupdict()
        # ``volume_str`` carries the structural breakdown so callers can
        # display it cleanly without re-parsing.
        volume_str = f"seg={cnj['segmento']} tr={cnj['tribunal']} orig={cnj['origem']}"
        yield CitationAnnotation(
            coords=match.span("full"),
            text=match.group("full"),
            source=match.group("full"),
            year=int(cnj["ano"]),
            volume_str=volume_str,
            locale="pt",
        )


def get_citation_annotations(text: str) -> Iterator[CitationAnnotation]:
    """Yield every Brazilian case citation we know how to extract.

    Order: short-form abbreviations first (``REsp 12.345``), then CNJ
    process numbers (``1234567-89.2020.8.26.0100``).
    """
    yield from get_case_citation_annotations(text)
    yield from get_cnj_process_annotations(text)


def get_citations(text: str) -> Iterator[str]:
    """Yield the surface form of every case citation in *text*."""
    for ant in get_citation_annotations(text):
        yield ant.text


def get_citation_annotation_list(text: str) -> list[CitationAnnotation]:
    """Return all citation annotations in *text* as a list."""
    return list(get_citation_annotations(text))


def get_citation_list(text: str) -> list[str]:
    """Return all citation surface strings in *text* as a list."""
    return list(get_citations(text))


__all__ = [
    "CASE_CITATION_RE",
    "CNJ_PROCESS_RE",
    "get_case_citation_annotations",
    "get_citation_annotation_list",
    "get_citation_annotations",
    "get_citation_list",
    "get_citations",
    "get_cnj_process_annotations",
]
