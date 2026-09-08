"""Amount extraction for Portuguese (pt-BR).

Mirrors :mod:`lexnlp.extract.en.amounts`, tuned to Brazilian Portuguese
formatting (``.`` as thousands separator, ``,`` as the decimal mark) and
to the full pt-BR cardinal vocabulary up through ``trilhão`` /
``trilhões``. Recognises:

- Numeric forms: ``1.234,56``, ``12,5``, ``100``.
- Pure word forms: ``cento e vinte e cinco``, ``mil duzentos e cinquenta``,
  ``dois milhões e quinhentos mil``, ``um trilhão e meio bilhão``.
- Mixed forms: ``5 milhões``, ``2,5 bilhões``, ``1,2 trilhão``.

Anything outside this vocabulary is reported as a single ``None`` amount
so callers can decide whether to fall back to a different parser.
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

from lexnlp.extract.common.annotations.amount_annotation import AmountAnnotation

# ---------------------------------------------------------------------------
# pt-BR cardinal vocabulary (up to ``trilhão`` / ``trilhões``)
# ---------------------------------------------------------------------------

# Units (0-9). ``um/uma`` and ``dois/duas`` carry gender; we accept both
# spellings transparently.
_UNITS = {
    "zero": 0,
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "três": 3,
    "tres": 3,  # "tres" is the unaccented OCR fallback
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
}

# Teens (10-19). ``catorze``/``quatorze`` are both standard; we accept both.
_TEENS = {
    "dez": 10,
    "onze": 11,
    "doze": 12,
    "treze": 13,
    "catorze": 14,
    "quatorze": 14,
    "quinze": 15,
    "dezesseis": 16,
    "dezasseis": 16,  # dezasseis = pt-PT spelling
    "dezessete": 17,
    "dezassete": 17,
    "dezoito": 18,
    "dezenove": 19,
    "dezanove": 19,
}

_TENS = {
    "vinte": 20,
    "trinta": 30,
    "quarenta": 40,
    # ``cinqüenta`` is the pre-1990-Acordo-Ortográfico spelling (with
    # trema), still common in older Brazilian legal documents and OCR;
    # ``cincoenta`` is an older popular variant.
    "cinquenta": 50,
    "cinqüenta": 50,
    "cincoenta": 50,
    "sessenta": 60,
    "setenta": 70,
    "oitenta": 80,
    "noventa": 90,
}

# Hundreds. ``cem`` is the bare form (= 100); ``cento`` is the bound form
# used when followed by tens/units (``cento e dois``).
_HUNDREDS = {
    "cem": 100,
    "cento": 100,
    "duzentos": 200,
    "duzentas": 200,
    "trezentos": 300,
    "trezentas": 300,
    "quatrocentos": 400,
    "quatrocentas": 400,
    "quinhentos": 500,
    "quinhentas": 500,
    "seiscentos": 600,
    "seiscentas": 600,
    "setecentos": 700,
    "setecentas": 700,
    "oitocentos": 800,
    "oitocentas": 800,
    "novecentos": 900,
    "novecentas": 900,
}

_SMALL = {**_UNITS, **_TEENS, **_TENS, **_HUNDREDS}

# Big-scale multipliers. ``mil`` is invariant; ``milhão`` etc. take a
# plural form when ≥ 2.
_MULTIPLIERS = {
    "mil": 1000,
    "milhão": 1_000_000,
    "milhao": 1_000_000,
    "milhões": 1_000_000,
    "milhoes": 1_000_000,
    "bilhão": 1_000_000_000,
    "bilhao": 1_000_000_000,
    "bilhões": 1_000_000_000,
    "bilhoes": 1_000_000_000,
    "trilhão": 1_000_000_000_000,
    "trilhao": 1_000_000_000_000,
    "trilhões": 1_000_000_000_000,
    "trilhoes": 1_000_000_000_000,
}

# Halves and quarters are common in legal writing ("um milhão e meio").
_FRACTIONS = {
    "meio": Decimal("0.5"),
    "meia": Decimal("0.5"),
    "metade": Decimal("0.5"),
    "terço": Decimal("1") / Decimal("3"),
    "terco": Decimal("1") / Decimal("3"),
    "quarto": Decimal("0.25"),
    "quartos": Decimal("0.25"),
}

# Combined word list, longest-first so the regex prefers ``trezentos`` over
# ``trez`` etc. The regex also matches the connector ``e`` between parts.
_WORD_TOKENS = sorted(
    set(_SMALL) | set(_MULTIPLIERS) | set(_FRACTIONS) | {"e"},
    key=len,
    reverse=True,
)
_WORD_PTN = "|".join(re.escape(t) for t in _WORD_TOKENS)
# Recognises a contiguous run of pt-BR number words with optional ``e``
# connectors and inter-token whitespace / hyphens (``vinte-e-um``).
_WORD_NUMBER_RE = re.compile(
    rf"(?<!\w)(?:{_WORD_PTN})(?:[\s-]+(?:{_WORD_PTN}))*(?!\w)",
    re.UNICODE | re.IGNORECASE,
)

# Brazilian-numeric fragment, optionally followed by a multiplier word
# ("2,5 milhões", "1.234,56 mil"). We match this *before* the bare
# word-only matcher so the multiplier is consumed only once.
_PT_NUMBER_PTN = r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?"
_MULT_RE_PART = "|".join(re.escape(t) for t in sorted(_MULTIPLIERS, key=len, reverse=True))
_NUM_WITH_MULT_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_PT_NUMBER_PTN})\s+(?P<mult>{_MULT_RE_PART})(?!\w)",
    re.UNICODE | re.IGNORECASE,
)
_PLAIN_NUMBER_RE = re.compile(rf"(?<![\w.,])(?P<num>{_PT_NUMBER_PTN})(?!\d)")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _parse_pt_number(raw: str) -> Decimal:
    """Convert a Brazilian-formatted numeric string to :class:`Decimal`."""
    return Decimal(raw.replace(".", "").replace(",", "."))


def text_to_number(text: str) -> Decimal | None:
    """Convert a Portuguese cardinal phrase to :class:`Decimal`.

    Accepts everything from ``"um"`` up through ``"um trilhão e quinhentos
    bilhões"``. Returns ``None`` for empty or unrecognisable input.

    The algorithm is the standard sum-product walk over tokens:

    * units / teens / tens / hundreds accumulate into ``current``;
    * a multiplier (``mil``, ``milhão`` …) commits ``current`` (or 1 if
      empty, so bare ``mil`` -> 1000) and resets it;
    * the connector ``"e"`` is ignored; a trailing ``"e meio"`` /
      ``"e meia"`` adds ½ of the most recent multiplier section.
    """
    if not text:
        return None
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return None

    total = Decimal(0)
    current = Decimal(0)
    last_mult = Decimal(1)
    saw_anything = False

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "e":
            i += 1
            continue
        if tok in _SMALL:
            current += Decimal(_SMALL[tok])
            saw_anything = True
        elif tok in _MULTIPLIERS:
            mult = Decimal(_MULTIPLIERS[tok])
            if current == 0:
                current = Decimal(1)
            total += current * mult
            current = Decimal(0)
            last_mult = mult
            saw_anything = True
        elif tok in _FRACTIONS:
            frac = _FRACTIONS[tok]
            # Look ahead one token (skipping a connector "e") for a
            # multiplier so "meio milhão" / "meia bilhão" resolve to
            # 500_000 / 500_000_000 instead of 0.5 (which would be the
            # result of using ``last_mult`` = 1 at that point).
            j = i + 1
            if j < len(tokens) and tokens[j] == "e":
                j += 1
            if j < len(tokens) and tokens[j] in _MULTIPLIERS:
                mult = Decimal(_MULTIPLIERS[tokens[j]])
                if current != 0:
                    # "um e meio milhão" -> (1 + 0.5) * 1_000_000 = 1_500_000
                    total += (current + frac) * mult
                    current = Decimal(0)
                else:
                    # "meio milhão" -> 0.5 * 1_000_000 = 500_000
                    total += frac * mult
                last_mult = mult
                i = j  # advance past the consumed multiplier
            else:
                # "um milhão e meio" -> add half of the most-recent
                # multiplier scale; bare "meio" with no context falls
                # back to multiplier = 1 (i.e. the literal value 0.5).
                total += frac * last_mult
            saw_anything = True
        else:
            # Unknown token; bail and let callers decide what to do.
            return None
        i += 1

    return (total + current) if saw_anything else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_amount_annotations(text: str, float_digits: int = 4) -> Iterator[AmountAnnotation]:
    """Yield :class:`AmountAnnotation` for every numeric/word amount in *text*.

    Order: numeric-with-multiplier matches first (``2,5 milhões``), then
    plain numerics, then word-only phrases. Spans already covered by an
    earlier match are skipped to avoid duplicates.
    """
    seen_spans: list[tuple[int, int]] = []

    def _is_dup(span: tuple[int, int]) -> bool:
        return any(s <= span[0] and span[1] <= e for s, e in seen_spans)

    # 1) "<number> <multiplier>"
    for match in _NUM_WITH_MULT_RE.finditer(text):
        span = match.span()
        seen_spans.append(span)
        amount = _parse_pt_number(match.group("num")) * Decimal(_MULTIPLIERS[match.group("mult").lower()])
        if float_digits:
            amount = round(amount, float_digits)
        yield AmountAnnotation(coords=span, text=match.group(), value=amount, locale="pt")

    # 2) word-only phrases ("cento e vinte e cinco")
    for match in _WORD_NUMBER_RE.finditer(text):
        span = match.span()
        if _is_dup(span):
            continue
        # Reject single-token "e" matches (the regex allows leading "e"
        # because it appears in the alternation; just skip those).
        phrase = match.group()
        stripped = phrase.strip()
        if stripped.lower() == "e":
            continue
        value = text_to_number(stripped)
        if value is None:
            continue
        seen_spans.append(span)
        if float_digits:
            value = round(value, float_digits)
        yield AmountAnnotation(coords=span, text=phrase, value=value, locale="pt")

    # 3) plain numerics
    for match in _PLAIN_NUMBER_RE.finditer(text):
        span = match.span("num")
        if _is_dup(span):
            continue
        amount = _parse_pt_number(match.group("num"))
        if float_digits:
            amount = round(amount, float_digits)
        yield AmountAnnotation(coords=span, text=match.group("num"), value=amount, locale="pt")


def get_amounts(text: str, float_digits: int = 4) -> Iterator[Decimal]:
    """Yield only the :class:`Decimal` values found in *text*."""
    for ant in get_amount_annotations(text, float_digits):
        yield ant.value


def get_amount_annotation_list(text: str, float_digits: int = 4) -> list[AmountAnnotation]:
    """Return all :class:`AmountAnnotation` instances for *text* as a list."""
    return list(get_amount_annotations(text, float_digits))


def get_amount_list(text: str, float_digits: int = 4) -> list[Decimal]:
    """Return all amounts in *text* as a list of :class:`Decimal`."""
    return list(get_amounts(text, float_digits))


# Public number pattern, exposed so dependent modules (``distances``,
# ``ratios``, ``money``) can reuse the canonical pt-BR numeric form.
NUM_PTN: str = _PT_NUMBER_PTN

__all__ = [
    "NUM_PTN",
    "get_amount_annotation_list",
    "get_amount_annotations",
    "get_amount_list",
    "get_amounts",
    "text_to_number",
]
