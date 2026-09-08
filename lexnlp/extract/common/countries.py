"""Country / currency / language helpers backed by ``pycountry``.

``pycountry`` was only imported in one address-feature helper before this
module. It ships ISO 3166 country/subdivision metadata, ISO 4217
currency metadata, and ISO 639 language codes — all directly useful to
LexNLP's geoentity, money, and locale extractors.

The helpers below keep a tight, caching layer over ``pycountry``'s
lookups so hot paths (e.g. money extraction validating currency codes)
don't re-iterate the full catalogue on every call.
"""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from dataclasses import dataclass
from functools import lru_cache

import pycountry


@dataclass(frozen=True, slots=True)
class CountryInfo:
    """Minimal projection of a ``pycountry.db.Country`` entry."""

    alpha_2: str
    alpha_3: str
    name: str
    official_name: str | None


def _country_to_info(country: object) -> CountryInfo:
    return CountryInfo(
        alpha_2=getattr(country, "alpha_2", ""),
        alpha_3=getattr(country, "alpha_3", ""),
        name=getattr(country, "name", ""),
        official_name=getattr(country, "official_name", None),
    )


@lru_cache(maxsize=4096)
def lookup_country(text: str) -> CountryInfo | None:
    """Look up a country by name, alpha-2, or alpha-3 code."""
    if not text:
        return None
    key = text.strip()
    country = None
    # Try alpha-2 / alpha-3 / name directly.
    for attr in ("alpha_2", "alpha_3", "name"):
        try:
            country = pycountry.countries.get(**{attr: key})
        except LookupError:
            country = None
        if country is not None:
            break
    if country is None:
        # Fall back to case-insensitive name match via iteration; pycountry
        # name indexes are case-sensitive.
        upper = key.upper()
        for c in pycountry.countries:
            if c.name.upper() == upper:
                country = c
                break
            official = getattr(c, "official_name", None)
            if official and official.upper() == upper:
                country = c
                break
    if country is None:
        return None
    return _country_to_info(country)


@lru_cache(maxsize=2048)
def fuzzy_country(text: str, *, max_results: int = 1) -> tuple[CountryInfo, ...]:
    """Fuzzy-match ``text`` against known country names.

    Uses ``pycountry.countries.search_fuzzy`` which handles common OCR
    errors and partial names (e.g. ``"Unitd States"`` → United States).

    Args:
        text: The candidate country string.
        max_results: Number of candidate matches to return. Must be a
            positive integer; defaults to 1 to keep callers on the
            precision side. Raise this to surface alternative matches for
            disambiguation UIs.

    Returns:
        A tuple of :class:`CountryInfo` matches. Empty when ``pycountry``
        reports no candidates.

    Raises:
        ValueError: If ``max_results`` is not a positive integer.
        TypeError: If ``max_results`` is not an ``int`` (booleans rejected).
    """
    # Reject non-int max_results so floats/strings can't accidentally slice.
    if not isinstance(max_results, int) or isinstance(max_results, bool):
        raise TypeError(f"max_results must be an int, got {type(max_results).__name__}")
    if max_results <= 0:
        raise ValueError(f"max_results must be a positive integer, got {max_results!r}")
    if not text:
        return ()
    try:
        matches = pycountry.countries.search_fuzzy(text.strip())
    except (LookupError, KeyError):
        return ()
    return tuple(_country_to_info(m) for m in matches[:max_results])


@lru_cache(maxsize=1)
def currency_codes() -> frozenset[str]:
    """Return the set of ISO 4217 currency alpha codes (``"USD"``, ``"EUR"``…)."""
    return frozenset(c.alpha_3 for c in pycountry.currencies)


def is_currency_code(code: str) -> bool:
    """Check whether ``code`` matches a known ISO 4217 currency code."""
    return code.upper() in currency_codes() if code else False


@lru_cache(maxsize=1)
def language_codes() -> frozenset[str]:
    """Return the set of ISO 639-1 alpha-2 language codes when available."""
    codes: set[str] = set()
    for lang in pycountry.languages:
        alpha_2 = getattr(lang, "alpha_2", None)
        if alpha_2:
            codes.add(alpha_2.lower())
    return frozenset(codes)


def is_language_code(code: str) -> bool:
    """Check whether ``code`` matches a known ISO 639-1 language code."""
    return code.lower() in language_codes() if code else False


__all__ = [
    "CountryInfo",
    "currency_codes",
    "fuzzy_country",
    "is_currency_code",
    "is_language_code",
    "language_codes",
    "lookup_country",
]
