__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator

from lexnlp.extract.all_locales.languages import LANG_DE, Locale
from lexnlp.extract.common.annotations.court_citation_annotation import CourtCitationAnnotation
from lexnlp.extract.de.court_citations import get_court_citation_annotations as get_court_citation_annotations_de

ROUTINE_BY_LOCALE = {LANG_DE.code: get_court_citation_annotations_de}


def get_court_citation_annotations(
    locale: str, text: str, language: str | None = None
) -> Generator[CourtCitationAnnotation]:
    """
    Yield court citation annotations extracted from `text` using a routine selected by `locale`.

    Falls back to the German extraction routine when no routine is registered for the computed locale language.

    Parameters:
        locale (str): Locale identifier used to select the extraction routine (e.g., "de_DE").
        text (str): Text to scan for court citation annotations.
        language (str | None): Optional language code passed to the extraction routine to refine or override locale-derived language selection.

    Returns:
        Generator[CourtCitationAnnotation]: Yields `CourtCitationAnnotation` objects found in `text`.
    """
    locale_language = Locale(locale).language
    routine = ROUTINE_BY_LOCALE.get(locale_language, ROUTINE_BY_LOCALE[LANG_DE.code])
    # When the caller does not name a language, take it from the locale rather
    # than forwarding ``None``, which left the annotation with no language at
    # all. Fall back to German alongside the routine fallback above.
    annotation_language = language or (
        locale_language if locale_language in ROUTINE_BY_LOCALE else LANG_DE.code
    )
    yield from routine(text, annotation_language)
