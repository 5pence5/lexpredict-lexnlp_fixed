__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator

from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN, LANG_PT, Locale
from lexnlp.extract.common.annotations.amount_annotation import AmountAnnotation
from lexnlp.extract.de.amounts import get_amount_annotations as get_amount_annotations_de
from lexnlp.extract.en.amounts import get_amount_annotations as get_amount_annotations_en
from lexnlp.extract.pt.amounts import get_amount_annotations as get_amount_annotations_pt

ROUTINE_BY_LOCALE = {
    LANG_EN.code: get_amount_annotations_en,
    LANG_DE.code: get_amount_annotations_de,
    LANG_PT.code: get_amount_annotations_pt,
}


def get_amount_annotations(
    locale: str,
    text: str,
    extended_sources: bool = True,
    float_digits: int = 4,
) -> Generator[AmountAnnotation]:
    # Resolve the *effective* language first. Branching the calling convention
    # on the requested locale while selecting the routine with a fallback lets
    # the two disagree: an unregistered locale (fr, it, ...) picks the English
    # routine but takes a non-English branch. That is exactly how the date
    # dispatcher started raising TypeError. Normalising here keeps the branch
    # and the routine permanently in step.
    language = Locale(locale).language
    if language not in ROUTINE_BY_LOCALE:
        language = DEFAULT_LANGUAGE.code
    routine = ROUTINE_BY_LOCALE[language]
    # The per-locale routines do not share a parameter order: English takes
    # (text, extended_sources, float_digits), German takes
    # (text, float_digits, return_sources) and Portuguese takes
    # (text, float_digits) with no sources switch at all. Dispatching
    # positionally silently passed ``extended_sources`` as ``float_digits``
    # for German, so German amounts were rounded to one decimal place instead
    # of four. Always pass by name.
    if language == LANG_DE.code:
        yield from routine(text=text, float_digits=float_digits, return_sources=extended_sources)
    elif language == LANG_PT.code:
        yield from routine(text=text, float_digits=float_digits)
    else:
        yield from routine(text=text, extended_sources=extended_sources, float_digits=float_digits)
