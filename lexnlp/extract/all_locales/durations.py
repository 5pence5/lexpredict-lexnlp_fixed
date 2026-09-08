__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator

from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN, LANG_PT, Locale
from lexnlp.extract.common.annotations.duration_annotation import DurationAnnotation
from lexnlp.extract.de.durations import get_duration_annotations as get_duration_annotations_de
from lexnlp.extract.en.durations import get_duration_annotations as get_duration_annotations_en
from lexnlp.extract.pt.durations import get_duration_annotations as get_duration_annotations_pt

ROUTINE_BY_LOCALE = {
    LANG_EN.code: get_duration_annotations_en,
    LANG_DE.code: get_duration_annotations_de,
    LANG_PT.code: get_duration_annotations_pt,
}


def get_duration_annotations(locale: str, text: str, float_digits: int = 4) -> Generator[DurationAnnotation]:
    # Normalise to a registered language before branching -- see the note in
    # ``lexnlp.extract.all_locales.amounts`` for why the branch must never be
    # taken on an unresolved locale string.
    language = Locale(locale).language
    if language not in ROUTINE_BY_LOCALE:
        language = DEFAULT_LANGUAGE.code
    routine = ROUTINE_BY_LOCALE[language]
    # The Portuguese routine reports whole units and takes no ``float_digits``.
    if language == LANG_PT.code:
        yield from routine(text)
    else:
        yield from routine(text, float_digits)
