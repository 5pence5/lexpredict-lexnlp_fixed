__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator

from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN, LANG_PT, Locale
from lexnlp.extract.common.annotations.percent_annotation import PercentAnnotation
from lexnlp.extract.de.percents import get_percent_annotations as get_percent_annotations_de
from lexnlp.extract.en.percents import get_percent_annotations as get_percent_annotations_en
from lexnlp.extract.pt.percents import get_percent_annotations as get_percent_annotations_pt

ROUTINE_BY_LOCALE = {
    LANG_EN.code: get_percent_annotations_en,
    LANG_DE.code: get_percent_annotations_de,
    LANG_PT.code: get_percent_annotations_pt,
}


def get_percent_annotations(
    locale: str,
    text: str,
    float_digits: int = 4,
) -> Generator[PercentAnnotation]:
    routine = ROUTINE_BY_LOCALE.get(Locale(locale).language, ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code])
    yield from routine(text, float_digits)
