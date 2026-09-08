__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from collections.abc import Generator
from datetime import datetime

from lexnlp.extract.all_locales.languages import (
    DEFAULT_LANGUAGE,
    LANG_DE,
    LANG_EN,
    LANG_ES,
    LANG_PT,
    Locale,
)
from lexnlp.extract.common.annotations.date_annotation import DateAnnotation
from lexnlp.extract.de.dates import get_date_annotations as get_date_annotations_de
from lexnlp.extract.en.dates import get_date_annotations as get_date_annotations_en
from lexnlp.extract.es.dates import get_date_annotations as get_date_annotations_es
from lexnlp.extract.pt.dates import get_date_annotations as get_date_annotations_pt

ROUTINE_BY_LOCALE = {
    LANG_EN.code: get_date_annotations_en,
    LANG_DE.code: get_date_annotations_de,
    LANG_ES.code: get_date_annotations_es,
    LANG_PT.code: get_date_annotations_pt,
}


def get_date_annotations(
    locale: str, text: str, strict: bool | None = None, base_date: datetime | None = None, threshold: float = 0.50
) -> Generator[DateAnnotation]:
    language = Locale(locale).language
    routine = ROUTINE_BY_LOCALE.get(language, ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code])
    # Only the English routine accepts ``base_date`` and ``threshold``; the
    # German, Spanish and Portuguese parsers take (text, locale, strict).
    # Passing five positional arguments raised TypeError for every non-English
    # locale, so this dispatcher was unusable outside English. Pass by name and
    # only forward what each routine accepts.
    if language == LANG_EN.code:
        yield from routine(text=text, strict=strict, locale=locale, base_date=base_date, threshold=threshold)
    else:
        # These parsers read attributes off a Locale, not a language string,
        # and declare ``strict: bool = True``. Forwarding the dispatcher's
        # ``None`` default reaches dateparser as STRICT_PARSING=None, which it
        # rejects, so leave ``strict`` out and let each parser default.
        extra = {} if strict is None else {"strict": strict}
        yield from routine(text=text, locale=Locale(locale), **extra)
