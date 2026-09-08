__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.common.annotations.copyright_annotation import CopyrightAnnotation
from lexnlp.extract.common.copyrights.copyright_parser import CopyrightParser
from lexnlp.extract.common.copyrights.copyright_pattern_found import CopyrightPatternFound
from lexnlp.extract.common.pattern_found import PatternFound
from lexnlp.utils.lines_processing.line_processor import LineOrPhrase, LineSplitParams


def _make_parser() -> CopyrightParser:
    return CopyrightParser(parsing_functions=[], split_params=LineSplitParams())


def _make_pattern(
    name: str = "Copyright 2015-2021, ContraxSuite, LLC",
    start: int = 0,
    end: int = 10,
    company: str = "ContraxSuite, LLC",
    start_year: int = 2015,
    end_year: int = 2021,
) -> CopyrightPatternFound:
    base = PatternFound()
    base.name = name
    base.start = start
    base.end = end
    base.probability = 100
    found = CopyrightPatternFound(base)
    found.company = company
    found.start_year = start_year
    found.end_year = end_year
    return found


class TestMakeAnnotationFromPattern:
    def test_copies_company_and_years(self) -> None:
        phrase = LineOrPhrase(text="Copyright 2015-2021, ContraxSuite, LLC", start=0)
        ant = _make_parser().make_annotation_from_pattern("en", _make_pattern(), phrase)
        assert isinstance(ant, CopyrightAnnotation)
        assert ant.company == "ContraxSuite, LLC"
        assert ant.year_start == 2015
        assert ant.year_end == 2021

    def test_copies_name_coords_text_and_locale(self) -> None:
        text = "xx Copyright 2020 Acme yy"
        phrase = LineOrPhrase(text=text, start=0)
        ant = _make_parser().make_annotation_from_pattern(
            "en", _make_pattern(name="Copyright 2020 Acme", start=3, end=21), phrase
        )
        assert ant.name == "Copyright 2020 Acme"
        assert ant.coords == (3, 21)
        assert ant.text == text[3:21]
        assert ant.locale == "en"

    def test_empty_company_and_zero_years(self) -> None:
        phrase = LineOrPhrase(text="Copyright Acme", start=0)
        ant = _make_parser().make_annotation_from_pattern(
            "en",
            _make_pattern(name="Copyright Acme", start=0, end=14, company="", start_year=0, end_year=0),
            phrase,
        )
        assert ant.company == ""
        assert not ant.year_start
        assert not ant.year_end
        assert ant.text == "Copyright Acme"


class TestParseIntegration:
    def test_parse_yields_copyright_annotation(self) -> None:
        text = "Copyright 2020 Acme Corp"

        def find_copyrights(phrase_text: str) -> list[PatternFound]:
            found = CopyrightPatternFound()
            found.name = phrase_text
            found.start = 0
            found.end = len(phrase_text)
            found.probability = 100
            found.company = "Acme Corp"
            found.start_year = 2020
            found.end_year = 2020
            return [found]

        parser = CopyrightParser(parsing_functions=[find_copyrights], split_params=LineSplitParams())
        annotations = list(parser.parse(text, locale="en"))
        assert len(annotations) == 1
        ant = annotations[0]
        assert isinstance(ant, CopyrightAnnotation)
        assert ant.company == "Acme Corp"
        assert ant.year_start == 2020
        assert ant.year_end == 2020
        assert ant.text == text
