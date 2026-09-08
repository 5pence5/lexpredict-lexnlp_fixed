"""Coverage tests for lexnlp.extract.common.copyrights.copyright_parsing_methods."""

from __future__ import annotations

from lexnlp.extract.common import year_parser
from lexnlp.extract.common.copyrights.copyright_parsing_methods import CopyrightParsingMethods
from lexnlp.extract.common.copyrights.copyright_pattern_found import CopyrightPatternFound
from lexnlp.extract.common.pattern_found import PatternFound


class EnglishCopyrightMethods(CopyrightParsingMethods):
    def init_trigger_words(self) -> None:
        self.trigger_words = r"copyright|©"


def _parser() -> EnglishCopyrightMethods:
    return EnglishCopyrightMethods()


class TestInit:
    def test_base_class_leaves_trigger_words_empty(self) -> None:
        parser = CopyrightParsingMethods()
        assert parser.trigger_words == ""
        assert parser.reg_trigger_words.search("anything") is not None
        assert parser.reg_company_name.pattern == r"[\p{L}\s]+"

    def test_subclass_compiles_year_regexes(self) -> None:
        parser = _parser()
        assert parser.trigger_words == r"copyright|©"
        assert "copyright" in parser.reg_word_c_years.pattern
        assert "copyright" in parser.reg_c_years_word.pattern
        assert parser.reg_trigger_words.search("Copyright 1996") is not None
        assert parser.reg_trigger_words.search("no marker") is None


class TestMatchWordCYears:
    def test_returns_empty_without_trigger(self) -> None:
        assert _parser().match_word_c_years("Siemens 1996 – 2019") == []

    def test_extracts_company_and_year_span(self) -> None:
        matches = _parser().match_word_c_years("Siemens © 1996 – 2019")
        assert len(matches) == 1
        match = matches[0]
        assert isinstance(match, CopyrightPatternFound)
        assert match.name == "Siemens © 1996 – 2019"
        assert match.company == "Siemens"
        assert match.start_year == 1996
        assert match.end_year == 2019
        assert match.probability == 100
        assert match.start == 0
        assert match.end == len("Siemens © 1996 – 2019")

    def test_single_year_sets_end_year_only(self) -> None:
        matches = _parser().match_word_c_years("Foo copyright 1996")
        assert len(matches) == 1
        assert matches[0].start_year == 0
        assert matches[0].end_year == 1996
        assert matches[0].company == "Foo "


class TestMatchCYearsWord:
    def test_returns_empty_without_trigger(self) -> None:
        assert _parser().match_c_years_word("1996 – 2019, Siemens") == []

    def test_extracts_trailing_company(self) -> None:
        matches = _parser().match_c_years_word("Copyright 1996 – 2019, Siemens")
        assert len(matches) == 1
        match = matches[0]
        assert match.name == "Copyright 1996 – 2019, Siemens"
        assert match.company == "Siemen"
        assert match.start_year == 1996
        assert match.end_year == 2019

    def test_single_year_and_company_after_marker(self) -> None:
        matches = _parser().match_c_years_word("copyright 2019 Siemens AG")
        assert len(matches) == 1
        assert matches[0].start_year == 0
        assert matches[0].end_year == 2019
        assert matches[0].company == "Siemens A"

    def test_phrase_starting_with_marker_and_company_first_does_not_match(self) -> None:
        assert _parser().match_c_years_word("© Siemens 1996 – 2019") == []
        assert _parser().match_word_c_years("© Siemens 1996 – 2019") == []


class TestPreProcessFoundMatches:
    def test_zero_one_and_two_years(self) -> None:
        parser = _parser()

        none = PatternFound()
        none.name = "copyright only Siemens"
        none.start = 0
        none.end = len(none.name)
        none.probability = 80

        one = PatternFound()
        one.name = "copyright 2011 Acme"
        one.start = 0
        one.end = len(one.name)
        one.probability = 90

        two = PatternFound()
        two.name = "copyright 2019-1996 Acme Corp"
        two.start = 0
        two.end = len(two.name)
        two.probability = 100

        processed = parser.pre_process_found_matches([none, one, two], "end")
        assert [item.start_year for item in processed] == [0, 0, 1996]
        assert [item.end_year for item in processed] == [0, 2011, 2019]
        assert processed[0].company == " only Siemen"
        assert processed[1].company == "Acm"
        assert processed[2].company == "Acme Cor"
        assert all(isinstance(item, CopyrightPatternFound) for item in processed)

    def test_empty_company_is_left_blank(self) -> None:
        parser = _parser()
        match = PatternFound()
        match.name = "copyright 1996"
        match.start = 0
        match.end = len(match.name)
        match.probability = 100
        processed = parser.pre_process_found_matches([match], "start")
        assert processed[0].company == ""
        assert processed[0].end_year == 1996
        assert processed[0].start_year == 0


class TestGetCompanyNameFromMatch:
    def test_start_option_resets_when_year_begins_at_zero(self) -> None:
        parser = _parser()
        text = "1996 Siemens"
        years = year_parser.year_parser.get_years_with_coords_from_string(text)
        assert years[0][1] == 0
        assert parser.get_company_name_from_match(text, "start", years) == "Siemen"

    def test_end_option_resets_when_start_is_last_index(self) -> None:
        parser = _parser()
        text = "1996XY"
        years = year_parser.year_parser.get_years_with_coords_from_string(text)
        assert years[-1][2] + 1 == len(text) - 1
        assert parser.get_company_name_from_match(text, "end", years) == "X"

    def test_end_option_reads_text_after_last_year(self) -> None:
        parser = _parser()
        text = "copyright 2011 Acme"
        years = year_parser.year_parser.get_years_with_coords_from_string(text)
        assert parser.get_company_name_from_match(text, "end", years) == "Acm"

    def test_unknown_option_scans_almost_whole_text(self) -> None:
        parser = _parser()
        assert parser.get_company_name_from_match("Copyright Siemens", "middle", []) == " Siemen"

    def test_returns_empty_when_only_trigger_and_year_remain(self) -> None:
        parser = _parser()
        text = "copyright 1996"
        years = year_parser.year_parser.get_years_with_coords_from_string(text)
        assert parser.get_company_name_from_match(text, "start", years) == ""

    def test_no_years_start_search_uses_full_letter_runs(self) -> None:
        parser = _parser()
        assert parser.get_company_name_from_match("Siemens copyright here", "start", []) == "Siemens  her"
