"""Coverage tests for lexnlp/extract/common/date_parsing/datefinder.py.

Covers DateFragment helpers, DateFinder.find_dates source/index/strict
branches (including the unparseable-fragment skip), the parse_date_string
short-string guard and locale double-ValueError path, the dateutil fallback
with timezone re-attachment, and the module-level find_dates entry point.
"""

from __future__ import annotations

import datetime

from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.common.date_parsing.datefinder import DateFinder, DateFragment, find_dates

BASE_DATE = datetime.datetime(2000, 1, 1)

EMPTY_CAPTURES: dict[str, list[str]] = {
    "delimiters": [],
    "extra_tokens": [],
    "digits": [],
    "time": [],
    "digits_modifier": [],
    "days": [],
    "months": [],
    "timezones": [],
    "time_periods": [],
    "hours": [],
    "minutes": [],
    "seconds": [],
    "microseconds": [],
}


def _finder() -> DateFinder:
    return DateFinder(base_date=BASE_DATE)


class TestDateFragment:
    def test_repr_contains_match_indices_and_captures(self) -> None:
        frag = DateFragment()
        frag.match_str = "January 5, 2024"
        frag.indices = (0, 15)
        frag.captures = {"months": ["January"], "digits": ["5", "2024"]}
        text = repr(frag)
        assert "January 5, 2024" in text
        assert "[0, 15]" in text
        assert '"months": [["January"]]' in text or '"months"' in text
        assert "January" in text

    def test_repr_empty_captures(self) -> None:
        frag = DateFragment()
        text = repr(frag)
        assert "[0, 0]" in text
        assert "Captures:" in text

    def test_get_captures_count_sums_lengths(self) -> None:
        frag = DateFragment()
        frag.captures = {"months": ["January"], "digits": ["5", "2024"], "days": []}
        assert frag.get_captures_count() == 3

    def test_get_captures_count_empty(self) -> None:
        assert DateFragment().get_captures_count() == 0

    def test_get_captures_group_count_counts_non_empty_groups(self) -> None:
        frag = DateFragment()
        frag.captures = {"months": ["January"], "digits": ["5", "2024"], "days": []}
        assert frag.get_captures_group_count() == 2

    def test_get_captures_group_count_empty(self) -> None:
        assert DateFragment().get_captures_group_count() == 0


class TestTokenGroup:
    def test_empty_captures_yield_empty_group(self) -> None:
        assert DateFinder.get_token_group({}) == ""

    def test_all_empty_lists_yield_empty_group(self) -> None:
        assert DateFinder.get_token_group({"digits": [], "months": []}) == ""

    def test_first_non_empty_group_wins(self) -> None:
        assert DateFinder.get_token_group({"digits": ["5"], "months": ["January"]}) == "digits"


class TestFindDates:
    def test_bare_datetime_when_no_flags(self) -> None:
        results = list(_finder().find_dates("January 5, 2024"))
        assert results == [datetime.datetime(2024, 1, 5)]

    def test_source_flag_appends_match_string(self) -> None:
        results = list(_finder().find_dates("January 5, 2024", source=True))
        assert len(results) == 1
        as_dt, source = results[0]
        assert as_dt == datetime.datetime(2024, 1, 5)
        assert source == "January 5, 2024"

    def test_index_flag_appends_indices(self) -> None:
        results = list(_finder().find_dates("January 5, 2024", index=True))
        assert len(results) == 1
        as_dt, indices = results[0]
        assert as_dt == datetime.datetime(2024, 1, 5)
        assert indices == (0, 15)

    def test_source_and_index_flags(self) -> None:
        results = list(_finder().find_dates("January 5, 2024", source=True, index=True))
        assert results == [(datetime.datetime(2024, 1, 5), "January 5, 2024", (0, 15))]

    def test_unparseable_fragment_is_skipped(self) -> None:
        # "99/99/9999" tokenizes into a fragment but no parser accepts it,
        # so find_dates skips it and yields nothing.
        finder = _finder()
        assert len(list(finder.extract_date_strings("99/99/9999"))) == 1
        assert list(finder.find_dates("99/99/9999")) == []
        assert list(finder.find_dates("99/99/9999", source=True, index=True)) == []

    def test_strict_keeps_complete_date(self) -> None:
        results = list(_finder().find_dates("May 16, 2015", strict=True))
        assert results == [datetime.datetime(2015, 5, 16)]

    def test_multiple_dates_all_returned(self) -> None:
        results = list(_finder().find_dates("January 5, 2024 and May 16, 2015"))
        assert datetime.datetime(2024, 1, 5) in results
        assert datetime.datetime(2015, 5, 16) in results


class TestParseDateStringBranches:
    def test_short_garbage_returns_none_via_length_guard(self) -> None:
        # "zz" fails both parsers, reaches the dateutil fallback, and is
        # rejected by the minimum-length guard.
        assert _finder().parse_date_string("zz", dict(EMPTY_CAPTURES)) is None

    def test_fallback_reattaches_timezone(self) -> None:
        # The leading "due " defeats the first-pass parsers; the fallback
        # strips it plus the timezone, parses with dateutil, then re-attaches EST.
        captures = dict(EMPTY_CAPTURES)
        captures["timezones"] = ["EST"]
        result = _finder().parse_date_string("due January 5, 2024 EST", captures)
        assert result is not None
        assert (result.year, result.month, result.day) == (2024, 1, 5)
        assert result.tzinfo is not None

    def test_fallback_without_timezone(self) -> None:
        captures = dict(EMPTY_CAPTURES)
        result = _finder().parse_date_string("due January 5, 2024", captures)
        assert result is not None
        assert (result.year, result.month, result.day) == (2024, 1, 5)

    def test_unknown_locale_double_value_error_returns_none(self) -> None:
        # locales=["xx-YY"] and languages=["xx"] both raise ValueError inside
        # dateparser, exercising the language-fallback except path.
        result = _finder().parse_date_string("January 5, 2024", dict(EMPTY_CAPTURES), locale=Locale("xx-YY"))
        assert result is None

    def test_known_locale_still_parses(self) -> None:
        result = _finder().parse_date_string("January 5, 2024", dict(EMPTY_CAPTURES), locale=Locale("en-GB"))
        assert result == datetime.datetime(2024, 1, 5)

    def test_dateutil_wins_on_parser_mismatch(self) -> None:
        # dateparser misparses the time in "29MAY19 1350", so the dateutil
        # result replaces it.
        result = _finder().parse_date_string("29MAY19 1350", dict(EMPTY_CAPTURES))
        assert result == datetime.datetime(2019, 5, 29, 13, 50)


class TestTimezoneHelpers:
    def test_add_tzinfo_none_returns_none(self) -> None:
        assert _finder()._add_tzinfo(None, "EST") is None

    def test_add_tzinfo_attaches_zone(self) -> None:
        naive = datetime.datetime(2024, 1, 5)
        aware = _finder()._add_tzinfo(naive, "EST")
        assert aware is not None
        assert aware.replace(tzinfo=None) == naive
        assert aware.tzinfo is not None
        assert aware.utcoffset() == datetime.timedelta(hours=-5)

    def test_find_and_replace_strips_timezone_and_extra_tokens(self) -> None:
        captures = dict(EMPTY_CAPTURES)
        captures["timezones"] = ["EST"]
        date_string, tz_string = _finder()._find_and_replace("January 5, 2024 EST due", captures)
        assert tz_string == "EST"
        assert "est" not in date_string
        assert "due" not in date_string
        assert "january 5, 2024" in date_string

    def test_find_and_replace_without_timezones(self) -> None:
        date_string, tz_string = _finder()._find_and_replace("due January 5, 2024", dict(EMPTY_CAPTURES))
        assert tz_string == ""
        assert "due" not in date_string

    def test_pop_tz_string_returns_last_sorted(self) -> None:
        zones = ["EST", "PST"]
        assert _finder()._pop_tz_string(zones) == "PST"
        assert zones == ["EST"]

    def test_pop_tz_string_maps_full_name_to_abbreviation(self) -> None:
        assert _finder()._pop_tz_string(["pacific"]) == "PST"
        assert _finder()._pop_tz_string(["eastern"]) == "EST"

    def test_pop_tz_string_empty_returns_empty(self) -> None:
        assert _finder()._pop_tz_string([]) == ""


class TestModuleFindDates:
    def test_module_entry_point_with_flags(self) -> None:
        results = list(find_dates("January 5, 2024", source=True, index=True, base_date=BASE_DATE))
        assert results == [(datetime.datetime(2024, 1, 5), "January 5, 2024", (0, 15))]

    def test_module_entry_point_bare(self) -> None:
        results = list(find_dates("May 16, 2015", base_date=BASE_DATE))
        assert results == [datetime.datetime(2015, 5, 16)]

    def test_module_entry_point_strict(self) -> None:
        results = list(find_dates("May 16, 2015", strict=True, base_date=BASE_DATE))
        assert results == [datetime.datetime(2015, 5, 16)]
