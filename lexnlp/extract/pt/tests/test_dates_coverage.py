"""Coverage tests for lexnlp.extract.pt.dates missing lines."""

import datetime

from lexnlp.extract.pt.dates import PtDateParser


def _parser() -> PtDateParser:
    return PtDateParser()


class TestPassedGeneralCheck:
    def test_empty_token_rejected(self) -> None:
        assert _parser().passed_general_check("", datetime.datetime(2020, 1, 1)) is False

    def test_short_token_rejected(self) -> None:
        assert _parser().passed_general_check("ab", datetime.datetime(2020, 1, 1)) is False

    def test_weekday_without_year_rejected(self) -> None:
        assert _parser().passed_general_check("terça feira", datetime.datetime(2020, 1, 1)) is False

    def test_weekday_with_year_accepted(self) -> None:
        assert _parser().passed_general_check("terça feira 2020", datetime.datetime(2020, 1, 1)) is True


class TestCoerceYear:
    def test_empty_returns_none(self) -> None:
        assert PtDateParser._coerce_year("") is None

    def test_non_numeric_returns_none(self) -> None:
        assert PtDateParser._coerce_year("abcd") is None

    def test_two_digit_year_maps_to_2000s(self) -> None:
        assert PtDateParser._coerce_year("05") == 2005
        assert PtDateParser._coerce_year("20") == 2020
        assert PtDateParser._coerce_year("49") == 2049

    def test_two_digit_year_maps_to_1900s(self) -> None:
        assert PtDateParser._coerce_year("50") == 1950
        assert PtDateParser._coerce_year("95") == 1995

    def test_four_digit_year_unchanged(self) -> None:
        assert PtDateParser._coerce_year("2020") == 2020


class TestBuildDate:
    def test_day_zero_returns_none(self) -> None:
        assert _parser()._build_date(0, 5, 2020) is None

    def test_month_out_of_range_returns_none(self) -> None:
        assert _parser()._build_date(15, 13, 2020) is None

    def test_impossible_calendar_date_returns_none(self) -> None:
        assert _parser()._build_date(31, 4, 2020) is None
        assert _parser()._build_date(30, 2, 2021) is None

    def test_valid_date_built(self) -> None:
        assert _parser()._build_date(15, 2, 2020) == datetime.datetime(2020, 2, 15)


class TestExtraDatesEndToEnd:
    def test_locality_date_extracted(self) -> None:
        ants = list(_parser().get_date_annotations("Brasília, 12 de março de 2024", strict=False))
        assert len(ants) == 1
        assert ants[0].date == datetime.datetime(2024, 3, 12)
        assert ants[0].coords == (10, 29)
        assert ants[0].text == "12 de março de 2024"

    def test_two_digit_numeric_year_resolves_to_2000s(self) -> None:
        ants = list(_parser().get_date_annotations("reunião em 15/02/20 com todos", strict=False))
        assert len(ants) == 1
        assert ants[0].date == datetime.datetime(2020, 2, 15)


class TestLocalityDateDefensiveSkips:
    """Lines 316-317 (int() failure) and 319 (unknown month) in get_extra_dates.

    The real LOCALITY_DATE_RE only matches numeric day/year groups and month
    aliases from _MONTH_NUMBER, so both guards are defensive: reaching them
    requires patching the regex or the month table.
    """

    def _spy_on_build_date(self, monkeypatch) -> list:
        calls: list = []
        origin = PtDateParser._build_date

        def spy(self, day, month, year):
            calls.append((day, month, year))
            return origin(self, day, month, year)

        monkeypatch.setattr(PtDateParser, "_build_date", spy)
        return calls

    def test_locality_date_builds_expected_datetime(self, monkeypatch) -> None:
        parser = PtDateParser("Brasília, 12 de março de 2024")
        parser.dates = []
        calls = self._spy_on_build_date(monkeypatch)
        parser.get_extra_dates(strict=False)
        assert calls == [(12, 3, 2024)]
        assert parser.dates == [("12 de março de 2024", datetime.datetime(2024, 3, 12))]

    def test_locality_unknown_month_skips_build(self, monkeypatch) -> None:
        import lexnlp.extract.pt.dates as pt_dates

        monkeypatch.setattr(pt_dates, "_MONTH_NUMBER", {})
        parser = PtDateParser("Brasília, 12 de março de 2024")
        parser.dates = []
        calls = self._spy_on_build_date(monkeypatch)
        parser.get_extra_dates(strict=False)
        # Line 319 `continue`: the locality branch bailed out before _build_date.
        assert calls == []
        # The sequential-year branch still yields the date via dateparser.
        assert parser.dates == [("12 de março de 2024", datetime.datetime(2024, 3, 12))]

    def test_locality_unparsable_ints_skipped(self, monkeypatch) -> None:
        import re as stdlib_re

        fake_re = stdlib_re.compile(r"XXX(?P<day>[A-Z]+) de (?P<month>\S+) de (?P<year>[A-Z]+)")
        monkeypatch.setattr(PtDateParser, "LOCALITY_DATE_RE", fake_re)
        parser = PtDateParser("XXXAB de março de EF")
        parser.dates = []
        # Non-vacuous: the stub regex really matches once, so the only way to
        # reach zero _build_date calls is via the except/continue.
        assert len(list(fake_re.finditer(parser.text))) == 1
        calls = self._spy_on_build_date(monkeypatch)
        # int("AB") raises ValueError -> lines 316-317 `continue`.
        parser.get_extra_dates(strict=False)
        assert calls == []
        assert parser.dates == []
