__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import datetime
from types import SimpleNamespace

from lexnlp.extract.common.date_parsing.datefinder import DateFinder
from lexnlp.extract.en import dates as dates_mod
from lexnlp.extract.en.dates import (
    check_date_parts_are_in_date,
    get_raw_date_list,
)

BASE = datetime.datetime(2019, 1, 1)


def _props(**overrides):
    props = {
        "digits_modifier": [],
        "delimiters": [],
        "extra_tokens": [],
        "months": [],
        "digits": [],
        "time": [],
        "days": [],
        "timezones": [],
        "time_periods": [],
        "hours": [],
        "minutes": [],
        "seconds": [],
        "microseconds": [],
    }
    props.update(overrides)
    return props


def _run_with_candidates(monkeypatch, text, candidates, **kwargs):
    def fake_extract(self, _text, strict=False):
        yield from candidates

    monkeypatch.setattr(DateFinder, "extract_date_strings", fake_extract)
    kwargs.setdefault("base_date", BASE)
    return get_raw_date_list(text, **kwargs)


class TestLocale_epi:
    def test_locale_string_is_converted(self) -> None:
        assert get_raw_date_list("June 1, 2017", locale="en-US", base_date=BASE) == [datetime.date(2017, 6, 1)]


class TestRawDateFilters:
    def test_double_month_rejected(self) -> None:
        assert get_raw_date_list("June July 2017", base_date=BASE) == []

    def test_month_glued_to_word_rejected(self, monkeypatch) -> None:
        candidates = [
            ("Marquee 2017", (0, 12), _props(months=["Mar"], extra_tokens=["quee"], digits=["2017"])),
        ]
        assert _run_with_candidates(monkeypatch, "Marquee 2017", candidates) == []

    def test_digits_modifier_only_rejected(self, monkeypatch) -> None:
        candidates = [("1st", (0, 3), _props(digits_modifier=["1st"]))]
        assert _run_with_candidates(monkeypatch, "1st", candidates) == []

    def test_day_of_week_only_rejected(self, monkeypatch) -> None:
        candidates = [("Monday", (0, 6), _props(days=["Monday"]))]
        assert _run_with_candidates(monkeypatch, "Monday", candidates) == []

    def test_dotted_number_before_word_rejected(self, monkeypatch) -> None:
        candidates = [
            (
                "1962. June 5 Report",
                (0, 19),
                _props(digits=["1962", "5"], months=["June"], delimiters=[" "]),
            ),
        ]
        assert _run_with_candidates(monkeypatch, "1962. June 5 Report", candidates) == []

    def test_to_token_survives_cleanup_while_others_stripped(self, monkeypatch) -> None:
        candidates = [
            (
                "June x 5, 2017",
                (0, 14),
                _props(
                    months=["June"],
                    digits=["5", "2017"],
                    delimiters=["", ",", " "],
                    extra_tokens=["to", "x"],
                ),
            ),
        ]
        assert _run_with_candidates(monkeypatch, "June x 5, 2017", candidates) == [datetime.date(2017, 6, 5)]

    def test_overlong_candidate_rejected(self, monkeypatch) -> None:
        date_string = "June 2017" + " x" * 20
        assert len(date_string) > dates_mod.DATE_MAX_LENGTH
        candidates = [
            (
                date_string,
                (0, len(date_string)),
                _props(months=["June"], digits=["2017"], delimiters=[" "]),
            ),
        ]
        assert _run_with_candidates(monkeypatch, date_string, candidates) == []

    def test_to_token_left_in_place_while_others_stripped(self) -> None:
        assert get_raw_date_list("June 1 to June 5, 2017", base_date=BASE) == [
            datetime.date(2019, 6, 1),
            datetime.date(2017, 6, 5),
        ]

    def test_day_of_merge_with_previous_ordinal(self, monkeypatch) -> None:
        candidates = [
            ("1st", (0, 3), _props(digits_modifier=["1st"])),
            (
                "of June 2017",
                (4, 16),
                _props(
                    digits=["2017"],
                    months=["June"],
                    delimiters=["", ""],
                    extra_tokens=["of"],
                ),
            ),
        ]
        assert _run_with_candidates(monkeypatch, "x" * 20, candidates) == [datetime.date(2017, 6, 1)]

    def test_parse_failure_yields_no_date(self, monkeypatch) -> None:
        def boom(self, date_string, captures, locale=None):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(DateFinder, "parse_date_string", boom)
        assert get_raw_date_list("June 5, 2017", base_date=BASE) == []

    def test_date_parts_mismatch_yields_no_date(self, monkeypatch) -> None:
        candidates = [
            (
                "June 5, 2017",
                (0, 12),
                _props(months=["July"], digits=["5", "2017"], delimiters=["", ","]),
            ),
        ]
        assert _run_with_candidates(monkeypatch, "June 5, 2017", candidates) == []

    def test_broken_tzinfo_date_rejected(self, monkeypatch) -> None:
        from dateutil.tz import tzoffset

        broken = datetime.datetime(2001, 1, 22, 20, 1, tzinfo=tzoffset(None, -104400))

        def fake_parse(self, date_string, captures, locale=None):
            return broken

        monkeypatch.setattr(DateFinder, "parse_date_string", fake_parse)
        candidates = [
            (
                "Year 2001 month 01",
                (0, 19),
                _props(digits=["2001", "1"], delimiters=["-"]),
            ),
        ]
        assert _run_with_candidates(monkeypatch, "Year 2001 month 01", candidates) == []


class TestCheckDateParts:
    def test_month_mismatch_returns_false(self) -> None:
        assert (
            check_date_parts_are_in_date(
                datetime.datetime(2017, 5, 1),
                _props(months=["june"]),
            )
            is False
        )

    def test_unmatched_extra_digit_returns_false(self) -> None:
        assert (
            check_date_parts_are_in_date(
                datetime.datetime(2017, 6, 15),
                _props(digits=["2017", "6", "99"]),
            )
            is False
        )

    def test_matching_parts_return_true(self) -> None:
        assert (
            check_date_parts_are_in_date(
                datetime.datetime(2017, 6, 15),
                _props(digits=["2017", "6", "15"], months=["june"]),
            )
            is True
        )


class TestTrainDefaultModel:
    def test_save_false_builds_then_removes_probe_file(self, monkeypatch, tmp_path) -> None:
        calls: dict[str, object] = {}

        def fake_build(examples, output_file, parse_dates, characters):
            calls["n_examples"] = len(examples)
            calls["output_file"] = output_file
            calls["characters"] = characters
            assert len(examples) > 0
            assert callable(parse_dates)

        monkeypatch.setattr(dates_mod, "build_date_model", fake_build)
        monkeypatch.setattr(dates_mod, "random", SimpleNamespace(random=lambda: 0.99, randint=lambda a, b: 2))
        monkeypatch.chdir(tmp_path)
        (tmp_path / "test_date_model.pickle").write_bytes(b"stale")

        dates_mod.train_default_model(save=False)

        assert calls["output_file"] == "test_date_model.pickle"
        assert not (tmp_path / "test_date_model.pickle").exists()

    def test_save_true_uses_module_path_and_random_examples(self, monkeypatch) -> None:
        calls: dict[str, object] = {}

        def fake_build(examples, output_file, parse_dates, characters):
            calls["n_examples"] = len(examples)
            calls["output_file"] = output_file
            sample = [ex for ex in examples if ex[0].startswith("on ")]
            assert len(sample) > 0
            assert callable(parse_dates)

        monkeypatch.setattr(dates_mod, "build_date_model", fake_build)
        monkeypatch.setattr(dates_mod, "random", SimpleNamespace(random=lambda: 0.0, randint=lambda a, b: 2))

        dates_mod.train_default_model(save=True)

        assert str(calls["output_file"]).endswith("date_model.pickle")
        assert dates_mod.MODULE_PATH in str(calls["output_file"])
        assert calls["n_examples"] > 100
