"""Coverage tests for German date-classifier sample generation and training glue."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from num2words import num2words

from lexnlp.extract.de.date_model import MONTH_NAMES
from lexnlp.extract.de.dates_de_classifier import (
    MODULE_PATH,
    TODAY,
    WRITTEN_DATE_NUMS,
    add_numeric_date_samples,
    get_written_date_num,
    make_date_samples,
    setup_date_parser,
    train_default_model,
)
from lexnlp.extract.de.de_date_parser import DeDateParser


class TestSetupDateParser:
    def test_parser_locale_and_classifier_flag(self) -> None:
        enabled = setup_date_parser(True)
        disabled = setup_date_parser(False)
        assert isinstance(enabled, DeDateParser)
        assert isinstance(disabled, DeDateParser)
        assert enabled.enable_classifier_check is True
        assert disabled.enable_classifier_check is False
        assert enabled.locale.get_locale() == "de-DE"
        assert enabled.dateparser_settings["DATE_ORDER"] == "DMY"
        assert enabled.count_words is True
        assert enabled.classifier_model is disabled.classifier_model


class TestMakeDateSamples:
    def test_fixed_examples_include_written_and_numeric_forms(self) -> None:
        examples = make_date_samples()
        by_text = {text: dates for text, dates in examples}
        assert by_text["Spätestens am 01.06.2017"] == [datetime.date(2017, 6, 1)]
        assert by_text["Wird bis Juni 2017 abgeschlossen sein"] == [datetime.date(2017, 6, 1)]
        assert by_text["Abschnitt über 6.25"] == []
        assert by_text["am siebzehnten Oktober eintausendneunhundertdreiundachtzig"] == [
            datetime.date(1983, 10, 17)
        ]
        assert by_text["Anfangsdatum: 11/11/1993"] == [datetime.date(1993, 11, 11)]
        assert by_text["Anfangsdatum: 27/02/2023"] == [datetime.date(2023, 2, 27)]
        assert examples[0][0] == "Spätestens am 01.06.2017"
        assert examples[-1][0] == "Anfangsdatum: 11/11/1993"


class TestGetWrittenDateNum:
    def test_table_covers_one_through_nineteen(self) -> None:
        assert get_written_date_num(1) == "ersten"
        assert get_written_date_num(19) == "neunzehnten"
        assert [get_written_date_num(day) for day in range(1, 20)] == WRITTEN_DATE_NUMS

    def test_twenty_and_above_use_num2words_sten_suffix(self) -> None:
        assert get_written_date_num(20) == num2words(20, lang="de") + "sten"
        assert get_written_date_num(21) == num2words(21, lang="de") + "sten"
        assert get_written_date_num(31) == num2words(31, lang="de") + "sten"


class TestAddNumericDateSamples:
    def test_numeric_and_written_samples_include_invalid_day_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("lexnlp.extract.de.dates_de_classifier.random.random", lambda: 0.0)
        monkeypatch.setattr("lexnlp.extract.de.dates_de_classifier.random.randint", lambda _a, _b: 3)
        examples: list[tuple[str, list[datetime.date]]] = []
        add_numeric_date_samples(examples)

        texts = {text for text, _dates in examples}
        first = datetime.date(1970, 1, 1)
        later = first + datetime.timedelta(days=3)
        assert f"{first.year}/{first.month}/{first.day}" in texts
        assert f"{first.year}.{first.month}.{first.day}" in texts
        assert f"bis {first.year}-{first.month}-{first.day}" in texts
        assert ("bis " + first.strftime("%b %d, %Y"), [first]) in examples
        assert ("am " + first.strftime("%B %d, %Y"), [first]) in examples
        assert (f"bis {first} zum {later}", [first, later]) in examples
        assert (f"{first.isoformat()} bis {later.isoformat()}", [first, later]) in examples
        # Feb 30 never becomes a datetime.date; the ValueError path skips it.
        assert "1970/2/30" not in texts
        assert "1970.2.31" not in texts

        written_without_year = f"der {get_written_date_num(1)} {MONTH_NAMES[0]}"
        assert (written_without_year, [datetime.date(TODAY.year, 1, 1)]) in examples
        written_with_year = f"am {get_written_date_num(2)} {MONTH_NAMES[0]} 2010"
        assert (written_with_year, [datetime.date(2010, 1, 2)]) in examples
        dotted = f"{3}. {MONTH_NAMES[5]} 1981"
        assert (dotted, [datetime.date(1981, 6, 3)]) in examples
        assert any(row[0].startswith("am 30. ") or "dreißigsten" in row[0] or "30." in row[0] for row in examples)


class TestTrainDefaultModel:
    def test_save_false_builds_examples_and_unlinks_temp_pickle(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("lexnlp.extract.de.dates_de_classifier.random.random", lambda: 1.0)
        captured: dict = {}

        def fake_build(examples, output_path, parse_dates, **kwargs) -> None:
            Path(output_path).write_bytes(b"model")
            captured["examples"] = examples
            captured["output_path"] = output_path
            captured["verbose"] = kwargs.get("verbose")
            captured["characters"] = kwargs.get("characters")
            captured["count_words"] = kwargs.get("count_words")
            captured["parsed"] = parse_dates("Spätestens am 01.06.2017")

        with patch("lexnlp.extract.de.dates_de_classifier.build_date_model", side_effect=fake_build):
            train_default_model(save=False, verbose=True, check_date_strings=True)

        assert captured["output_path"] == "test_date_model.pickle"
        assert captured["verbose"] is True
        assert captured["count_words"] is True
        assert not Path("test_date_model.pickle").exists()
        texts = [text for text, _dates in captured["examples"]]
        assert "Spätestens am 01.06.2017" in texts
        assert "mit 1" in texts
        assert "mit 24" in texts
        assert "Leasing mit 1.500€ Anzahlung" in texts
        assert "2 Jahren" in texts
        assert "18 Jahren" in texts
        parsed = captured["parsed"]
        assert parsed
        values = [item[0] for item in parsed]
        assert datetime.date(2017, 6, 1) in values

    def test_save_true_writes_under_module_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("lexnlp.extract.de.dates_de_classifier.MODULE_PATH", str(tmp_path))
        monkeypatch.setattr("lexnlp.extract.de.dates_de_classifier.random.random", lambda: 1.0)
        captured: dict = {}

        def fake_build(examples, output_path, parse_dates, **kwargs) -> None:
            Path(output_path).write_bytes(b"model")
            captured["output_path"] = output_path
            captured["example_count"] = len(examples)

        with patch("lexnlp.extract.de.dates_de_classifier.build_date_model", side_effect=fake_build):
            train_default_model(save=True, verbose=False, check_date_strings=False)

        expected = tmp_path / "date_model.pickle"
        assert captured["output_path"] == str(expected)
        assert expected.is_file()
        assert captured["example_count"] > 20
        assert Path(MODULE_PATH).name == "de"
