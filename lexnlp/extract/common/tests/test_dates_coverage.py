"""Coverage tests for lexnlp.extract.common.dates."""

import datetime
from unittest.mock import patch

import numpy
import pytest

from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.common.annotations.date_annotation import DateAnnotation
from lexnlp.extract.common.dates import DateParser, LocaleInfoImport


class TestLocaleInfoImportCoverage:
    def test_locale_specific_date_order(self) -> None:
        assert LocaleInfoImport(Locale("en-001")).date_order == "DMY"

    def test_default_date_order(self) -> None:
        assert LocaleInfoImport(Locale("en-US")).date_order == "MDY"

    def test_unknown_language_falls_back_to_mdy(self) -> None:
        assert LocaleInfoImport(Locale("xx-YY")).date_order == "MDY"


class TestGetDateAnnotationsCoverage:
    def test_missing_text_raises(self) -> None:
        parser = DateParser(characters=[], text=None, locale=Locale("en"), enable_classifier_check=False)
        with pytest.raises(RuntimeError, match="Define text and language"):
            list(parser.get_date_annotations())

    def test_missing_language_raises(self) -> None:
        parser = DateParser(characters=[], text="hello", locale=Locale(""), enable_classifier_check=False)
        with pytest.raises(RuntimeError, match="Define text and language"):
            list(parser.get_date_annotations())

    def test_search_error_yields_nothing_and_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = DateParser(characters=[], text=None, locale=Locale("en"), enable_classifier_check=False)
        with patch.object(parser, "get_dateparser_dates", side_effect=ValueError("boom")):
            assert list(parser.get_date_annotations("January 5, 2024")) == []
        assert "boom" in capsys.readouterr().out


class _RejectingModel:
    """Minimal classifier stub whose date-class probability is below threshold."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def predict_proba(self, rows: list[list[float]]) -> numpy.ndarray:
        assert len(rows) == 1
        assert len(rows[0]) == len(self.columns)
        return numpy.array([[0.99, 0.01]])


class TestClassifierCheckCoverage:
    TEXT = "The meeting is on January 5, 2024."

    def _parser(self, enable_classifier_check: bool) -> DateParser:
        characters = ["a", "b"]
        columns = ["char_a", "char_b", "bigram_ab", "bigram_ba"]
        return DateParser(
            characters=characters,
            text=None,
            locale=Locale("en"),
            enable_classifier_check=enable_classifier_check,
            classifier_model=_RejectingModel(columns),
            classifier_threshold=0.5,
        )

    def test_rejected_date_is_skipped(self) -> None:
        assert self._parser(True).get_date_annotation_list(self.TEXT, Locale("en-US")) == []

    def test_without_classifier_check_date_is_found(self) -> None:
        ants = self._parser(False).get_date_annotation_list(self.TEXT, Locale("en-US"))
        assert len(ants) == 1
        assert ants[0].text == "on January 5, 2024"


class TestGetDateAnnotationListCoverage:
    def test_returns_date_annotations(self) -> None:
        parser = DateParser(characters=[], text=None, locale=Locale("en"), enable_classifier_check=False)
        ants = parser.get_date_annotation_list("The meeting is on January 5, 2024.", Locale("en-US"))
        assert len(ants) == 1
        ant = ants[0]
        assert isinstance(ant, DateAnnotation)
        assert ant.coords == (15, 33)
        assert ant.text == "on January 5, 2024"
        assert ant.date == datetime.datetime(2024, 1, 5)
        assert ant.locale == "en"
