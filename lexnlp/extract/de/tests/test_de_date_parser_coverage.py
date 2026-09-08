"""Coverage tests for lexnlp.extract.de.de_date_parser missing lines."""

import datetime

import numpy as np
import pytest
import zahlwort2num as w2n

from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.de.date_model import DATE_MODEL_CHARS, DE_ALPHA_CHAR_SET
from lexnlp.extract.de.de_date_parser import DatePart, DeDateParser


def _parser(**kwargs) -> DeDateParser:
    settings = {
        "PREFER_DAY_OF_MONTH": "first",
        "STRICT_PARSING": False,
        "DATE_ORDER": "DMY",
    }
    kwargs.setdefault("locale", Locale("de-DE"))
    kwargs.setdefault("enable_classifier_check", False)
    kwargs.setdefault("dateparser_settings", settings)
    kwargs.setdefault("alphabet_character_set", DE_ALPHA_CHAR_SET)
    return DeDateParser(DATE_MODEL_CHARS, **kwargs)


class TestDatePartStr:
    def test_str_shows_text_and_category(self) -> None:
        assert str(DatePart("Juli", "month")) == "Juli [month]"

    def test_repr_matches_str(self) -> None:
        part = DatePart("2011", "number", 2011)
        assert repr(part) == "2011 [number]"
        assert repr(part) == str(part)


class TestGetWordParts:
    def test_empty_token_skipped(self) -> None:
        parts = _parser().get_word_parts(" Oktober 2011")
        assert [(p.text, p.category, p.num_value) for p in parts] == [
            ("Oktober", "month", 0),
            ("2011", "number", 2011),
        ]

    def test_negative_numeral_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_convert = w2n.convert

        def _convert(word: str):
            if word == "minuswort":
                return -4
            return real_convert(word)

        monkeypatch.setattr(w2n, "convert", _convert)
        parts = _parser().get_word_parts("Oktober minuswort 2011")
        assert [(p.text, p.category) for p in parts] == [("Oktober", "month"), ("2011", "number")]

    def test_non_alpha_token_skipped(self) -> None:
        parts = _parser().get_word_parts("§5 Mai 2020")
        assert [(p.text, p.category, p.num_value) for p in parts] == [
            ("Mai", "month", 0),
            ("2020", "number", 2020),
        ]


class TestGetDateAnnotations:
    def test_empty_segment_after_und_split_skipped(self) -> None:
        ants = list(_parser().get_date_annotations("5. Oktober 2011 und ", strict=False))
        assert len(ants) == 1
        assert ants[0].date == datetime.datetime(2011, 10, 5)
        assert ants[0].coords == (0, 15)

    def test_missing_language_raises(self) -> None:
        parser = DeDateParser(DATE_MODEL_CHARS, locale=Locale(""))
        with pytest.raises(RuntimeError, match=r"Define text and language\."):
            list(parser.get_date_annotations("5. Oktober 2011"))

    def test_dateparser_error_printed_and_skipped(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        parser = _parser()

        def _boom(text_part: str, strict: bool):
            raise RuntimeError("boom")

        monkeypatch.setattr(parser, "get_dateparser_dates", _boom)
        assert list(parser.get_date_annotations("5. Oktober 2011", strict=False)) == []
        assert "boom" in capsys.readouterr().out

    def test_overlapping_candidate_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _parser()
        monkeypatch.setattr(
            parser,
            "get_dateparser_dates",
            lambda text_part, strict: [
                ("Oktober 2011", datetime.datetime(2011, 10, 1)),
                ("5. Oktober 2011", datetime.datetime(2011, 10, 5)),
            ],
        )
        ants = list(parser.get_date_annotations("Am 5. Oktober 2011.", strict=False))
        assert len(ants) == 1
        assert ants[0].date == datetime.datetime(2011, 10, 5)
        assert ants[0].text == "5. Oktober 2011"

    def test_classifier_reject_yields_nothing(self) -> None:
        parser = _parser(enable_classifier_check=True, classifier_model=_StubClassifier(0.01))
        assert list(parser.get_date_annotations("5. Oktober 2011", strict=False)) == []

    def test_classifier_accept_yields_annotation(self) -> None:
        parser = _parser(enable_classifier_check=True, classifier_model=_StubClassifier(0.99))
        ants = list(parser.get_date_annotations("5. Oktober 2011", strict=False))
        assert len(ants) == 1
        assert ants[0].date == datetime.datetime(2011, 10, 5)


class _StubClassifier:
    """Minimal stand-in for the bundled sklearn date model."""

    def __init__(self, date_probability: float) -> None:
        self.columns: list[str] = []
        self._date_probability = date_probability

    def predict_proba(self, rows: list[list[float]]) -> np.ndarray:
        return np.array([[1.0 - self._date_probability, self._date_probability] for _ in rows])
