__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import datetime
from functools import partial

import joblib
import pytest

from lexnlp.extract.common.dates_classifier_model import build_date_model, get_date_features

CHARACTERS = list("abcdefghij0123456789!@#$%^&*()")
ALPHABET = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

POS_DATE = datetime.date(2020, 5, 5)
NEG_DATE = datetime.date(1999, 1, 1)


def _parse_dates(date_str: str):
    if date_str.startswith("MATCH"):
        return [(POS_DATE, (0, 8))]
    return [(POS_DATE, (0, 8)), (NEG_DATE, (10, 18))]


def _examples(n: int = 12):
    text = "0123456789ABCDEFGHIJ0123456789"
    matched = [(f"MATCH{text}", [POS_DATE]) for _ in range(n)]
    mismatched = [(text, [POS_DATE]) for _ in range(n)]
    return matched + mismatched


class TestBuildDateModel:
    def test_trains_selects_and_dumps_best_model(self, monkeypatch, tmp_path) -> None:
        import sklearn.ensemble

        from lexnlp.extract.common import dates_classifier_model as model_mod

        real_forest = sklearn.ensemble.RandomForestClassifier
        monkeypatch.setattr(
            sklearn.ensemble, "RandomForestClassifier", partial(real_forest, n_estimators=10)
        )
        monkeypatch.setattr(model_mod.sklearn.ensemble, "RandomForestClassifier", partial(real_forest, n_estimators=10))

        output_file = tmp_path / "date_model.joblib"
        examples = _examples()
        # One mismatched example exercises the verbose diff-reporting branch.
        examples = [(examples[0][0], [datetime.date(2001, 1, 1)])] + examples

        build_date_model(
            examples,
            str(output_file),
            _parse_dates,
            characters=CHARACTERS,
            verbose=True,
            alphabet_char_set=ALPHABET,
            count_words=True,
        )

        assert output_file.exists()
        model = joblib.load(output_file)
        assert model is not None
        assert hasattr(model, "columns")
        assert len(model.columns) > 0

        rows = [
            get_date_features(
                text,
                0,
                8,
                characters=CHARACTERS,
                alphabet_char_set=ALPHABET,
                count_words=True,
            )
            for text, _ in _examples(4)
        ]
        feature_list = [[row[col] for col in model.columns] for row in rows]
        predictions = model.predict(feature_list)
        assert len(predictions) == 8
        assert set(predictions) <= {0, 1}

    def test_unhashable_parse_result_reraises(self, tmp_path) -> None:
        def bad_parse(date_str: str):
            return [({"not": "hashable"}, (0, 5))]

        with pytest.raises(TypeError):
            build_date_model(
                [("sometext", [POS_DATE])],
                str(tmp_path / "model.joblib"),
                bad_parse,
                characters=["a", "b"],
            )


class TestCountWordsEmptyToken:
    def test_empty_split_token_is_skipped(self) -> None:
        # A leading separator makes REG_WORD_SEPARATOR.split produce a "" token.
        features = get_date_features(
            ",,Al 12",
            0,
            7,
            characters=["A", "l", ",", "1", "2", " "],
            alphabet_char_set=ALPHABET,
            include_bigrams=False,
            norm=False,
            count_words=True,
        )
        assert features["nb31"] == 1
        assert features["na31"] == 0
        assert features["wr_l"] == 0
        assert features["wr_u"] == 1
