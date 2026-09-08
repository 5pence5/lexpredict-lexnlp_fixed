"""Coverage tests for BaseTokenSequenceClassifierModel.

Targets the base-class helpers in
``lexnlp/extract/ml/classifier/base_token_sequence_classifier_model.py``:
get_classifier, pickle save/load round-trips, the abstract-method guards,
train_model, and every run_model strict/non-strict branch.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy
import pytest

from lexnlp.extract.ml.classifier.base_token_sequence_classifier_model import BaseTokenSequenceClassifierModel
from lexnlp.extract.ml.classifier.spacy_token_sequence_model import SpacyTokenSequenceClassifierModel
from lexnlp.extract.ml.classifier.token_sequence_model import TokenSequenceClassifierModel


def _small_model(**kwargs) -> TokenSequenceClassifierModel:
    return TokenSequenceClassifierModel(
        letter_set=["a", "b"],
        digit_set=["1"],
        punc_set=["."],
        symbol_set=["$"],
        match_tokens=["Hi"],
        **kwargs,
    )


class _FakeEstimator:
    """Minimal estimator stub exposing fit/predict."""

    def __init__(self, classes: list[int]) -> None:
        self._classes = list(classes)
        self.fit_X = None
        self.fit_y = None

    def fit(self, feature_data, target_data):
        self.fit_X = feature_data
        self.fit_y = target_data
        return self

    def predict(self, feature_data):
        return numpy.array(self._classes)


def _model_with_predictions(classes: list[int], tokens: list[tuple[int, int]]) -> TokenSequenceClassifierModel:
    model = _small_model()
    model.get_feature_data = lambda text, feature_mask=None: (None, tokens)  # type: ignore[method-assign]
    model.model = _FakeEstimator(classes)
    return model


class TestGetClassifier:
    def test_non_spacy_returns_token_sequence_model(self) -> None:
        model = BaseTokenSequenceClassifierModel.get_classifier(use_spacy=False, letter_set=["a"])
        assert isinstance(model, TokenSequenceClassifierModel)
        assert model.letter_set == ["a"]

    def test_spacy_returns_spacy_model(self) -> None:
        model = BaseTokenSequenceClassifierModel.get_classifier(use_spacy=True, letter_set=["a"])
        assert isinstance(model, SpacyTokenSequenceClassifierModel)
        assert model.letter_set == ["a"]

    def test_kwargs_flow_through(self) -> None:
        model = BaseTokenSequenceClassifierModel.get_classifier(
            use_spacy=False, pre_window=1, post_window=2, string_checks=True
        )
        assert isinstance(model, TokenSequenceClassifierModel)
        assert model.pre_window == 1
        assert model.post_window == 2
        assert model.string_checks is True


class TestSaveLoad:
    def test_pickle_round_trip(self, tmp_path: Path) -> None:
        model = _small_model(pre_window=1)
        dest = tmp_path / "model.pickle"
        model.save_in_file(str(dest))
        assert dest.exists()
        loaded = BaseTokenSequenceClassifierModel.load_from_file(str(dest))
        assert isinstance(loaded, TokenSequenceClassifierModel)
        assert loaded.feature_list == model.feature_list
        assert loaded.letter_set == ["a", "b"]

    def test_compressed_round_trip(self, tmp_path: Path) -> None:
        model = _small_model()
        dest = tmp_path / "model.pickle.gz"
        model.save_in_file_compressed(str(dest))
        assert dest.exists()
        assert dest.stat().st_size > 0
        loaded = BaseTokenSequenceClassifierModel.load_from_file_compressed(str(dest))
        assert isinstance(loaded, TokenSequenceClassifierModel)
        assert loaded.feature_list == model.feature_list
        assert loaded.symbol_set == ["$"]

    def test_load_from_stream(self) -> None:
        model = _small_model()
        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf.seek(0)
        loaded = BaseTokenSequenceClassifierModel.load_from_stream(buf)
        assert isinstance(loaded, TokenSequenceClassifierModel)
        assert loaded.feature_list == model.feature_list

    def test_skops_suffix_dispatches_to_skops_loader(self, tmp_path: Path) -> None:
        from lexnlp.ml.model_io import dump_model

        model = _small_model()
        skops_path = dump_model(model, tmp_path / "model.skops")
        loaded = BaseTokenSequenceClassifierModel.load_from_file(str(skops_path))
        assert isinstance(loaded, TokenSequenceClassifierModel)
        assert loaded.feature_list == model.feature_list


class TestAbstractGuards:
    def test_get_feature_list_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="get_feature_list"):
            BaseTokenSequenceClassifierModel.get_feature_list(None)

    def test_get_feature_data_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="get_feature_data"):
            BaseTokenSequenceClassifierModel.get_feature_data(None, "some text")


class TestTrainModel:
    def test_fit_result_stored(self) -> None:
        model = _small_model()
        estimator = _FakeEstimator([0, 1])
        feature_data = numpy.zeros((2, 3))
        target_data = [0, 1]
        model.train_model(estimator, feature_data, target_data)
        assert model.model is estimator
        assert estimator.fit_X is feature_data
        assert estimator.fit_y == [0, 1]


_TOKENS = [(0, 2), (3, 5), (6, 8)]


class TestRunModel:
    def test_strict_start_then_end_yields_span(self) -> None:
        model = _model_with_predictions([1, 3, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == [(0, 5)]

    def test_strict_end_without_start_yields_nothing(self) -> None:
        model = _model_with_predictions([3, 0, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == []

    def test_strict_start_then_outer_resets_without_yield(self) -> None:
        model = _model_with_predictions([1, 0, 3], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == []

    def test_non_strict_start_then_end_yields_span(self) -> None:
        model = _model_with_predictions([1, 3, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=False)) == [(0, 5)]

    def test_non_strict_end_without_start_yields_nothing(self) -> None:
        model = _model_with_predictions([3, 0, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=False)) == []

    def test_inner_starts_span_for_strict_end(self) -> None:
        model = _model_with_predictions([2, 3, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == [(0, 5)]

    def test_inner_when_span_open_is_ignored(self) -> None:
        model = _model_with_predictions([1, 2, 3], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == [(0, 8)]

    def test_repeated_start_keeps_first(self) -> None:
        model = _model_with_predictions([1, 1, 3], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=True)) == [(0, 8)]

    def test_non_strict_outer_closes_open_span(self) -> None:
        model = _model_with_predictions([1, 0, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=False)) == [(0, 5)]

    def test_outer_with_no_open_span_yields_nothing(self) -> None:
        model = _model_with_predictions([0, 0, 0], _TOKENS)
        assert list(model.run_model("ab cd ef", strict=False)) == []
        model2 = _model_with_predictions([0, 0, 0], _TOKENS)
        assert list(model2.run_model("ab cd ef", strict=True)) == []

    def test_feature_mask_passed_through(self) -> None:
        seen: dict[str, object] = {}

        def fake_get_feature_data(text, feature_mask=None):
            seen["mask"] = feature_mask
            return None, _TOKENS

        model = _small_model()
        model.get_feature_data = fake_get_feature_data  # type: ignore[method-assign]
        model.model = _FakeEstimator([0, 0, 0])
        assert list(model.run_model("ab cd ef", feature_mask=[1, 2, 3])) == []
        assert seen["mask"] == [1, 2, 3]
