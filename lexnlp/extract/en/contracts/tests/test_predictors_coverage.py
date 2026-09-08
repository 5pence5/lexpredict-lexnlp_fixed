"""Coverage tests for lexnlp/extract/en/contracts/predictors.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pandas import Series

from lexnlp.extract.en.contracts.predictors import (
    ProbabilityPredictorContractType,
    ProbabilityPredictorIsContract,
)
from lexnlp.extract.en.contracts import runtime_model
from lexnlp.ml import catalog as ml_catalog
from lexnlp.ml.predictor import ProbabilityPredictor


def _is_contract_instance(proba_positive: float) -> ProbabilityPredictorIsContract:
    obj = ProbabilityPredictorIsContract.__new__(ProbabilityPredictorIsContract)

    class _FakePipeline:
        classes_ = np.array([0, 1])

        def predict_proba(self, X=()):  # noqa: N803
            return np.array([[1.0 - proba_positive, proba_positive]])

    obj.pipeline = _FakePipeline()  # type: ignore[assignment]
    return obj


def _contract_type_instance(
    probas: list[float], classes: list[str]
) -> ProbabilityPredictorContractType:
    obj = ProbabilityPredictorContractType.__new__(ProbabilityPredictorContractType)

    class _FakePipeline:
        classes_ = np.array(classes)

        def predict_proba(self, X=()):  # noqa: N803
            return np.array([probas])

    obj.pipeline = _FakePipeline()  # type: ignore[assignment]
    return obj


class TestIsContractFallbackFailure:
    def test_legacy_fallback_failure_raises_runtime_error(self, monkeypatch, tmp_path):
        def raise_default_load_error(cls):
            raise FileNotFoundError("missing default model")

        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setattr(
            ProbabilityPredictor,
            "get_default_pipeline",
            classmethod(raise_default_load_error),
        )
        legacy_path = tmp_path / "legacy.cloudpickle"
        legacy_path.write_bytes(b"junk")
        monkeypatch.setattr(
            ml_catalog, "get_path_from_catalog", lambda tag: legacy_path
        )

        import cloudpickle

        def fake_load(_file_obj):
            raise ValueError("cannot unpickle legacy model")

        monkeypatch.setattr(cloudpickle, "load", fake_load)
        with pytest.raises(RuntimeError, match="legacy fallback model"):
            ProbabilityPredictorIsContract.get_default_pipeline()


class TestIsContractSanityCheck:
    def test_three_classes_raises_value_error(self):
        obj = ProbabilityPredictorIsContract.__new__(
            ProbabilityPredictorIsContract
        )
        obj.pipeline = SimpleNamespace(classes_=[0, 1, 2])  # type: ignore[assignment]
        with pytest.raises(ValueError):
            obj._sanity_check()

    def test_two_classes_passes(self):
        obj = ProbabilityPredictorIsContract.__new__(
            ProbabilityPredictorIsContract
        )
        obj.pipeline = SimpleNamespace(classes_=[0, 1])  # type: ignore[assignment]
        assert obj._sanity_check() is None


class TestIsContractReturnProbability:
    def test_positive_with_probability(self):
        predictor = _is_contract_instance(0.7)
        result = predictor.is_contract("some text", return_probability=True)
        assert isinstance(result, tuple)
        classification, probability = result
        assert classification == True  # noqa: E712
        assert probability == pytest.approx(0.7)

    def test_negative_with_probability(self):
        predictor = _is_contract_instance(0.2)
        classification, probability = predictor.is_contract(
            "some text", min_probability=0.5, return_probability=True
        )
        assert classification == False  # noqa: E712
        assert probability == pytest.approx(0.2)

    def test_plain_bool_result(self):
        predictor = _is_contract_instance(0.9)
        assert predictor.is_contract("some text") == True  # noqa: E712
        low = _is_contract_instance(0.1)
        assert low.is_contract("some text") == False  # noqa: E712


class TestContractTypeFallbackFailure:
    def test_runtime_fallback_failure_raises_runtime_error(self, monkeypatch):
        def raise_legacy_load_error(cls):
            raise TypeError("legacy model pickle is incompatible")

        monkeypatch.delenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG", raising=False)
        monkeypatch.setattr(
            ProbabilityPredictor,
            "get_default_pipeline",
            classmethod(raise_legacy_load_error),
        )
        monkeypatch.setattr(
            runtime_model,
            "ensure_runtime_contract_type_model",
            lambda *, target_tag: None,
        )

        def fake_load(tag: str):
            raise OSError("runtime model missing")

        monkeypatch.setattr(runtime_model, "load_pipeline_for_tag", fake_load)
        with pytest.raises(RuntimeError, match="runtime fallback model"):
            ProbabilityPredictorContractType.get_default_pipeline()


class TestMakePredictions:
    def test_sorted_descending_and_top_n(self):
        predictor = _contract_type_instance(
            [0.1, 0.7, 0.2], ["AAA", "BBB", "CCC"]
        )
        result = predictor.make_predictions("contract text", top_n=2)
        assert isinstance(result, Series)
        assert list(result.index) == ["BBB", "CCC"]
        assert list(result.values) == pytest.approx([0.7, 0.2])

    def test_top_n_larger_than_classes(self):
        predictor = _contract_type_instance([0.4, 0.6], ["AAA", "BBB"])
        result = predictor.make_predictions("contract text", top_n=5)
        assert len(result) == 2
        assert result.index[0] == "BBB"
        assert result.iloc[0] == pytest.approx(0.6)


class TestInferClassification:
    def test_empty_predictions_returns_unknown(self):
        empty = Series([], dtype=float)
        assert empty.empty
        assert (
            ProbabilityPredictorContractType.infer_classification(
                empty, unknown_classification="UNK"
            )
            == "UNK"
        )

    def test_below_min_probability_returns_unknown(self):
        predictions = Series(
            data=[0.1, 0.05], index=["BBB", "AAA"]
        )
        assert (
            ProbabilityPredictorContractType.infer_classification(
                predictions,
                min_probability=0.15,
                unknown_classification="UNK",
            )
            == "UNK"
        )

    def test_close_runner_up_returns_unknown(self):
        predictions = Series(
            data=[0.46, 0.40], index=["BBB", "AAA"]
        )
        assert (
            ProbabilityPredictorContractType.infer_classification(
                predictions,
                min_probability=0.15,
                max_closest_probability=0.75,
                unknown_classification="UNK",
            )
            == "UNK"
        )

    def test_single_prediction_returns_label(self):
        predictions = Series(data=[0.9], index=["BBB"])
        assert (
            ProbabilityPredictorContractType.infer_classification(
                predictions, unknown_classification="UNK"
            )
            == "BBB"
        )

    def test_confident_prediction_returns_top_label(self):
        predictions = Series(
            data=[0.8, 0.1], index=["BBB", "AAA"]
        )
        assert (
            ProbabilityPredictorContractType.infer_classification(
                predictions, unknown_classification="UNK"
            )
            == "BBB"
        )


class TestDetectContractType:
    def test_detect_known_type(self):
        predictor = _contract_type_instance(
            [0.1, 0.8, 0.1], ["AAA", "BBB", "CCC"]
        )
        assert predictor.detect_contract_type("contract text") == "BBB"

    def test_detect_unknown_type(self):
        predictor = _contract_type_instance(
            [0.05, 0.06], ["AAA", "BBB"]
        )
        assert (
            predictor.detect_contract_type(
                "contract text", unknown_classification="UNK"
            )
            == "UNK"
        )
