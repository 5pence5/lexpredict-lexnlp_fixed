"""Coverage tests for :mod:`lexnlp.ml.predictor` missing lines."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lexnlp.ml.predictor import ProbabilityPredictor


class _DummyPredictor(ProbabilityPredictor):
    _DEFAULT_PIPELINE = "pipeline/test/0.1"

    def _sanity_check(self) -> None:
        return None


def _fitted_logreg_pipeline() -> Pipeline:
    pipe: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]], [0, 0, 1, 1])
    return pipe


def _fitted_linreg_pipeline() -> Pipeline:
    pipe: Pipeline = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    pipe.fit([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]], [0.0, 1.0, 2.0, 3.0])
    return pipe


class TestInitSubclass:
    def test_missing_default_pipeline_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="_DEFAULT_PIPELINE"):

            class _BadPredictor(ProbabilityPredictor):  # noqa: F841
                def _sanity_check(self) -> None:
                    return None


class TestUnfittedPipeline:
    def test_unfitted_final_estimator_raises_value_error(self) -> None:
        pipe: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        with pytest.raises(ValueError, match="is not fitted"):
            _DummyPredictor(pipeline=pipe)


class TestPredictProbaProtocol:
    def test_final_estimator_without_predict_proba_raises(self) -> None:
        pipe = _fitted_linreg_pipeline()
        assert not hasattr(pipe._final_estimator, "predict_proba")
        with pytest.raises(ValueError, match="ScikitLearnHasPredictProba"):
            _DummyPredictor(pipeline=pipe)

    def test_fitted_classifier_inits_fine(self) -> None:
        pipe = _fitted_logreg_pipeline()
        predictor = _DummyPredictor(pipeline=pipe)
        assert predictor.pipeline is pipe


class TestLegacyPatch:
    def test_sigma_patches_var_and_variance(self) -> None:
        pipe = _fitted_logreg_pipeline()
        estimator = pipe._final_estimator
        assert not hasattr(estimator, "var_")
        assert not hasattr(estimator, "variance_")
        estimator.sigma_ = 0.5
        predictor = _DummyPredictor(pipeline=pipe)
        assert predictor.pipeline._final_estimator.var_ == 0.5
        assert predictor.pipeline._final_estimator.variance_ == 0.5

    def test_existing_var_not_overwritten_but_variance_patched(self) -> None:
        pipe = _fitted_logreg_pipeline()
        estimator = pipe._final_estimator
        estimator.sigma_ = 0.25
        estimator.var_ = 0.75
        assert not hasattr(estimator, "variance_")
        predictor = _DummyPredictor(pipeline=pipe)
        assert predictor.pipeline._final_estimator.var_ == 0.75
        assert predictor.pipeline._final_estimator.variance_ == 0.75


class TestBaseSanityCheck:
    def test_base_sanity_check_raises(self) -> None:
        pipe = _fitted_logreg_pipeline()
        predictor = _DummyPredictor(pipeline=pipe)
        with pytest.raises(NotImplementedError):
            ProbabilityPredictor._sanity_check(predictor)
