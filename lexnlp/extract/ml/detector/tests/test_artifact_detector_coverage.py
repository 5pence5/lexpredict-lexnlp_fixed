"""Coverage tests for lexnlp.extract.ml.detector.artifact_detector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import num2words
import numpy
import pandas
import pytest

from lexnlp.extract.ml.classifier.base_token_sequence_classifier_model import (
    BaseTokenSequenceClassifierModel,
)
from lexnlp.extract.ml.detector.artifact_detector import ArtifactDetector
from lexnlp.extract.ml.detector.detecting_settings import DetectingSettings


class FakeSklearnModel:
    def __init__(self, predictions: numpy.ndarray) -> None:
        self.predictions = predictions
        self.seen: list[numpy.ndarray] = []

    def predict(self, features: numpy.ndarray) -> numpy.ndarray:
        self.seen.append(features)
        return self.predictions


class DummyDetector(ArtifactDetector):
    """Concrete detector with deterministic sample processing."""

    def __init__(self, feature_data: numpy.ndarray, target_data: numpy.ndarray) -> None:
        super().__init__()
        self.feature_data = feature_data
        self.target_data = target_data
        self.seen_sizes: list[int] = []

    def process_sample(
        self, sample_df: pandas.DataFrame, build_target_data: bool = False
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        self.seen_sizes.append(len(sample_df))
        return self.feature_data, self.target_data


def _detector_with_model(
    predictions: numpy.ndarray | None = None,
    feature_rows: int = 3,
    feature_cols: int = 2,
) -> DummyDetector:
    feature_data = numpy.arange(feature_rows * feature_cols, dtype=float).reshape(feature_rows, feature_cols)
    target_data = numpy.zeros(feature_rows, dtype=int)
    detector = DummyDetector(feature_data, target_data)
    predicted = predictions if predictions is not None else numpy.zeros(feature_rows, dtype=int)
    detector.model = SimpleNamespace(
        model=FakeSklearnModel(predicted),
        get_feature_data=lambda text, mask=None: (
            feature_data,
            [(0, 5), (6, 11)][:feature_rows],
        ),
    )
    return detector


class TestLoadPaths:
    def test_load_delegates_to_file_loader(self, monkeypatch, tmp_path) -> None:
        sentinel = object()
        seen: dict[str, Any] = {}

        def fake_load(path: str) -> object:
            seen["path"] = path
            return sentinel

        monkeypatch.setattr(BaseTokenSequenceClassifierModel, "load_from_file", staticmethod(fake_load))
        detector = ArtifactDetector()
        detector.load(str(tmp_path / "model.bin"))
        assert detector.model is sentinel
        assert seen["path"] == str(tmp_path / "model.bin")

    def test_load_compressed_delegates_to_compressed_loader(self, monkeypatch, tmp_path) -> None:
        sentinel = object()
        seen: dict[str, Any] = {}

        def fake_load(path: str) -> object:
            seen["path"] = path
            return sentinel

        monkeypatch.setattr(BaseTokenSequenceClassifierModel, "load_from_file_compressed", staticmethod(fake_load))
        detector = ArtifactDetector()
        detector.load_compressed(str(tmp_path / "model.gz"))
        assert detector.model is sentinel
        assert seen["path"] == str(tmp_path / "model.gz")

    def test_load_from_stream_delegates_to_stream_loader(self, monkeypatch) -> None:
        sentinel = object()
        stream = object()
        seen: dict[str, Any] = {}

        def fake_load(data: object) -> object:
            seen["stream"] = data
            return sentinel

        monkeypatch.setattr(BaseTokenSequenceClassifierModel, "load_from_stream", staticmethod(fake_load))
        detector = ArtifactDetector()
        detector.load_from_stream(stream)
        assert detector.model is sentinel
        assert seen["stream"] is stream


class TestBaseProcessSample:
    def test_base_process_sample_raises_not_implemented(self) -> None:
        detector = ArtifactDetector()
        sample = pandas.DataFrame([{"sentence": "some text"}])
        with pytest.raises(NotImplementedError, match="process_sample"):
            ArtifactDetector.process_sample(detector, sample)


class TestPredict:
    def test_predict_without_limit_uses_full_sample(self) -> None:
        detector = _detector_with_model(predictions=numpy.array([1, 0, 1]))
        sample = pandas.DataFrame([{"sentence": f"text {i}"} for i in range(3)])
        predicted, target = detector.predict(sample)
        assert list(predicted) == [1, 0, 1]
        assert list(target) == [0, 0, 0]
        assert detector.seen_sizes == [3]

    def test_predict_with_size_limit_truncates_sample(self) -> None:
        detector = _detector_with_model(predictions=numpy.array([1, 0, 1]))
        sample = pandas.DataFrame([{"sentence": f"text {i}"} for i in range(3)])
        predicted, _target = detector.predict(sample, size_limit=2)
        assert detector.seen_sizes == [2]
        assert list(predicted) == [1, 0, 1]


class TestPredictText:
    def test_predict_text_joins_start_end_into_char_span(self) -> None:
        detector = _detector_with_model(feature_rows=2)
        detector.model.model = FakeSklearnModel(numpy.array([1, 3]))
        spans = list(detector.predict_text("hello world"))
        assert spans == [(0, 11)]

    def test_predict_text_forwards_feature_mask(self) -> None:
        seen: dict[str, Any] = {}
        feature_data = numpy.zeros((2, 2))
        tokens = [(0, 5), (6, 11)]

        def fake_features(text: str, mask: list[int] | None = None) -> tuple[numpy.ndarray, list[tuple[int, int]]]:
            seen["text"] = text
            seen["mask"] = mask
            return feature_data, tokens

        detector = _detector_with_model(feature_rows=2)
        detector.model.get_feature_data = fake_features
        detector.model.model = FakeSklearnModel(numpy.array([0, 0]))
        assert list(detector.predict_text("hello world", feature_mask=[1, 0])) == []
        assert seen["text"] == "hello world"
        assert seen["mask"] == [1, 0]


class TestTrainAndSave:
    def test_train_and_save_wires_sample_tokens_and_tokens_training(self, monkeypatch) -> None:
        seen: dict[str, Any] = {}
        sample = pandas.DataFrame([{"sentence": "a b"}])
        tokens = ["first", "second"]
        settings = DetectingSettings()

        def fake_read(train_file: str, train_size: int) -> pandas.DataFrame:
            seen["train_file"] = train_file
            seen["train_size"] = train_size
            return sample

        def fake_tokens() -> list[str]:
            return tokens

        def fake_train(
            amount_tokens: list[str],
            save_path: str,
            active: DetectingSettings,
            train_df: pandas.DataFrame,
            compress: bool = False,
        ) -> None:
            seen["amount_tokens"] = amount_tokens
            seen["save_path"] = save_path
            seen["settings"] = active
            seen["train_df"] = train_df
            seen["compress"] = compress

        detector = DummyDetector(numpy.zeros((1, 1)), numpy.zeros(1, dtype=int))
        monkeypatch.setattr(detector, "read_sample_df", fake_read)
        monkeypatch.setattr(detector, "build_amount_tokens", fake_tokens)
        monkeypatch.setattr(detector, "train_and_save_on_tokens", fake_train)
        detector.train_and_save(settings, "train.csv", train_size=7, save_path="out.bin", compress=True)
        assert seen["train_file"] == "train.csv"
        assert seen["train_size"] == 7
        assert seen["amount_tokens"] == tokens
        assert seen["save_path"] == "out.bin"
        assert seen["settings"] is settings
        assert seen["train_df"] is sample
        assert seen["compress"] is True

    @pytest.mark.parametrize("model_type", ["extra_trees", "random_forest"])
    def test_train_on_tokens_trains_and_saves_uncompressed(self, tmp_path, model_type: str) -> None:
        feature_data = numpy.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        target_data = numpy.array([0, 1, 0, 1])
        detector = DummyDetector(feature_data, target_data)
        settings = DetectingSettings(model_type=model_type)
        sample = pandas.DataFrame([{"sentence": "a b"}])
        save_path = str(tmp_path / "model.bin")
        detector.train_and_save_on_tokens(["first"], save_path, settings, sample)
        assert detector.model is not None
        assert detector.model.model is not None
        with open(save_path, "rb") as handle:
            assert len(handle.read()) > 0

    def test_train_on_tokens_saves_compressed(self, tmp_path) -> None:
        feature_data = numpy.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        target_data = numpy.array([0, 1, 0, 1])
        detector = DummyDetector(feature_data, target_data)
        settings = DetectingSettings(model_type="extra_trees")
        sample = pandas.DataFrame([{"sentence": "a b"}])
        save_path = str(tmp_path / "model.gz")
        detector.train_and_save_on_tokens(["first"], save_path, settings, sample, compress=True)
        with open(save_path, "rb") as handle:
            assert handle.read(2) == b"\x1f\x8b"


class TestSaveHelpers:
    def test_save_model_delegates_with_path(self, tmp_path) -> None:
        seen: dict[str, Any] = {}
        detector = ArtifactDetector()
        detector.model = SimpleNamespace(save_in_file=lambda path: seen.setdefault("path", path))
        target = str(tmp_path / "model.bin")
        detector.save_model(target)
        assert seen["path"] == target

    def test_save_compressed_model_delegates_with_path(self, tmp_path) -> None:
        seen: dict[str, Any] = {}
        detector = ArtifactDetector()
        detector.model = SimpleNamespace(save_in_file_compressed=lambda path: seen.setdefault("path", path))
        target = str(tmp_path / "model.gz")
        detector.save_compressed_model(target)
        assert seen["path"] == target


class TestReadSampleDf:
    def test_positive_train_size_returns_head(self, tmp_path) -> None:
        csv_path = tmp_path / "train.csv"
        pandas.DataFrame([{"sentence": f"text {i}"} for i in range(5)]).to_csv(csv_path, index=False)
        detector = ArtifactDetector()
        sample = detector.read_sample_df(str(csv_path), 2)
        assert list(sample["sentence"]) == ["text 0", "text 1"]

    def test_non_positive_train_size_returns_everything(self, tmp_path) -> None:
        csv_path = tmp_path / "train.csv"
        pandas.DataFrame([{"sentence": f"text {i}"} for i in range(4)]).to_csv(csv_path, index=False)
        detector = ArtifactDetector()
        assert len(detector.read_sample_df(str(csv_path), 0)) == 4
        assert len(detector.read_sample_df(str(csv_path), -1)) == 4


class TestBuildAmountTokens:
    def test_builds_ordinal_variants_plus_quantity_words(self) -> None:
        detector = ArtifactDetector()
        tokens = detector.build_amount_tokens()
        assert len(tokens) == 100 * 6 + 7
        assert tokens[:6] == [
            num2words.num2words(1, to="ordinal"),
            num2words.num2words(1, to="ordinal").lower(),
            num2words.num2words(1, to="ordinal").upper(),
            num2words.num2words(1, to="ordinal_num"),
            num2words.num2words(1, to="ordinal_num").lower(),
            num2words.num2words(1, to="ordinal_num").upper(),
        ]
        assert tokens[-7:] == ["dozen", "million", "millionth", "billion", "billionth", "trillion", "trillionth"]
        assert "first" in tokens
