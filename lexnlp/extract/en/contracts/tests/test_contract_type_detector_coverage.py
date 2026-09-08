"""Coverage tests for lexnlp.extract.en.contracts.contract_type_detector."""

from __future__ import annotations

import joblib
import pytest
from gensim.models.doc2vec import Doc2Vec
from pandas import Series

from lexnlp.extract.en.contracts.contract_type_detector import ContractTypeDetector


def _vector(values: list[float], labels: list[str]) -> Series:
    return Series(values, index=labels)


class TestInit:
    def test_models_are_loaded_from_paths(self, tmp_path, monkeypatch) -> None:
        rf_path = tmp_path / "rf.model"
        d2v_path = tmp_path / "d2v.model"
        rf_path.write_bytes(b"rf")
        d2v_path.write_bytes(b"d2v")
        rf_sentinel = object()
        d2v_sentinel = object()
        seen: dict[str, object] = {}

        def _fake_joblib_load(handle):
            seen["rf_name"] = handle.name
            return rf_sentinel

        def _fake_d2v_load(path):
            seen["d2v_path"] = path
            return d2v_sentinel

        monkeypatch.setattr(joblib, "load", _fake_joblib_load)
        monkeypatch.setattr(Doc2Vec, "load", _fake_d2v_load)
        detector = ContractTypeDetector(str(rf_path), str(d2v_path))
        assert detector.rf_model is rf_sentinel
        assert detector.d2v_model is d2v_sentinel
        assert seen == {"rf_name": str(rf_path), "d2v_path": str(d2v_path)}


class TestDetectContractType:
    def test_empty_vector_returns_unknown(self) -> None:
        assert ContractTypeDetector.detect_contract_type(Series([], dtype=float), unknown_category="UNK") == "UNK"

    def test_default_unknown_is_empty_string(self) -> None:
        assert ContractTypeDetector.detect_contract_type(Series([], dtype=float)) == ""

    def test_below_min_prob_returns_unknown(self) -> None:
        vector = _vector([0.10, 0.05], ["AAA", "BBB"])
        assert ContractTypeDetector.detect_contract_type(vector, min_prob=0.15, unknown_category="?") == "?"

    def test_close_runner_up_returns_unknown(self) -> None:
        vector = _vector([0.50, 0.40], ["AAA", "BBB"])
        assert ContractTypeDetector.detect_contract_type(vector, unknown_category="?") == "?"

    def test_clear_winner_returns_top_label(self) -> None:
        vector = _vector([0.80, 0.10], ["AAA", "BBB"])
        assert ContractTypeDetector.detect_contract_type(vector, unknown_category="?") == "AAA"

    def test_single_entry_vector_returns_top_label(self) -> None:
        vector = _vector([0.90], ["AAA"])
        assert ContractTypeDetector.detect_contract_type(vector, unknown_category="?") == "AAA"

    def test_runner_up_exactly_at_threshold_wins(self) -> None:
        vector = _vector([0.50, 0.40], ["AAA", "BBB"])
        assert (
            ContractTypeDetector.detect_contract_type(vector, max_closest_prob_percent=80, unknown_category="?")
            == "AAA"
        )


class TestDetectContractTypeVector:
    def _detector(self, rf_model, d2v_model) -> ContractTypeDetector:
        detector = ContractTypeDetector.__new__(ContractTypeDetector)
        detector.rf_model = rf_model
        detector.d2v_model = d2v_model
        return detector

    def test_uninitialized_raises(self) -> None:
        detector = self._detector(None, None)
        with pytest.raises(RuntimeError, match="not initialized"):
            detector.detect_contract_type_vector("some text")

    def test_uninitialized_d2v_raises(self) -> None:
        detector = self._detector(object(), None)
        with pytest.raises(RuntimeError, match="not initialized"):
            detector.detect_contract_type_vector("some text")

    def test_vector_is_sorted_descending(self) -> None:
        seen: dict[str, object] = {}

        class _FakeD2V:
            def infer_vector(self, tokens: list[str]):
                seen["tokens"] = tokens
                assert tokens == ContractTypeDetector.process_document("hello world")
                return [1.0, 2.0]

        class _FakeRF:
            classes_ = ["X", "Y", "Z"]

            def predict_proba(self, vectors):
                seen["vectors"] = vectors
                assert vectors == [[1.0, 2.0]]
                return [[0.20, 0.70, 0.10]]

        detector = self._detector(_FakeRF(), _FakeD2V())
        result = detector.detect_contract_type_vector("hello world")
        assert list(result.index) == ["Y", "X", "Z"]
        assert list(result.values) == pytest.approx([0.70, 0.20, 0.10])
        assert seen["tokens"] == ["hello", "world"]


class TestProcessDocument:
    def test_lowercase_alpha_only(self) -> None:
        assert ContractTypeDetector.process_document("Hello World! 123 Testing, foobar.") == [
            "hello",
            "world",
            "testing",
            "foobar",
        ]

    def test_stopwords_and_numbers_dropped(self) -> None:
        assert ContractTypeDetector.process_document("The parties agree to the terms here.") == [
            "parties",
            "agree",
            "terms",
        ]

    def test_empty_text_gives_empty_list(self) -> None:
        assert ContractTypeDetector.process_document("") == []
