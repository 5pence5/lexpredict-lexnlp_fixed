"""Coverage tests for scripts/model_quality_gate.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import model_quality_gate


class TestResolveContractModelTag:
    def test_default_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        assert model_quality_gate.resolve_contract_model_tag() == "pipeline/is-contract/0.2"

    def test_primary_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", "pipeline/custom/1.0")
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/other/1.0")
        assert model_quality_gate.resolve_contract_model_tag() == "pipeline/custom/1.0"

    def test_legacy_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/legacy/0.1")
        assert model_quality_gate.resolve_contract_model_tag() == "pipeline/legacy/0.1"

    def test_value_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", "  pipeline/custom/1.0  ")
        assert model_quality_gate.resolve_contract_model_tag() == "pipeline/custom/1.0"


class TestParseArgs:
    def test_candidate_tag_required(self) -> None:
        with pytest.raises(SystemExit):
            model_quality_gate.parse_args([])

    def test_minimal_argv_uses_defaults(self, tmp_path: Path) -> None:
        fixture = tmp_path / "fix.csv"
        args = model_quality_gate.parse_args(["--candidate-tag", "pipeline/cand/1.0", "--fixture", str(fixture)])
        assert args.candidate_tag == "pipeline/cand/1.0"
        assert args.fixture == fixture
        assert args.baseline_metrics_json is None
        assert args.baseline_metrics_tolerance == 1e-9
        assert args.min_probability == 0.3
        assert args.max_accuracy_regression == 0.0
        assert args.max_f1_regression == 0.0
        assert args.min_candidate_accuracy == 0.0
        assert args.output_json is None
        assert args.write_baseline_metrics_json is None

    def test_baseline_tag_default_tracks_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", "pipeline/env-tag/9.9")
        args = model_quality_gate.parse_args(["--candidate-tag", "pipeline/cand/1.0"])
        assert args.baseline_tag == "pipeline/env-tag/9.9"

    def test_explicit_options_parsed(self, tmp_path: Path) -> None:
        metrics = tmp_path / "m.json"
        out = tmp_path / "out.json"
        written = tmp_path / "written.json"
        args = model_quality_gate.parse_args(
            [
                "--baseline-tag",
                "pipeline/base/0.1",
                "--candidate-tag",
                "pipeline/cand/0.2",
                "--baseline-metrics-json",
                str(metrics),
                "--baseline-metrics-tolerance",
                "0.01",
                "--min-probability",
                "0.5",
                "--max-accuracy-regression",
                "0.02",
                "--max-f1-regression",
                "0.03",
                "--min-candidate-accuracy",
                "0.7",
                "--output-json",
                str(out),
                "--write-baseline-metrics-json",
                str(written),
            ]
        )
        assert args.baseline_tag == "pipeline/base/0.1"
        assert args.baseline_metrics_json == metrics
        assert args.baseline_metrics_tolerance == pytest.approx(0.01)
        assert args.min_probability == pytest.approx(0.5)
        assert args.max_accuracy_regression == pytest.approx(0.02)
        assert args.max_f1_regression == pytest.approx(0.03)
        assert args.min_candidate_accuracy == pytest.approx(0.7)
        assert args.output_json == out
        assert args.write_baseline_metrics_json == written


def _write_fixture_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Text", "Is_Contract"])
        writer.writerows(rows)
    return path


class TestLoadFixture:
    def test_loads_texts_and_labels(self, tmp_path: Path) -> None:
        fixture = _write_fixture_csv(
            tmp_path / "fix.csv",
            [("contract agreement", "true"), ("patent widget", "False"), ("deed transfer", " TRUE ")],
        )
        texts, labels = model_quality_gate.load_fixture(fixture)
        assert texts == ["contract agreement", "patent widget", "deed transfer"]
        assert labels == [True, False, True]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Fixture file not found"):
            model_quality_gate.load_fixture(tmp_path / "nope.csv")

    def test_unexpected_label_raises(self, tmp_path: Path) -> None:
        fixture = _write_fixture_csv(tmp_path / "fix.csv", [("some text", "maybe")])
        with pytest.raises(ValueError, match="Unexpected label value"):
            model_quality_gate.load_fixture(fixture)

    def test_empty_fixture_raises(self, tmp_path: Path) -> None:
        fixture = _write_fixture_csv(tmp_path / "fix.csv", [])
        with pytest.raises(ValueError, match="contains no rows"):
            model_quality_gate.load_fixture(fixture)


class TestEnsureTagDownloaded:
    def test_returns_cached_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        expected = tmp_path / "model.bin"
        expected.write_bytes(b"x")
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: expected)
        calls: list[str] = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append(tag),
        )
        assert model_quality_gate.ensure_tag_downloaded("pipeline/x/1.0") == expected
        assert calls == []

    def test_downloads_on_cache_miss(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        expected = tmp_path / "model.bin"
        expected.write_bytes(b"x")
        attempts = {"n": 0}

        def fake_get(tag: str) -> Path:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise FileNotFoundError(tag)
            return expected

        downloaded: list[str] = []
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", fake_get)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: downloaded.append(tag),
        )
        assert model_quality_gate.ensure_tag_downloaded("pipeline/x/1.0") == expected
        assert downloaded == ["pipeline/x/1.0"]


class TestLoadPipelineForTag:
    def test_roundtrip_through_cloudpickle(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from cloudpickle import dump
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=200))])
        pipeline.fit(["contract agreement", "patent widget"], [1, 0])
        model_path = tmp_path / "model.cloudpickle"
        with model_path.open("wb") as handle:
            dump(pipeline, handle)
        monkeypatch.setattr("model_quality_gate.ensure_tag_downloaded", lambda tag: model_path)
        loaded = model_quality_gate.load_pipeline_for_tag("pipeline/x/1.0")
        assert list(loaded.classes_) == [0, 1]
        assert loaded.predict(["contract agreement"]) is not None


def _make_tiny_pipeline():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=500))])
    pipeline.fit(
        [
            "this agreement is a contract between the parties",
            "contract for the sale of goods hereby agreed",
            "patent describes a widget manufacturing process",
            "weather report predicts sunny skies tomorrow",
        ],
        [1, 1, 0, 0],
    )
    return pipeline


class TestScorePipeline:
    def test_metrics_match_independent_computation(self) -> None:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        from lexnlp.extract.en.contracts.predictors import ProbabilityPredictorIsContract

        pipeline = _make_tiny_pipeline()
        texts = [
            "this agreement is a contract between the parties",
            "patent describes a widget manufacturing process",
            "brand new contract text about obligations",
            "brand new weather text about rainfall",
        ]
        labels = [True, False, True, False]
        result = model_quality_gate.score_pipeline(pipeline, texts, labels, 0.3)
        assert set(result.keys()) == {"accuracy", "f1", "precision", "recall"}
        assert all(isinstance(value, float) for value in result.values())

        predictor = ProbabilityPredictorIsContract(pipeline=pipeline)
        expected_predictions = [bool(predictor.is_contract(text=text, min_probability=0.3)) for text in texts]
        assert result["accuracy"] == float(accuracy_score(labels, expected_predictions))
        assert result["f1"] == float(f1_score(labels, expected_predictions))
        assert result["precision"] == float(precision_score(labels, expected_predictions, zero_division=0))
        assert result["recall"] == float(recall_score(labels, expected_predictions, zero_division=0))

    def test_threshold_changes_predictions(self) -> None:
        pipeline = _make_tiny_pipeline()
        texts = ["contract agreement obligations", "widget patent process"]
        labels = [True, False]
        low = model_quality_gate.score_pipeline(pipeline, texts, labels, 0.01)
        high = model_quality_gate.score_pipeline(pipeline, texts, labels, 0.99)
        assert low["recall"] >= high["recall"]


class TestParseMetrics:
    def test_parses_and_casts_to_float(self) -> None:
        parsed = model_quality_gate.parse_metrics(
            {"accuracy": 1, "f1": "0.5", "precision": 0.25, "recall": 0.75}, "test-source"
        )
        assert parsed == {"accuracy": 1.0, "f1": 0.5, "precision": 0.25, "recall": 0.75}
        assert all(isinstance(value, float) for value in parsed.values())

    def test_missing_keys_raise(self) -> None:
        with pytest.raises(ValueError, match="Missing metric keys in test-source"):
            model_quality_gate.parse_metrics({"accuracy": 1.0}, "test-source")

    def test_missing_keys_named_in_message(self) -> None:
        with pytest.raises(ValueError, match="f1"):
            model_quality_gate.parse_metrics({"accuracy": 1.0, "precision": 1.0}, "src")


class TestLoadBaselineMetrics:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Baseline metrics file not found"):
            model_quality_gate.load_baseline_metrics(tmp_path / "missing.json")

    def test_non_object_payload_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be an object"):
            model_quality_gate.load_baseline_metrics(path)

    def test_metrics_branch(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps(
                {
                    "baseline_tag": "pipeline/base/0.1",
                    "fixture": "some/fixture.csv",
                    "min_probability": 0.3,
                    "metrics": {"accuracy": 0.9, "f1": 0.8, "precision": 0.85, "recall": 0.75},
                }
            ),
            encoding="utf-8",
        )
        loaded = model_quality_gate.load_baseline_metrics(path)
        assert loaded["metrics"] == {"accuracy": 0.9, "f1": 0.8, "precision": 0.85, "recall": 0.75}
        assert loaded["baseline_tag"] == "pipeline/base/0.1"
        assert loaded["fixture"] == "some/fixture.csv"
        assert loaded["min_probability"] == 0.3
        assert loaded["raw"]["baseline_tag"] == "pipeline/base/0.1"

    def test_baseline_branch(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(
            json.dumps({"baseline": {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}}),
            encoding="utf-8",
        )
        loaded = model_quality_gate.load_baseline_metrics(path)
        assert loaded["metrics"] == {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
        assert loaded["baseline_tag"] is None
        assert loaded["fixture"] is None
        assert loaded["min_probability"] is None

    def test_neither_branch_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"other": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain either"):
            model_quality_gate.load_baseline_metrics(path)


PERFECT = {"accuracy": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
WEAK = {"accuracy": 0.5, "f1": 0.4, "precision": 0.5, "recall": 0.4}


def _install_canned_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> None:
    def fake_load(tag: str) -> str:
        return f"pipeline:{tag}"

    def fake_score(pipeline, texts, labels, min_probability: float) -> dict[str, float]:
        if str(pipeline).endswith("base/0.1"):
            return dict(baseline)
        return dict(candidate)

    monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", fake_load)
    monkeypatch.setattr(model_quality_gate, "score_pipeline", fake_score)


class TestMainWithCannedMetrics:
    def _argv(self, tmp_path: Path, extra: list[str] | None = None) -> list[str]:
        fixture = _write_fixture_csv(
            tmp_path / "fix.csv",
            [("contract text", "true"), ("patent text", "false")],
        )
        argv = [
            "--baseline-tag",
            "pipeline/base/0.1",
            "--candidate-tag",
            "pipeline/cand/0.2",
            "--fixture",
            str(fixture),
        ]
        if extra:
            argv.extend(extra)
        return argv

    def test_pass_returns_zero_and_prints_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_canned_models(monkeypatch, baseline=PERFECT, candidate=PERFECT)
        rc = model_quality_gate.main(self._argv(tmp_path))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["baseline_tag"] == "pipeline/base/0.1"
        assert payload["candidate_tag"] == "pipeline/cand/0.2"
        assert payload["baseline"] == PERFECT
        assert payload["candidate"] == PERFECT
        assert payload["baseline_source"] == "tag"
        assert payload["baseline_metrics_json"] is None

    def test_accuracy_regression_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_canned_models(monkeypatch, baseline=PERFECT, candidate=WEAK)
        rc = model_quality_gate.main(self._argv(tmp_path))
        assert rc == 1
        out = capsys.readouterr().out
        assert "QUALITY GATE VIOLATION" in out
        assert "accuracy regression exceeds threshold" in out
        assert "f1 regression exceeds threshold" in out

    def test_regression_within_threshold_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_canned_models(monkeypatch, baseline=PERFECT, candidate=WEAK)
        rc = model_quality_gate.main(
            self._argv(
                tmp_path,
                ["--max-accuracy-regression", "0.6", "--max-f1-regression", "0.7"],
            )
        )
        assert rc == 0

    def test_min_candidate_accuracy_violation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_canned_models(monkeypatch, baseline=WEAK, candidate=WEAK)
        rc = model_quality_gate.main(self._argv(tmp_path, ["--min-candidate-accuracy", "0.9"]))
        assert rc == 1
        assert "candidate accuracy below minimum" in capsys.readouterr().out

    def test_output_json_written_with_parent_creation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_canned_models(monkeypatch, baseline=PERFECT, candidate=PERFECT)
        out = tmp_path / "nested" / "dir" / "result.json"
        rc = model_quality_gate.main(self._argv(tmp_path, ["--output-json", str(out)]))
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["candidate"] == PERFECT

    def test_write_baseline_metrics_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_canned_models(monkeypatch, baseline=PERFECT, candidate=PERFECT)
        written = tmp_path / "sub" / "baseline.json"
        rc = model_quality_gate.main(self._argv(tmp_path, ["--write-baseline-metrics-json", str(written)]))
        assert rc == 0
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["metrics"] == PERFECT
        assert payload["baseline_tag"] == "pipeline/base/0.1"
        assert payload["min_probability"] == pytest.approx(0.3)


class TestMainWithBaselineMetricsJson:
    def _argv_with_metrics(
        self, tmp_path: Path, metrics_path: Path, extra: list[str] | None = None
    ) -> tuple[list[str], Path]:
        fixture = _write_fixture_csv(
            tmp_path / "fix.csv",
            [("contract text", "true"), ("patent text", "false")],
        )
        argv = [
            "--baseline-tag",
            "pipeline/base/0.1",
            "--candidate-tag",
            "pipeline/cand/0.2",
            "--fixture",
            str(fixture),
            "--baseline-metrics-json",
            str(metrics_path),
        ]
        if extra:
            argv.extend(extra)
        return argv, fixture

    def _write_metrics(self, path: Path, fixture: Path, *, tag: str = "pipeline/base/0.1", prob: float = 0.3) -> None:
        path.write_text(
            json.dumps(
                {
                    "baseline_tag": tag,
                    "fixture": str(fixture),
                    "min_probability": prob,
                    "metrics": dict(PERFECT),
                }
            ),
            encoding="utf-8",
        )

    def test_metrics_json_used_as_baseline_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        loaded: list[str] = []
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: loaded.append(tag) or tag)
        monkeypatch.setattr(model_quality_gate, "score_pipeline", lambda pipeline, texts, labels, prob: dict(PERFECT))
        metrics_path = tmp_path / "baseline.json"
        fixture = _write_fixture_csv(tmp_path / "fix.csv", [("contract text", "true")])
        self._write_metrics(metrics_path, fixture)
        rc = model_quality_gate.main(
            [
                "--baseline-tag",
                "pipeline/base/0.1",
                "--candidate-tag",
                "pipeline/cand/0.2",
                "--fixture",
                str(fixture),
                "--baseline-metrics-json",
                str(metrics_path),
            ]
        )
        assert rc == 0
        # Only the candidate model is loaded when metrics JSON is provided.
        assert loaded == ["pipeline/cand/0.2"]
        payload = json.loads(capsys.readouterr().out)
        assert payload["baseline_source"] == "metrics-json"
        assert payload["baseline_metrics_json"] == str(metrics_path)

    def test_baseline_tag_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: tag)
        metrics_path = tmp_path / "baseline.json"
        argv, fixture = self._argv_with_metrics(tmp_path, metrics_path)
        self._write_metrics(metrics_path, fixture, tag="pipeline/other/9.9")
        with pytest.raises(ValueError, match="Baseline tag mismatch"):
            model_quality_gate.main(argv)

    def test_fixture_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: tag)
        metrics_path = tmp_path / "baseline.json"
        argv, _ = self._argv_with_metrics(tmp_path, metrics_path)
        self._write_metrics(metrics_path, tmp_path / "different.csv")
        with pytest.raises(ValueError, match="Fixture mismatch"):
            model_quality_gate.main(argv)

    def test_min_probability_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: tag)
        metrics_path = tmp_path / "baseline.json"
        argv, fixture = self._argv_with_metrics(tmp_path, metrics_path)
        self._write_metrics(metrics_path, fixture, prob=0.9)
        with pytest.raises(ValueError, match="min_probability mismatch"):
            model_quality_gate.main(argv)

    def test_min_probability_within_tolerance_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: tag)
        monkeypatch.setattr(model_quality_gate, "score_pipeline", lambda pipeline, texts, labels, prob: dict(PERFECT))
        metrics_path = tmp_path / "baseline.json"
        argv, fixture = self._argv_with_metrics(tmp_path, metrics_path)
        self._write_metrics(metrics_path, fixture, prob=0.3 + 5e-10)
        assert model_quality_gate.main(argv) == 0


class TestMainEndToEnd:
    def test_real_pipelines_and_baseline_reuse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        pipeline = _make_tiny_pipeline()
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: pipeline)
        fixture = _write_fixture_csv(
            tmp_path / "fix.csv",
            [
                ("this agreement is a contract between the parties", "true"),
                ("weather report predicts sunny skies tomorrow", "false"),
            ],
        )
        written = tmp_path / "baseline.json"
        rc = model_quality_gate.main(
            [
                "--baseline-tag",
                "pipeline/base/0.1",
                "--candidate-tag",
                "pipeline/base/0.1",
                "--fixture",
                str(fixture),
                "--write-baseline-metrics-json",
                str(written),
            ]
        )
        assert rc == 0
        first = json.loads(capsys.readouterr().out)
        assert first["baseline"] == first["candidate"]

        # Reuse the written baseline JSON: baseline must match without loading it.
        loaded: list[str] = []
        monkeypatch.setattr(model_quality_gate, "load_pipeline_for_tag", lambda tag: loaded.append(tag) or pipeline)
        rc = model_quality_gate.main(
            [
                "--baseline-tag",
                "pipeline/base/0.1",
                "--candidate-tag",
                "pipeline/base/0.1",
                "--fixture",
                str(fixture),
                "--baseline-metrics-json",
                str(written),
            ]
        )
        assert rc == 0
        assert loaded == ["pipeline/base/0.1"]
        second = json.loads(capsys.readouterr().out)
        assert second["baseline"] == first["baseline"]


class TestMainGuard:
    def test_run_as_main_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import runpy

        monkeypatch.setattr(sys, "argv", ["model_quality_gate.py"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPTS_DIR / "model_quality_gate.py"), run_name="__main__")
