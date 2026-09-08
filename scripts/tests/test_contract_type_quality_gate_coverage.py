"""Coverage tests for scripts/contract_type_quality_gate.py.

Catalog downloads and model scoring are mocked; fixture parsing, baseline
metrics loading and the gate logic in ``main()`` run for real.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import contract_type_quality_gate as gate  # required: import follows sys.path insertion for test-only script module resolution


def _write_fixture(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Text", "Contract_Type"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fixture_arg(tmp_path: Path) -> tuple[Path, list[str]]:
    fixture = _write_fixture(
        tmp_path / "fix.csv",
        [
            {"Text": "non-disclosure agreement", "Contract_Type": "NDA"},
            {"Text": "master services agreement", "Contract_Type": "MSA"},
        ],
    )
    return fixture, ["--fixture", str(fixture)]


def _write_baseline_json(
    path: Path,
    *,
    baseline_tag: str = "base",
    fixture: str = "fix.csv",
    top_n: int = 3,
    metrics: dict[str, float] | None = None,
    key: str = "metrics",
    raw: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = dict(raw) if raw is not None else {}
    if raw is None:
        payload = {
            "baseline_tag": baseline_tag,
            "fixture": fixture,
            "top_n": top_n,
            key: metrics
            or {
                "accuracy_top1": 0.5,
                "accuracy_topn": 1.0,
                "f1_macro": 0.5,
                "f1_weighted": 0.5,
            },
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadFixtureEmpty:
    def test_header_only_fixture_raises(self, tmp_path: Path) -> None:
        fixture = _write_fixture(tmp_path / "empty.csv", [])
        with pytest.raises(ValueError, match="no rows"):
            gate.load_fixture(fixture)


class TestEnsureTagDownloaded:
    def test_catalog_hit_skips_download(self, tmp_path: Path, monkeypatch) -> None:
        import lexnlp.ml.catalog as catalog
        import lexnlp.ml.catalog.download as download

        expected = tmp_path / "model"
        calls: list[str] = []
        monkeypatch.setattr(catalog, "get_path_from_catalog", lambda tag: expected)
        monkeypatch.setattr(download, "download_github_release", lambda tag, prompt_user: calls.append(tag))
        assert gate.ensure_tag_downloaded("some-tag") == expected
        assert calls == []

    def test_catalog_miss_downloads_then_resolves(self, tmp_path: Path, monkeypatch) -> None:
        import lexnlp.ml.catalog as catalog
        import lexnlp.ml.catalog.download as download

        expected = tmp_path / "model"
        calls: list[tuple[str, bool]] = []
        state = {"missed": False}

        def fake_get(tag: str) -> Path:
            if not state["missed"]:
                state["missed"] = True
                raise FileNotFoundError(tag)
            return expected

        def fake_download(tag: str, prompt_user: bool = True) -> None:
            calls.append((tag, prompt_user))

        monkeypatch.setattr(catalog, "get_path_from_catalog", fake_get)
        monkeypatch.setattr(download, "download_github_release", fake_download)
        assert gate.ensure_tag_downloaded("some-tag") == expected
        assert calls == [("some-tag", False)]


class TestLoadPipelineForTag:
    def test_routes_through_model_io(self, tmp_path: Path, monkeypatch) -> None:
        import lexnlp.ml.model_io as model_io

        tag_path = tmp_path / "tag"
        seen: list[Path] = []
        monkeypatch.setattr(gate, "ensure_tag_downloaded", lambda tag: tag_path)
        monkeypatch.setattr(model_io, "load_model", lambda path, **kwargs: seen.append(path) or "PIPE")
        assert gate.load_pipeline_for_tag("candidate") == "PIPE"
        assert seen == [tag_path]


class TestLoadBaselineMetrics:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            gate.load_baseline_metrics(tmp_path / "nope.json")

    def test_non_object_payload_raises(self, tmp_path: Path) -> None:
        payload = tmp_path / "m.json"
        payload.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be an object"):
            gate.load_baseline_metrics(payload)

    def test_metrics_key(self, tmp_path: Path) -> None:
        payload = tmp_path / "m.json"
        _write_baseline_json(payload, baseline_tag="b", fixture="f.csv", top_n=3)
        result = gate.load_baseline_metrics(payload)
        assert result["metrics"]["accuracy_top1"] == pytest.approx(0.5)
        assert result["baseline_tag"] == "b"
        assert result["fixture"] == "f.csv"
        assert result["top_n"] == 3
        assert result["raw"]["baseline_tag"] == "b"

    def test_baseline_key_legacy(self, tmp_path: Path) -> None:
        payload = tmp_path / "m.json"
        _write_baseline_json(
            payload,
            key="baseline",
            metrics={
                "accuracy_top1": 0.7,
                "accuracy_topn": 0.8,
                "f1_macro": 0.6,
                "f1_weighted": 0.65,
            },
        )
        result = gate.load_baseline_metrics(payload)
        assert result["metrics"]["accuracy_top1"] == pytest.approx(0.7)

    def test_missing_section_raises(self, tmp_path: Path) -> None:
        payload = tmp_path / "m.json"
        payload.write_text(json.dumps({"baseline_tag": "b"}), encoding="utf-8")
        with pytest.raises(ValueError, match=r"metrics.*baseline"):
            gate.load_baseline_metrics(payload)


class TestMain:
    def test_top_n_zero_raises(self, tmp_path: Path) -> None:
        _, fixture_args = _fixture_arg(tmp_path)
        with pytest.raises(ValueError, match="--top-n"):
            gate.main(["--candidate-tag", "c", "--top-n", "0", *fixture_args])

    def test_metrics_json_passes_and_reports(self, tmp_path: Path, monkeypatch, capsys) -> None:
        fixture, fixture_args = _fixture_arg(tmp_path)
        baseline = tmp_path / "base.json"
        _write_baseline_json(baseline, baseline_tag="base", fixture=str(fixture), top_n=3)
        candidate_metrics = {
            "accuracy_top1": 0.5,
            "accuracy_topn": 1.0,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
        }
        monkeypatch.setattr(gate, "load_pipeline_for_tag", lambda tag: f"pipe:{tag}")
        monkeypatch.setattr(gate, "score_pipeline", lambda pipe, texts, labels, top_n: candidate_metrics)
        code = gate.main(
            [
                "--candidate-tag",
                "cand",
                "--baseline-tag",
                "base",
                "--baseline-metrics-json",
                str(baseline),
                *fixture_args,
            ]
        )
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is True
        assert result["baseline_source"] == "metrics-json"
        assert result["baseline_metrics_json"] == str(baseline)
        assert len(result["checks"]) == 4
        assert all(check["passed"] for check in result["checks"])

    def test_tag_baseline_scores_both_models(self, tmp_path: Path, monkeypatch, capsys) -> None:
        _, fixture_args = _fixture_arg(tmp_path)
        baseline_metrics = {
            "accuracy_top1": 0.5,
            "accuracy_topn": 1.0,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
        }
        candidate_metrics = dict(baseline_metrics)
        pipes: list[str] = []
        scores: list[str] = []

        def fake_load(tag: str) -> str:
            pipes.append(tag)
            return f"pipe:{tag}"

        def fake_score(pipe: str, texts: list[str], labels: list[str], top_n: int) -> dict[str, float]:
            scores.append(pipe)
            assert top_n == 3
            assert texts == ["non-disclosure agreement", "master services agreement"]
            assert labels == ["NDA", "MSA"]
            return baseline_metrics if pipe == "pipe:base" else candidate_metrics

        monkeypatch.setattr(gate, "load_pipeline_for_tag", fake_load)
        monkeypatch.setattr(gate, "score_pipeline", fake_score)
        code = gate.main(["--candidate-tag", "cand", "--baseline-tag", "base", *fixture_args])
        assert code == 0
        assert pipes == ["base", "cand"]
        assert scores == ["pipe:base", "pipe:cand"]
        result = json.loads(capsys.readouterr().out)
        assert result["baseline_source"] == "tag"
        assert result["candidate"] == candidate_metrics

    def test_baseline_tag_mismatch_raises(self, tmp_path: Path, monkeypatch) -> None:
        fixture, fixture_args = _fixture_arg(tmp_path)
        baseline = tmp_path / "base.json"
        _write_baseline_json(baseline, baseline_tag="other", fixture=str(fixture), top_n=3)
        with pytest.raises(ValueError, match="Baseline tag mismatch"):
            gate.main(
                [
                    "--candidate-tag",
                    "c",
                    "--baseline-tag",
                    "base",
                    "--baseline-metrics-json",
                    str(baseline),
                    *fixture_args,
                ]
            )

    def test_fixture_mismatch_raises(self, tmp_path: Path) -> None:
        _fixture, fixture_args = _fixture_arg(tmp_path)
        baseline = tmp_path / "base.json"
        _write_baseline_json(baseline, baseline_tag="base", fixture="different.csv", top_n=3)
        with pytest.raises(ValueError, match="Fixture mismatch"):
            gate.main(
                [
                    "--candidate-tag",
                    "c",
                    "--baseline-tag",
                    "base",
                    "--baseline-metrics-json",
                    str(baseline),
                    *fixture_args,
                ]
            )

    def test_top_n_mismatch_raises(self, tmp_path: Path) -> None:
        fixture, fixture_args = _fixture_arg(tmp_path)
        baseline = tmp_path / "base.json"
        _write_baseline_json(baseline, baseline_tag="base", fixture=str(fixture), top_n=5)
        with pytest.raises(ValueError, match="top_n mismatch"):
            gate.main(
                [
                    "--candidate-tag",
                    "c",
                    "--baseline-tag",
                    "base",
                    "--baseline-metrics-json",
                    str(baseline),
                    *fixture_args,
                ]
            )

    def test_regression_fails_gate(self, tmp_path: Path, monkeypatch, capsys) -> None:
        _, fixture_args = _fixture_arg(tmp_path)
        perfect = {
            "accuracy_top1": 1.0,
            "accuracy_topn": 1.0,
            "f1_macro": 1.0,
            "f1_weighted": 1.0,
        }
        weak = {
            "accuracy_top1": 0.0,
            "accuracy_topn": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }
        monkeypatch.setattr(gate, "load_pipeline_for_tag", lambda tag: tag)
        monkeypatch.setattr(
            gate,
            "score_pipeline",
            lambda pipe, texts, labels, top_n: perfect if pipe == "base" else weak,
        )
        code = gate.main(["--candidate-tag", "c", "--baseline-tag", "base", *fixture_args])
        assert code == 1
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is False
        by_metric = {check["metric"]: check for check in result["checks"]}
        assert by_metric["accuracy_top1"]["delta"] == pytest.approx(-1.0)
        assert by_metric["accuracy_top1"]["passed"] is False

    def test_regression_within_tolerance_passes(self, tmp_path: Path, monkeypatch, capsys) -> None:
        _, fixture_args = _fixture_arg(tmp_path)
        baseline_metrics = {
            "accuracy_top1": 1.0,
            "accuracy_topn": 1.0,
            "f1_macro": 1.0,
            "f1_weighted": 1.0,
        }
        candidate_metrics = {
            "accuracy_top1": 0.9,
            "accuracy_topn": 1.0,
            "f1_macro": 1.0,
            "f1_weighted": 1.0,
        }
        monkeypatch.setattr(gate, "load_pipeline_for_tag", lambda tag: tag)
        monkeypatch.setattr(
            gate,
            "score_pipeline",
            lambda pipe, texts, labels, top_n: baseline_metrics if pipe == "base" else candidate_metrics,
        )
        code = gate.main(
            [
                "--candidate-tag",
                "c",
                "--baseline-tag",
                "base",
                "--max-accuracy-top1-regression",
                "0.2",
                *fixture_args,
            ]
        )
        assert code == 0
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is True

    def test_min_candidate_accuracy_fails(self, tmp_path: Path, monkeypatch, capsys) -> None:
        _, fixture_args = _fixture_arg(tmp_path)
        metrics = {
            "accuracy_top1": 0.5,
            "accuracy_topn": 1.0,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
        }
        monkeypatch.setattr(gate, "load_pipeline_for_tag", lambda tag: tag)
        monkeypatch.setattr(gate, "score_pipeline", lambda pipe, texts, labels, top_n: metrics)
        code = gate.main(
            [
                "--candidate-tag",
                "c",
                "--baseline-tag",
                "base",
                "--min-candidate-accuracy-top1",
                "0.99",
                *fixture_args,
            ]
        )
        assert code == 1
        result = json.loads(capsys.readouterr().out)
        assert result["passed"] is False
        assert result["checks"][-1]["metric"] == "min_candidate_accuracy_top1"
        assert result["checks"][-1]["min_required"] == pytest.approx(0.99)

    def test_writes_baseline_and_output_json(self, tmp_path: Path, monkeypatch) -> None:
        fixture, fixture_args = _fixture_arg(tmp_path)
        metrics = {
            "accuracy_top1": 0.5,
            "accuracy_topn": 1.0,
            "f1_macro": 0.5,
            "f1_weighted": 0.5,
        }
        monkeypatch.setattr(gate, "load_pipeline_for_tag", lambda tag: tag)
        monkeypatch.setattr(gate, "score_pipeline", lambda pipe, texts, labels, top_n: metrics)
        baseline_out = tmp_path / "nested" / "baseline.json"
        result_out = tmp_path / "nested" / "result.json"
        code = gate.main(
            [
                "--candidate-tag",
                "cand",
                "--baseline-tag",
                "base",
                "--write-baseline-metrics-json",
                str(baseline_out),
                "--output-json",
                str(result_out),
                *fixture_args,
            ]
        )
        assert code == 0
        baseline_payload = json.loads(baseline_out.read_text(encoding="utf-8"))
        assert baseline_payload["baseline_tag"] == "base"
        assert baseline_payload["fixture"] == str(fixture)
        assert baseline_payload["top_n"] == 3
        assert baseline_payload["metrics"] == metrics
        result_payload = json.loads(result_out.read_text(encoding="utf-8"))
        assert result_payload["passed"] is True
        assert result_payload["candidate_tag"] == "cand"

    def test_resolve_default_tag(self, monkeypatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG", raising=False)
        assert gate.resolve_contract_type_model_tag() == "pipeline/contract-type/0.2-runtime"
        monkeypatch.setenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG", "  custom/tag  ")
        assert gate.resolve_contract_type_model_tag() == "custom/tag"
