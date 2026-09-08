"""Coverage tests for scripts/reexport_contract_model.py."""

from __future__ import annotations

import json
import pickle
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import reexport_contract_model as script_mod


class TestResolveContractModelTag:
    def test_default_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        assert script_mod.resolve_contract_model_tag() == "pipeline/is-contract/0.1"

    def test_primary_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", " pipeline/custom/1.0 ")
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/other/1.0")
        assert script_mod.resolve_contract_model_tag() == "pipeline/custom/1.0"

    def test_legacy_env_used_when_primary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/legacy/0.9")
        assert script_mod.resolve_contract_model_tag() == "pipeline/legacy/0.9"


class TestParseArgs:
    def test_target_tag_is_required(self) -> None:
        with pytest.raises(SystemExit):
            script_mod.parse_args([])

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        args = script_mod.parse_args(["--target-tag", "pipeline/out/1"])
        assert args.source_tag == "pipeline/is-contract/0.1"
        assert args.target_tag == "pipeline/out/1"
        assert args.force is False
        assert args.skip_quality_gate is False
        assert args.fixture == script_mod.DEFAULT_FIXTURE
        assert args.baseline_metrics_json == script_mod.DEFAULT_BASELINE_METRICS
        assert args.max_accuracy_regression == pytest.approx(0.0)
        assert args.max_f1_regression == pytest.approx(0.0)
        assert args.min_probability == pytest.approx(0.3)
        assert args.output_metadata_json is None
        assert args.max_legacy_warning_regression == 0

    def test_explicit_options(self, tmp_path: Path) -> None:
        fixture = tmp_path / "fix.csv"
        metrics = tmp_path / "metrics.json"
        meta = tmp_path / "meta.json"
        args = script_mod.parse_args(
            [
                "--source-tag",
                "pipeline/src/0.1",
                "--target-tag",
                "pipeline/dst/0.2",
                "--force",
                "--skip-quality-gate",
                "--fixture",
                str(fixture),
                "--baseline-metrics-json",
                str(metrics),
                "--max-accuracy-regression",
                "0.05",
                "--max-f1-regression",
                "0.07",
                "--min-probability",
                "0.4",
                "--output-metadata-json",
                str(meta),
                "--max-legacy-warning-regression",
                "2",
            ]
        )
        assert args.source_tag == "pipeline/src/0.1"
        assert args.target_tag == "pipeline/dst/0.2"
        assert args.force is True
        assert args.skip_quality_gate is True
        assert args.fixture == fixture
        assert args.baseline_metrics_json == metrics
        assert args.max_accuracy_regression == pytest.approx(0.05)
        assert args.max_f1_regression == pytest.approx(0.07)
        assert args.min_probability == pytest.approx(0.4)
        assert args.output_metadata_json == meta
        assert args.max_legacy_warning_regression == 2


class TestEnsureTagDownloaded:
    def test_returns_catalog_path_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model = tmp_path / "model.cloudpickle"
        model.write_bytes(b"ok")
        downloads: list[tuple[str, bool]] = []

        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=True: downloads.append((tag, prompt_user)),
        )
        assert script_mod.ensure_tag_downloaded("pipeline/is-contract/0.1") == model
        assert downloads == []

    def test_downloads_when_catalog_misses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model = tmp_path / "model.cloudpickle"
        model.write_bytes(b"ok")
        calls = {"n": 0}
        downloads: list[tuple[str, bool]] = []

        def get_path(tag: str) -> Path:
            calls["n"] += 1
            if calls["n"] == 1:
                raise FileNotFoundError(f"missing {tag}")
            return model

        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", get_path)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=True: downloads.append((tag, prompt_user)),
        )
        assert script_mod.ensure_tag_downloaded("pipeline/is-contract/0.1") == model
        assert downloads == [("pipeline/is-contract/0.1", False)]
        assert calls["n"] == 2


class TestRunQualityGate:
    def test_includes_baseline_metrics_when_file_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        metrics = tmp_path / "metrics.json"
        metrics.write_text("{}", encoding="utf-8")
        fixture = tmp_path / "fix.csv"
        seen: dict[str, object] = {}

        def fake_run(cmd, check, capture_output, text):
            seen["cmd"] = cmd
            seen["check"] = check
            seen["capture_output"] = capture_output
            seen["text"] = text
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        monkeypatch.setattr(script_mod.subprocess, "run", fake_run)
        script_mod.run_quality_gate(
            source_tag="pipeline/src/0.1",
            target_tag="pipeline/dst/0.2",
            fixture=fixture,
            baseline_metrics_json=metrics,
            min_probability=0.3,
            max_accuracy_regression=0.01,
            max_f1_regression=0.02,
        )
        cmd = seen["cmd"]
        assert seen["check"] is True
        assert seen["capture_output"] is True
        assert seen["text"] is True
        assert cmd[0] == sys.executable
        assert cmd[1].endswith("model_quality_gate.py")
        assert "--baseline-tag" in cmd and "pipeline/src/0.1" in cmd
        assert "--candidate-tag" in cmd and "pipeline/dst/0.2" in cmd
        assert "--fixture" in cmd and str(fixture) in cmd
        assert "--baseline-metrics-json" in cmd and str(metrics) in cmd
        assert "--min-probability" in cmd and "0.3" in cmd
        assert "--max-accuracy-regression" in cmd and "0.01" in cmd
        assert "--max-f1-regression" in cmd and "0.02" in cmd

    def test_omits_baseline_metrics_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        metrics = tmp_path / "missing.json"
        seen: dict[str, object] = {}

        def fake_run(cmd, check, capture_output, text):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(script_mod.subprocess, "run", fake_run)
        script_mod.run_quality_gate(
            source_tag="s",
            target_tag="t",
            fixture=tmp_path / "fix.csv",
            baseline_metrics_json=metrics,
            min_probability=0.5,
            max_accuracy_regression=0.0,
            max_f1_regression=0.0,
        )
        assert "--baseline-metrics-json" not in seen["cmd"]

    def test_failed_gate_replays_stdout_and_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_run(cmd, check, capture_output, text):
            raise subprocess.CalledProcessError(2, cmd, output="gate stdout\n", stderr="gate stderr\n")

        monkeypatch.setattr(script_mod.subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            script_mod.run_quality_gate(
                source_tag="s",
                target_tag="t",
                fixture=tmp_path / "fix.csv",
                baseline_metrics_json=tmp_path / "missing.json",
                min_probability=0.3,
                max_accuracy_regression=0.0,
                max_f1_regression=0.0,
            )
        captured = capsys.readouterr()
        assert "gate stdout" in captured.out
        assert "gate stderr" in captured.err

    def test_failed_gate_with_empty_streams_still_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_run(cmd, check, capture_output, text):
            raise subprocess.CalledProcessError(1, cmd, output=None, stderr=None)

        monkeypatch.setattr(script_mod.subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            script_mod.run_quality_gate(
                source_tag="s",
                target_tag="t",
                fixture=tmp_path / "fix.csv",
                baseline_metrics_json=tmp_path / "missing.json",
                min_probability=0.3,
                max_accuracy_regression=0.0,
                max_f1_regression=0.0,
            )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestGetLegacyWarningMessages:
    def test_returns_empty_list_for_clean_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "clean.pkl"
        with path.open("wb") as handle:
            pickle.dump({"pipeline": True}, handle)
        assert script_mod.get_legacy_warning_messages(path) == []

    def test_corrupt_pickle_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not-a-pickle")
        with pytest.raises(subprocess.CalledProcessError):
            script_mod.get_legacy_warning_messages(path)


def _patch_main_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    source: Path,
    pipeline: object,
    catalog: Path,
    warning_messages: dict[str, list[str]] | None = None,
    quality_calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    seen: dict[str, object] = {"predictor": None, "quality": quality_calls if quality_calls is not None else []}

    monkeypatch.setattr(script_mod, "ensure_tag_downloaded", lambda tag: source)
    monkeypatch.setattr("lexnlp.ml.catalog.CATALOG", catalog)
    monkeypatch.setattr(script_mod, "load", lambda _fh: pipeline)

    class FakePredictor:
        def __init__(self, pipeline=None) -> None:
            seen["predictor"] = pipeline

    monkeypatch.setattr(
        "lexnlp.extract.en.contracts.predictors.ProbabilityPredictorIsContract",
        FakePredictor,
    )

    def fake_warnings(path: Path) -> list[str]:
        mapping = warning_messages or {}
        if path == source:
            return mapping.get("source", [])
        return mapping.get("candidate", [])

    monkeypatch.setattr(script_mod, "get_legacy_warning_messages", fake_warnings)

    def fake_gate(**kwargs: object) -> None:
        seen["quality"].append(kwargs)

    monkeypatch.setattr(script_mod, "run_quality_gate", fake_gate)
    return seen


class TestMain:
    def test_identical_tags_raise(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            script_mod.main(["--source-tag", "pipeline/same/1", "--target-tag", "pipeline/same/1"])

    def test_existing_destination_without_force_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        source = tmp_path / "model.cloudpickle"
        source.write_bytes(b"src")
        catalog = tmp_path / "catalog"
        dest_dir = catalog / "pipeline" / "dst" / "1"
        dest_dir.mkdir(parents=True)
        (dest_dir / source.name).write_bytes(b"existing")
        _patch_main_runtime(monkeypatch, tmp_path, source=source, pipeline={"p": 1}, catalog=catalog)
        with pytest.raises(FileExistsError, match="use --force"):
            script_mod.main(
                [
                    "--source-tag",
                    "pipeline/src/1",
                    "--target-tag",
                    "pipeline/dst/1",
                    "--skip-quality-gate",
                ]
            )

    def test_force_overwrites_and_skips_quality_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "model.cloudpickle"
        source.write_bytes(b"src")
        catalog = tmp_path / "catalog"
        dest_dir = catalog / "pipeline" / "dst" / "1"
        dest_dir.mkdir(parents=True)
        dest = dest_dir / source.name
        dest.write_bytes(b"old")
        pipeline = {"kind": "pipeline", "n": 7}
        meta = tmp_path / "out" / "meta.json"
        seen = _patch_main_runtime(
            monkeypatch,
            tmp_path,
            source=source,
            pipeline=pipeline,
            catalog=catalog,
            warning_messages={"source": ["old warn"], "candidate": ["old warn"]},
        )
        rc = script_mod.main(
            [
                "--source-tag",
                "pipeline/src/1",
                "--target-tag",
                "pipeline/dst/1",
                "--force",
                "--skip-quality-gate",
                "--output-metadata-json",
                str(meta),
                "--fixture",
                str(tmp_path / "fix.csv"),
                "--min-probability",
                "0.4",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        assert seen["predictor"] == pipeline
        assert seen["quality"] == []
        with dest.open("rb") as handle:
            assert pickle.load(handle) == pipeline
        payload = json.loads(meta.read_text(encoding="utf-8"))
        assert payload["source_tag"] == "pipeline/src/1"
        assert payload["target_tag"] == "pipeline/dst/1"
        assert payload["source_model_path"] == str(source)
        assert payload["target_model_path"] == str(dest)
        assert payload["fixture"] == str(tmp_path / "fix.csv")
        assert payload["min_probability"] == pytest.approx(0.4)
        assert "python" in payload["runtime"]
        assert "scikit_learn" in payload["runtime"]
        assert payload["runtime"]["lexnlp"]
        assert "created_at_utc" in payload
        assert f"re-export: wrote model to {dest}" in captured.out
        assert f"re-export: wrote metadata to {meta}" in captured.out
        assert "legacy sklearn warnings (source=1, candidate=1)" in captured.out
        assert "skipping quality gate by request" in captured.out

    def test_runs_quality_gate_and_writes_default_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "model.cloudpickle"
        source.write_bytes(b"src")
        catalog = tmp_path / "catalog"
        pipeline = {"kind": "pipeline"}
        seen = _patch_main_runtime(
            monkeypatch,
            tmp_path,
            source=source,
            pipeline=pipeline,
            catalog=catalog,
        )
        monkeypatch.chdir(tmp_path)
        rc = script_mod.main(
            [
                "--source-tag",
                "pipeline/src/1",
                "--target-tag",
                "pipeline/dst/2",
                "--fixture",
                str(tmp_path / "fix.csv"),
                "--baseline-metrics-json",
                str(tmp_path / "metrics.json"),
                "--max-accuracy-regression",
                "0.01",
                "--max-f1-regression",
                "0.02",
                "--min-probability",
                "0.35",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 0
        dest = catalog / "pipeline/dst/2" / source.name
        assert dest.exists()
        default_meta = tmp_path / "artifacts" / "model_reexports" / "pipeline__dst__2.metadata.json"
        assert default_meta.exists()
        assert seen["quality"] == [
            {
                "source_tag": "pipeline/src/1",
                "target_tag": "pipeline/dst/2",
                "fixture": tmp_path / "fix.csv",
                "baseline_metrics_json": tmp_path / "metrics.json",
                "min_probability": 0.35,
                "max_accuracy_regression": 0.01,
                "max_f1_regression": 0.02,
            }
        ]
        assert "quality gate passed" in captured.out

    def test_warning_regression_prints_candidate_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "model.cloudpickle"
        source.write_bytes(b"src")
        catalog = tmp_path / "catalog"
        _patch_main_runtime(
            monkeypatch,
            tmp_path,
            source=source,
            pipeline={"p": 1},
            catalog=catalog,
            warning_messages={"source": [], "candidate": ["Trying to unpickle estimator Foo"]},
        )
        rc = script_mod.main(
            [
                "--source-tag",
                "pipeline/src/1",
                "--target-tag",
                "pipeline/dst/1",
                "--skip-quality-gate",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "legacy warning regression exceeds threshold (1 > 0)" in captured.out
        assert "candidate legacy warnings:" in captured.out
        assert "Trying to unpickle estimator Foo" in captured.out

    def test_warning_regression_without_candidate_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "model.cloudpickle"
        source.write_bytes(b"src")
        catalog = tmp_path / "catalog"
        _patch_main_runtime(
            monkeypatch,
            tmp_path,
            source=source,
            pipeline={"p": 1},
            catalog=catalog,
            warning_messages={"source": [], "candidate": []},
        )
        rc = script_mod.main(
            [
                "--source-tag",
                "pipeline/src/1",
                "--target-tag",
                "pipeline/dst/1",
                "--skip-quality-gate",
                "--max-legacy-warning-regression",
                "-1",
            ]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "legacy warning regression exceeds threshold (0 > -1)" in captured.out
        assert "candidate legacy warnings:" not in captured.out


class TestMainGuard:
    """Exercise the ``if __name__ == "__main__"`` guard (line 301)."""

    def test_guard_identical_tags_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "reexport_contract_model.py",
                "--source-tag",
                "pipeline/same/1",
                "--target-tag",
                "pipeline/same/1",
            ],
        )
        with pytest.raises(ValueError, match="must differ"):
            runpy.run_path(str(Path(script_mod.__file__)), run_name="__main__")
