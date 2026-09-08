"""Coverage tests for scripts/train_contract_type_model.py ``__main__`` guard."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import lexnlp.extract.en.contracts.runtime_model as runtime_model


def _patch_runtime_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    corpus_path: Path,
    texts: list[str],
    labels: list[str],
    destination: Path,
    wrote: bool,
) -> None:
    """Stub the network/model-load boundary so the script runs hermetically."""
    pipeline = MagicMock()
    pipeline.predict.side_effect = lambda X: ["A"] * len(X)
    monkeypatch.setattr(runtime_model, "ensure_tag_downloaded", lambda tag: corpus_path)
    monkeypatch.setattr(
        runtime_model,
        "collect_contract_type_samples",
        lambda _archive, *, max_docs_per_label, head_character_n: (texts, labels, {"A": 5, "B": 5}),
    )
    monkeypatch.setattr(
        runtime_model,
        "train_contract_type_pipeline",
        lambda _texts, _labels, *, random_state: pipeline,
    )
    monkeypatch.setattr(
        runtime_model,
        "write_pipeline_to_catalog",
        lambda *, pipeline, target_tag, force: (destination, wrote),
    )


class TestMainGuard:
    """Exercise the ``if __name__ == "__main__"`` guard (line 177)."""

    def test_guard_writes_report_and_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        corpus = tmp_path / "corpus.tar.xz"
        corpus.write_bytes(b"")
        destination = tmp_path / "model.cloudpickle"
        destination.write_bytes(b"model")
        texts = [f"document {i}" for i in range(10)]
        labels = (["A"] * 5) + (["B"] * 5)
        output_json = tmp_path / "report.json"
        _patch_runtime_model(
            monkeypatch,
            corpus_path=corpus,
            texts=texts,
            labels=labels,
            destination=destination,
            wrote=True,
        )

        script = _SCRIPTS_DIR / "train_contract_type_model.py"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_contract_type_model.py",
                "--target-tag",
                "pipeline/test/0.1",
                "--output-json",
                str(output_json),
                "--force",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script), run_name="__main__")

        assert exc_info.value.code == 0
        report = json.loads(output_json.read_text(encoding="utf-8"))
        assert report["wrote_artifact"] is True
        assert report["target_tag"] == "pipeline/test/0.1"
        assert report["target_model_path"] == str(destination)
        assert report["dataset"]["samples_total"] == 10
        assert report["dataset"]["samples_train"] + report["dataset"]["samples_validation"] == 10
        assert report["dataset"]["stratified_split"] is True
        assert set(report["validation_metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}
        stdout_report = json.loads(capsys.readouterr().out)
        assert stdout_report == report

    def test_guard_propagates_failure_exit_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        corpus = tmp_path / "corpus.tar.xz"
        corpus.write_bytes(b"")
        destination = tmp_path / "model.cloudpickle"
        destination.write_bytes(b"existing")
        texts = [f"document {i}" for i in range(10)]
        labels = (["A"] * 5) + (["B"] * 5)
        output_json = tmp_path / "report.json"
        _patch_runtime_model(
            monkeypatch,
            corpus_path=corpus,
            texts=texts,
            labels=labels,
            destination=destination,
            wrote=False,
        )

        script = _SCRIPTS_DIR / "train_contract_type_model.py"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_contract_type_model.py",
                "--target-tag",
                "pipeline/test/0.1",
                "--output-json",
                str(output_json),
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(script), run_name="__main__")

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        report = json.loads(output_json.read_text(encoding="utf-8"))
        assert report["wrote_artifact"] is False
        assert json.loads(captured.out) == report
        assert "already exists" in captured.err
