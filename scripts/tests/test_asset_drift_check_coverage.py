"""Coverage tests for scripts/asset_drift_check.py."""

from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts import asset_drift_check as drift


def _write_manifest(path: Path, assets: list) -> None:
    path.write_text(json.dumps({"assets": assets}), encoding="utf-8")


def _write_model(tmp_path: Path, name: str = "model.pkl", data: bytes = b"payload") -> Path:
    model = tmp_path / name
    model.write_bytes(data)
    return model


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestIterAssets:
    def test_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="must be objects"):
            list(drift.iter_assets({"assets": ["not-a-dict"]}))

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing key=sha256"):
            list(drift.iter_assets({"assets": [{"tag": "t", "filename": "f"}]}))

    def test_valid_assets_are_yielded(self):
        assets = [{"tag": "t", "filename": "f", "sha256": "s"}]
        assert list(drift.iter_assets({"assets": assets})) == assets


class TestEnsureTagDownloaded:
    def test_catalog_hit_skips_download(self, monkeypatch, tmp_path):
        cached = _write_model(tmp_path)
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: cached)
        calls: list = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append((tag, prompt_user)),
        )
        assert drift.ensure_tag_downloaded("pipeline/x/0.1") == cached
        assert calls == []

    def test_catalog_miss_downloads_then_returns(self, monkeypatch, tmp_path):
        cached = _write_model(tmp_path)
        state = {"tries": 0}

        def fake_get(tag: str):
            state["tries"] += 1
            if state["tries"] == 1:
                raise FileNotFoundError(tag)
            return cached

        seen: dict = {}

        def fake_download(tag: str, prompt_user: bool = False):
            seen["tag"] = tag
            seen["prompt_user"] = prompt_user

        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", fake_get)
        monkeypatch.setattr("lexnlp.ml.catalog.download.download_github_release", fake_download)
        assert drift.ensure_tag_downloaded("pipeline/x/0.1") == cached
        assert seen == {"tag": "pipeline/x/0.1", "prompt_user": False}

    def test_unexpected_error_propagates_without_download(self, monkeypatch, tmp_path):
        def fake_get(tag: str):
            raise OSError("disk gone")

        calls: list = []
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", fake_get)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append(tag),
        )
        with pytest.raises(OSError, match="disk gone"):
            drift.ensure_tag_downloaded("pipeline/x/0.1")
        assert calls == []


class TestParseArgs:
    def test_defaults(self):
        args = drift.parse_args([])
        assert args.manifest == drift.DEFAULT_MANIFEST
        assert args.download_missing is False
        assert args.force_download is False


class TestMainDownloadBranches:
    def test_force_download_calls_downloader(self, monkeypatch, tmp_path, capsys):
        data = b"force content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [{"tag": "pipeline/x/0.1", "filename": "model.pkl", "sha256": _sha(data)}],
        )
        calls: list = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append((tag, prompt_user)),
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest), "--force-download"])
        assert rc == 0
        assert calls == [("pipeline/x/0.1", False)]
        assert "asset-drift: OK pipeline/x/0.1" in capsys.readouterr().out

    def test_download_missing_uses_ensure_helper(self, monkeypatch, tmp_path, capsys):
        data = b"helper content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [{"tag": "pipeline/x/0.1", "filename": "model.pkl", "sha256": _sha(data)}],
        )
        seen: list = []

        def fake_ensure(tag: str):
            seen.append(tag)
            return model

        monkeypatch.setattr(drift, "ensure_tag_downloaded", fake_ensure)
        rc = drift.main(["--manifest", str(manifest), "--download-missing"])
        assert rc == 0
        assert seen == ["pipeline/x/0.1"]
        assert "asset-drift: OK pipeline/x/0.1" in capsys.readouterr().out

    def test_default_path_uses_catalog_directly(self, monkeypatch, tmp_path, capsys):
        data = b"direct content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [{"tag": "pipeline/x/0.1", "filename": "model.pkl", "sha256": _sha(data)}],
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest)])
        assert rc == 0
        assert "asset-drift: OK pipeline/x/0.1" in capsys.readouterr().out


class TestMainFilenameAndSize:
    def test_filename_mismatch_fails(self, monkeypatch, tmp_path, capsys):
        data = b"named content"
        model = _write_model(tmp_path, name="other.pkl", data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [{"tag": "pipeline/x/0.1", "filename": "model.pkl", "sha256": _sha(data)}],
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest)])
        assert rc == 1
        assert "unexpected filename" in capsys.readouterr().err

    def test_invalid_size_fails(self, monkeypatch, tmp_path, capsys):
        data = b"sized content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [
                {
                    "tag": "pipeline/x/0.1",
                    "filename": "model.pkl",
                    "sha256": _sha(data),
                    "size": "not-a-number",
                }
            ],
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest)])
        assert rc == 1
        assert "invalid manifest size" in capsys.readouterr().err

    def test_size_mismatch_fails(self, monkeypatch, tmp_path, capsys):
        data = b"sized content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [
                {
                    "tag": "pipeline/x/0.1",
                    "filename": "model.pkl",
                    "sha256": _sha(data),
                    "size": model.stat().st_size + 100,
                }
            ],
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest)])
        assert rc == 1
        assert "size mismatch" in capsys.readouterr().err

    def test_matching_size_passes(self, monkeypatch, tmp_path, capsys):
        data = b"sized content"
        model = _write_model(tmp_path, data=data)
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            [
                {
                    "tag": "pipeline/x/0.1",
                    "filename": "model.pkl",
                    "sha256": _sha(data),
                    "size": model.stat().st_size,
                }
            ],
        )
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: model)
        rc = drift.main(["--manifest", str(manifest)])
        assert rc == 0
        assert "asset-drift: OK pipeline/x/0.1" in capsys.readouterr().out


class TestMainGuard:
    """Exercise the ``if __name__ == "__main__"`` guard (line 145)."""

    def test_guard_missing_tag_exits_one(self, tmp_path, monkeypatch, capsys):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "assets": [
                        {
                            "tag": "pipeline/does-not-exist/0.0",
                            "filename": "model.pkl",
                            "sha256": "0" * 64,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["asset_drift_check.py", "--manifest", str(manifest)],
        )
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(Path(drift.__file__)), run_name="__main__")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "asset-drift: ERROR pipeline/does-not-exist/0.0" in captured.err
        assert "missing/unreadable" in captured.err
        assert "FileNotFoundError" in captured.err
