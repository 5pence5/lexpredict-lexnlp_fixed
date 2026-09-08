"""Coverage tests for scripts/bootstrap_assets.py.

Only network calls, subprocess-adjacent downloads and large model loads are
mocked; all argument parsing, file handling, zip extraction and task
wiring run for real.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import bootstrap_assets


class TestResolveTags:
    def test_contract_model_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        assert bootstrap_assets.resolve_contract_model_tag() == "pipeline/is-contract/0.2"

    def test_contract_model_primary_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", "pipeline/custom/1.0")
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/other/1.0")
        assert bootstrap_assets.resolve_contract_model_tag() == "pipeline/custom/1.0"

    def test_contract_model_legacy_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setenv("LEXNLP_IS_CONTRACT_MODEL_TAG", "pipeline/legacy/0.1")
        assert bootstrap_assets.resolve_contract_model_tag() == "pipeline/legacy/0.1"

    def test_contract_type_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG", raising=False)
        assert bootstrap_assets.resolve_contract_type_model_tag() == "pipeline/contract-type/0.2-runtime"

    def test_contract_type_env_override_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG", "  pipeline/custom-type/3.0  ")
        assert bootstrap_assets.resolve_contract_type_model_tag() == "pipeline/custom-type/3.0"


class TestConfigureLogging:
    def test_verbose_uses_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.update(kwargs) or None)
        bootstrap_assets.configure_logging(True)
        assert calls["level"] == logging.DEBUG
        assert calls["format"] == "[bootstrap][%(levelname)s] %(message)s"

    def test_quiet_uses_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, object] = {}
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.update(kwargs) or None)
        bootstrap_assets.configure_logging(False)
        assert calls["level"] == logging.INFO


class TestParseArgs:
    def test_each_task_flag_parsed(self) -> None:
        for flag in ("--nltk", "--contract-model", "--contract-type-model", "--stanford", "--tika", "--all"):
            args = bootstrap_assets.parse_args([flag])
            key = flag.lstrip("-").replace("-", "_")
            assert getattr(args, key) is True

    def test_defaults(self) -> None:
        args = bootstrap_assets.parse_args(["--nltk"])
        assert args.stanford_dir == str(bootstrap_assets.DEFAULT_STANFORD_DIR)
        assert args.tika_dir == str(bootstrap_assets.DEFAULT_TIKA_DIR)
        assert args.dry_run is False
        assert args.force is False
        assert args.timeout == 60
        assert args.verbose is False

    def test_explicit_values(self, tmp_path: Path) -> None:
        args = bootstrap_assets.parse_args(
            [
                "--nltk",
                "--dry-run",
                "--force",
                "--verbose",
                "--stanford-dir",
                str(tmp_path / "stan"),
                "--tika-dir",
                str(tmp_path / "tika"),
                "--timeout",
                "7",
            ]
        )
        assert args.dry_run is True
        assert args.force is True
        assert args.verbose is True
        assert args.stanford_dir == str(tmp_path / "stan")
        assert args.tika_dir == str(tmp_path / "tika")
        assert args.timeout == 7

    def test_no_task_selected_errors(self) -> None:
        with pytest.raises(SystemExit):
            bootstrap_assets.parse_args([])

    def test_non_positive_timeout_errors(self) -> None:
        with pytest.raises(SystemExit):
            bootstrap_assets.parse_args(["--nltk", "--timeout", "0"])
        with pytest.raises(SystemExit):
            bootstrap_assets.parse_args(["--nltk", "--timeout", "-3"])


class TestEnsureDirectory:
    def test_dry_run_creates_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "nested"
        bootstrap_assets.ensure_directory(target, dry_run=True)
        assert not target.exists()

    def test_real_run_creates_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "nested"
        bootstrap_assets.ensure_directory(target, dry_run=False)
        assert target.is_dir()


def _mock_urlopen(monkeypatch: pytest.MonkeyPatch, chunks: list) -> MagicMock:
    response = MagicMock()
    response.read.side_effect = chunks
    context = MagicMock()
    context.__enter__.return_value = response
    context.__exit__.return_value = False
    mock_open = MagicMock(return_value=context)
    monkeypatch.setattr(bootstrap_assets, "urlopen", mock_open)
    return mock_open


class TestDownloadFile:
    def test_successful_download_writes_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_open = _mock_urlopen(monkeypatch, [b"chunk1", b"chunk2", b""])
        destination = tmp_path / "sub" / "file.zip"
        bootstrap_assets.download_file(
            "https://example.com/file.zip", destination, force=False, dry_run=False, timeout=5
        )
        assert destination.read_bytes() == b"chunk1chunk2"
        assert not destination.with_name(destination.name + ".part").exists()
        assert mock_open.called

    def test_stale_part_file_removed_first(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_urlopen(monkeypatch, [b"fresh", b""])
        destination = tmp_path / "file.zip"
        stale = destination.with_name(destination.name + ".part")
        stale.write_bytes(b"stale-part-contents")
        bootstrap_assets.download_file(
            "https://example.com/file.zip", destination, force=True, dry_run=False, timeout=5
        )
        assert destination.read_bytes() == b"fresh"
        assert not stale.exists()

    def test_failure_cleans_part_and_reraises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_urlopen(monkeypatch, [b"partial", OSError("connection reset")])
        destination = tmp_path / "file.zip"
        with pytest.raises(OSError, match="connection reset"):
            bootstrap_assets.download_file(
                "https://example.com/file.zip", destination, force=True, dry_run=False, timeout=5
            )
        assert not destination.exists()
        assert not destination.with_name(destination.name + ".part").exists()

    def test_dry_run_writes_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_open = _mock_urlopen(monkeypatch, [b"data", b""])
        destination = tmp_path / "file.zip"
        bootstrap_assets.download_file("https://example.com/file.zip", destination, force=True, dry_run=True, timeout=5)
        assert not destination.exists()
        mock_open.assert_not_called()


class TestDownloadMany:
    def test_delegates_per_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, Path]] = []

        def fake_download(url: str, destination: Path, *, force: bool, dry_run: bool, timeout: int) -> None:
            calls.append((url, destination))
            assert force is True
            assert dry_run is False
            assert timeout == 11

        monkeypatch.setattr(bootstrap_assets, "download_file", fake_download)
        destination_dir = tmp_path / "tika"
        bootstrap_assets.download_many(
            [("a.jar", "https://example.com/a.jar"), ("b.jar", "https://example.com/b.jar")],
            destination_dir,
            force=True,
            dry_run=False,
            timeout=11,
        )
        assert calls == [
            ("https://example.com/a.jar", destination_dir / "a.jar"),
            ("https://example.com/b.jar", destination_dir / "b.jar"),
        ]


class TestExtractZipDirectories:
    def test_directory_members_created(self, tmp_path: Path) -> None:
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("models/", b"")
            archive.writestr("models/english.tagger", b"tagger-bytes")
        archive_path = tmp_path / "archive.zip"
        archive_path.write_bytes(buf.getvalue())
        destination = tmp_path / "out"
        bootstrap_assets.extract_zip(archive_path, destination, dry_run=False)
        assert (destination / "models").is_dir()
        assert (destination / "models" / "english.tagger").read_bytes() == b"tagger-bytes"


def _touch_required_files(destination_dir: Path, *, archives: tuple[int, ...] = (0, 1)) -> None:
    for index in archives:
        for rel_path in bootstrap_assets.STANFORD_DOWNLOADS[index][2]:
            candidate = destination_dir / rel_path
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"present")


class TestBootstrapStanfordAssets:
    def test_dry_run_calls_helpers_without_writes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        downloads: list[str] = []
        extracts: list[Path] = []
        monkeypatch.setattr(bootstrap_assets, "download_file", lambda url, dest, **kwargs: downloads.append(url))
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: extracts.append(archive))
        bootstrap_assets.bootstrap_stanford_assets(tmp_path / "stan", force=False, dry_run=True, timeout=5)
        assert len(downloads) == len(bootstrap_assets.STANFORD_DOWNLOADS)
        assert len(extracts) == len(bootstrap_assets.STANFORD_DOWNLOADS)
        assert not (tmp_path / "stan").exists()

    def test_present_assets_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        destination = tmp_path / "stan"
        _touch_required_files(destination)
        downloads: list[str] = []
        monkeypatch.setattr(bootstrap_assets, "download_file", lambda url, dest, **kwargs: downloads.append(url))
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: None)
        bootstrap_assets.bootstrap_stanford_assets(destination, force=False, dry_run=False, timeout=5)
        assert downloads == []

    def test_force_redownloads_present_assets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        destination = tmp_path / "stan"
        _touch_required_files(destination)
        downloads: list[str] = []
        monkeypatch.setattr(bootstrap_assets, "download_file", lambda url, dest, **kwargs: downloads.append(url))
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: None)
        bootstrap_assets.bootstrap_stanford_assets(destination, force=True, dry_run=False, timeout=5)
        assert len(downloads) == len(bootstrap_assets.STANFORD_DOWNLOADS)

    def test_partial_assets_download_only_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        destination = tmp_path / "stan"
        _touch_required_files(destination, archives=(0,))
        downloads: list[str] = []
        monkeypatch.setattr(bootstrap_assets, "download_file", lambda url, dest, **kwargs: downloads.append(url))
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: None)
        with pytest.raises(RuntimeError, match="Missing required Stanford files"):
            bootstrap_assets.bootstrap_stanford_assets(destination, force=False, dry_run=False, timeout=5)
        assert downloads == [bootstrap_assets.STANFORD_DOWNLOADS[1][1]]

    def test_missing_after_download_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        destination = tmp_path / "stan"
        monkeypatch.setattr(bootstrap_assets, "download_file", lambda url, dest, **kwargs: None)
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: None)
        with pytest.raises(RuntimeError, match="Missing required Stanford files") as exc_info:
            bootstrap_assets.bootstrap_stanford_assets(destination, force=False, dry_run=False, timeout=5)
        assert "stanford-postagger.jar" in str(exc_info.value)
        assert "stanford-ner.jar" in str(exc_info.value)

    def test_successful_download_passes_verification(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        destination = tmp_path / "stan"

        def fake_download(url: str, archive: Path, **kwargs) -> None:
            _touch_required_files(destination)

        monkeypatch.setattr(bootstrap_assets, "download_file", fake_download)
        monkeypatch.setattr(bootstrap_assets, "extract_zip", lambda archive, dest, **kwargs: None)
        bootstrap_assets.bootstrap_stanford_assets(destination, force=False, dry_run=False, timeout=5)
        assert (destination / bootstrap_assets.STANFORD_DOWNLOADS[0][2][0]).exists()


def _make_fake_nltk(monkeypatch: pytest.MonkeyPatch):
    import types

    module = types.ModuleType("nltk")
    module.download = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk", module)
    return module


class TestBootstrapNltk:
    def test_dry_run_downloads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _make_fake_nltk(monkeypatch)
        bootstrap_assets.bootstrap_nltk(dry_run=True)
        module.download.assert_not_called()

    def test_missing_nltk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "nltk", None)
        with pytest.raises(RuntimeError, match="nltk is required"):
            bootstrap_assets.bootstrap_nltk(dry_run=False)

    def test_required_and_optional_downloaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = _make_fake_nltk(monkeypatch)
        bootstrap_assets.bootstrap_nltk(dry_run=False)
        downloaded = [call.args[0] for call in module.download.call_args_list]
        for resource in bootstrap_assets.NLTK_RESOURCES:
            assert resource in downloaded
        for resource in bootstrap_assets.OPTIONAL_NLTK_RESOURCES:
            assert resource in downloaded
        for call in module.download.call_args_list:
            assert call.kwargs == {"quiet": True, "raise_on_error": True}

    def test_optional_failure_warns_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        module = _make_fake_nltk(monkeypatch)

        def flaky(resource: str, **kwargs) -> bool:
            if resource == bootstrap_assets.OPTIONAL_NLTK_RESOURCES[0]:
                raise OSError("unavailable")
            return True

        module.download.side_effect = flaky  # type: ignore[attr-defined]
        with caplog.at_level(logging.WARNING, logger="lexnlp.bootstrap"):
            bootstrap_assets.bootstrap_nltk(dry_run=False)
        assert module.download.call_count == len(bootstrap_assets.NLTK_RESOURCES) + len(
            bootstrap_assets.OPTIONAL_NLTK_RESOURCES
        )
        assert "Optional NLTK resource unavailable" in caplog.text


class _TagDownloadError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class _StubIsContractPredictor:
    def __init__(self, pipeline=None) -> None:
        self.pipeline = pipeline


class TestBootstrapContractModel:
    def test_dry_run_downloads_nothing(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append(tag),
        )
        with caplog.at_level(logging.INFO, logger="lexnlp.bootstrap"):
            bootstrap_assets.bootstrap_contract_model(dry_run=True, tag="pipeline/is-contract/0.2")
        assert calls == []
        assert "DRY RUN" in caplog.text

    def test_missing_downloader_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lexnlp.ml.catalog.download", None)
        with pytest.raises(RuntimeError, match="Unable to import LexNLP catalog downloader"):
            bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/is-contract/0.2")

    def test_successful_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, bool]] = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append((tag, prompt_user)),
        )
        bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/is-contract/0.2")
        assert calls == [("pipeline/is-contract/0.2", False)]

    def test_explicit_tag_never_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_CONTRACT_MODEL_TAG", "pipeline/custom/9.9")
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: (_ for _ in ()).throw(_TagDownloadError(404)),
        )
        with pytest.raises(_TagDownloadError):
            bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/custom/9.9")

    def test_non_404_error_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: (_ for _ in ()).throw(_TagDownloadError(500)),
        )
        with pytest.raises(_TagDownloadError):
            bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/is-contract/0.2")

    def test_404_for_other_tag_reraises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: (_ for _ in ()).throw(_TagDownloadError(404)),
        )
        with pytest.raises(_TagDownloadError):
            bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/other/1.0")

    def _install_fallback_mocks(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sentinel: object):
        import lexnlp.ml.catalog as catalog

        legacy_path = tmp_path / "legacy" / "model.cloudpickle"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_bytes(b"legacy-artifact")
        monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog")
        monkeypatch.setattr(catalog, "get_path_from_catalog", lambda tag: legacy_path)
        monkeypatch.setattr("cloudpickle.load", lambda handle: sentinel)
        monkeypatch.setattr(
            "lexnlp.extract.en.contracts.predictors.ProbabilityPredictorIsContract",
            _StubIsContractPredictor,
        )
        downloads: list[str] = []

        def fake_download(tag: str, prompt_user: bool = False) -> None:
            downloads.append(tag)
            if tag == "pipeline/is-contract/0.2":
                raise _TagDownloadError(404)

        monkeypatch.setattr("lexnlp.ml.catalog.download.download_github_release", fake_download)
        monkeypatch.delenv("LEXNLP_CONTRACT_MODEL_TAG", raising=False)
        monkeypatch.delenv("LEXNLP_IS_CONTRACT_MODEL_TAG", raising=False)
        return downloads

    def test_404_falls_back_and_reexports(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import pickle

        sentinel = {"marker": "legacy-pipeline"}
        downloads = self._install_fallback_mocks(monkeypatch, tmp_path, sentinel)
        bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/is-contract/0.2")
        assert downloads == ["pipeline/is-contract/0.2", "pipeline/is-contract/0.1"]
        destination = tmp_path / "catalog" / "pipeline/is-contract/0.2" / "model.cloudpickle"
        assert destination.exists()
        with destination.open("rb") as handle:
            assert pickle.load(handle) == sentinel

    def test_reexport_failure_continues_with_legacy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        downloads = self._install_fallback_mocks(monkeypatch, tmp_path, object())

        class _BrokenPredictor:
            def __init__(self, pipeline=None) -> None:
                raise RuntimeError("incompatible artifact")

        monkeypatch.setattr(
            "lexnlp.extract.en.contracts.predictors.ProbabilityPredictorIsContract",
            _BrokenPredictor,
        )
        with caplog.at_level(logging.ERROR, logger="lexnlp.bootstrap"):
            bootstrap_assets.bootstrap_contract_model(dry_run=False, tag="pipeline/is-contract/0.2")
        assert downloads == ["pipeline/is-contract/0.2", "pipeline/is-contract/0.1"]
        assert "Failed to generate contract model" in caplog.text


class TestBootstrapContractTypeModel:
    def test_dry_run_builds_nothing(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        import types

        module = types.ModuleType("lexnlp.extract.en.contracts.runtime_model")
        module.ensure_runtime_contract_type_model = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lexnlp.extract.en.contracts.runtime_model", module)
        with caplog.at_level(logging.INFO, logger="lexnlp.bootstrap"):
            bootstrap_assets.bootstrap_contract_type_model(dry_run=True, tag="pipeline/contract-type/x")
        module.ensure_runtime_contract_type_model.assert_not_called()
        assert "DRY RUN" in caplog.text

    def test_missing_builder_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lexnlp.extract.en.contracts.runtime_model", None)
        with pytest.raises(RuntimeError, match="Unable to import contract-type runtime model builder"):
            bootstrap_assets.bootstrap_contract_type_model(dry_run=False, tag="pipeline/contract-type/x")

    def test_delegates_to_runtime_builder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import types

        module = types.ModuleType("lexnlp.extract.en.contracts.runtime_model")
        module.ensure_runtime_contract_type_model = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "lexnlp.extract.en.contracts.runtime_model", module)
        bootstrap_assets.bootstrap_contract_type_model(dry_run=False, tag="pipeline/contract-type/x")
        module.ensure_runtime_contract_type_model.assert_called_once_with(target_tag="pipeline/contract-type/x")


def _make_task_args(**overrides) -> argparse.Namespace:
    base = {
        "nltk": False,
        "contract_model": False,
        "contract_type_model": False,
        "stanford": False,
        "tika": False,
        "all": False,
        "stanford_dir": str(bootstrap_assets.DEFAULT_STANFORD_DIR),
        "tika_dir": str(bootstrap_assets.DEFAULT_TIKA_DIR),
        "dry_run": True,
        "force": False,
        "timeout": 30,
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRunSelectedTasks:
    def test_stanford_wiring(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[Path, bool, bool, int]] = []

        def fake_stanford(destination: Path, *, force: bool, dry_run: bool, timeout: int) -> None:
            calls.append((destination, force, dry_run, timeout))

        monkeypatch.setattr(bootstrap_assets, "bootstrap_stanford_assets", fake_stanford)
        stanford_dir = tmp_path / "stanford"
        args = _make_task_args(stanford=True, stanford_dir=str(stanford_dir), force=True, timeout=9)
        bootstrap_assets.run_selected_tasks(args)
        assert calls == [(stanford_dir.resolve(), True, True, 9)]

    def test_tika_wiring(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple] = []

        def fake_download_many(downloads, destination: Path, **kwargs) -> None:
            calls.append((list(downloads), destination, kwargs))

        monkeypatch.setattr(bootstrap_assets, "download_many", fake_download_many)
        tika_dir = tmp_path / "tika"
        args = _make_task_args(tika=True, tika_dir=str(tika_dir))
        bootstrap_assets.run_selected_tasks(args)
        assert len(calls) == 1
        downloads, destination, kwargs = calls[0]
        assert downloads == list(bootstrap_assets.TIKA_DOWNLOADS)
        assert destination == tika_dir.resolve()
        assert kwargs == {"force": False, "dry_run": True, "timeout": 30}

    def test_nltk_wiring(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bool] = []
        monkeypatch.setattr(bootstrap_assets, "bootstrap_nltk", lambda *, dry_run: calls.append(dry_run))
        bootstrap_assets.run_selected_tasks(_make_task_args(nltk=True, dry_run=False))
        assert calls == [False]

    def test_all_runs_every_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        monkeypatch.setattr(bootstrap_assets, "bootstrap_nltk", lambda **kwargs: ran.append("nltk"))
        monkeypatch.setattr(bootstrap_assets, "bootstrap_contract_model", lambda **kwargs: ran.append("contract-model"))
        monkeypatch.setattr(
            bootstrap_assets,
            "bootstrap_contract_type_model",
            lambda **kwargs: ran.append("contract-type-model"),
        )
        monkeypatch.setattr(
            bootstrap_assets, "bootstrap_stanford_assets", lambda *args, **kwargs: ran.append("stanford")
        )
        monkeypatch.setattr(bootstrap_assets, "download_many", lambda *args, **kwargs: ran.append("tika"))
        bootstrap_assets.run_selected_tasks(_make_task_args(all=True))
        assert ran == ["nltk", "contract-model", "contract-type-model", "stanford", "tika"]

    def test_failures_aggregated_and_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []

        def boom(**kwargs) -> None:
            ran.append("nltk")
            raise RuntimeError("boom")

        monkeypatch.setattr(bootstrap_assets, "bootstrap_nltk", boom)
        monkeypatch.setattr(bootstrap_assets, "download_many", lambda *args, **kwargs: ran.append("tika"))
        with pytest.raises(bootstrap_assets.BootstrapError, match="Failed tasks: nltk"):
            bootstrap_assets.run_selected_tasks(_make_task_args(nltk=True, tika=True))
        assert ran == ["nltk", "tika"]

    def test_multiple_failures_all_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom_nltk(**kwargs) -> None:
            raise RuntimeError("nltk boom")

        def boom_tika(*args, **kwargs) -> None:
            raise RuntimeError("tika boom")

        monkeypatch.setattr(bootstrap_assets, "bootstrap_nltk", boom_nltk)
        monkeypatch.setattr(bootstrap_assets, "download_many", boom_tika)
        with pytest.raises(bootstrap_assets.BootstrapError, match="Failed tasks: nltk, tika"):
            bootstrap_assets.run_selected_tasks(_make_task_args(nltk=True, tika=True))


class TestMain:
    def test_dry_run_nltk_succeeds(self) -> None:
        assert bootstrap_assets.main(["--nltk", "--dry-run"]) == 0

    def test_bootstrap_error_returns_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(args) -> None:
            raise bootstrap_assets.BootstrapError("Failed tasks: nltk")

        monkeypatch.setattr(bootstrap_assets, "run_selected_tasks", boom)
        assert bootstrap_assets.main(["--nltk"]) == 1

    def test_run_as_main_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import runpy

        monkeypatch.setattr(sys, "argv", ["bootstrap_assets.py"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPTS_DIR / "bootstrap_assets.py"), run_name="__main__")
