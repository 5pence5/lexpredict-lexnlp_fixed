"""Coverage tests for lexnlp.ml.catalog.download trust, session, and verify paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from requests import Session
from requests.adapters import HTTPAdapter

from lexnlp.ml.catalog import download


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    *,
    assets: list[dict],
    models_repo_slug: str | None = "reviewed/models",
    models_repo_base_url: str | None = None,
    schema_version: int = 1,
) -> Path:
    payload: dict = {"schema_version": schema_version, "assets": assets}
    if models_repo_base_url is not None:
        payload["models_repo_base_url"] = models_repo_base_url
    elif models_repo_slug is not None:
        payload["models_repo_slug"] = models_repo_slug
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _asset_entry(
    tag: str = "pipeline/example/1",
    filename: str = "model.bin",
    payload: bytes = b"reviewed",
    *,
    size: int | None = None,
    sha256: str | None = None,
) -> dict:
    return {
        "tag": tag,
        "filename": filename,
        "size": len(payload) if size is None else size,
        "sha256": _sha256(payload) if sha256 is None else sha256,
    }


class TestBuildRetrySession:
    def test_returns_session_with_http_and_https_adapters(self) -> None:
        session = download.build_retry_session(total_retries=2, backoff_factor=0.5)

        assert isinstance(session, Session)
        assert isinstance(session.adapters["http://"], HTTPAdapter)
        assert isinstance(session.adapters["https://"], HTTPAdapter)
        https_retry = session.adapters["https://"].max_retries
        assert https_retry.total == 2
        assert https_retry.backoff_factor == 0.5
        assert 429 in https_retry.status_forcelist
        assert 503 in https_retry.status_forcelist

    def test_process_session_is_lazy_singleton(self) -> None:
        original = download._SESSION
        download._SESSION = None
        try:
            first = download._session()
            second = download._session()
            assert first is second
            assert isinstance(first, Session)
        finally:
            if download._SESSION is not None:
                download._SESSION.close()
            download._SESSION = original


class TestAssetTrustTypes:
    def test_asset_trust_error_is_runtime_error(self) -> None:
        error = download.AssetTrustError("untrusted")
        assert isinstance(error, RuntimeError)
        assert str(error) == "untrusted"

    def test_missing_trusted_asset_error_from_manifest_get(self) -> None:
        manifest = download.AssetManifest(
            models_repo="https://api.github.com/repos/reviewed/models/releases/tags/",
            assets={},
        )
        with pytest.raises(download.MissingTrustedAssetError, match="not present") as caught:
            manifest.get("pipeline/missing/1")
        assert isinstance(caught.value, download.AssetTrustError)
        assert "pipeline/missing/1" in str(caught.value)
        assert download.DEFAULT_MANIFEST_RESOURCE in str(caught.value)

    def test_trusted_asset_fields_round_trip(self) -> None:
        payload = b"abc"
        trusted = download.TrustedAsset(
            tag="pipeline/example/1",
            filename="model.bin",
            size=len(payload),
            sha256=_sha256(payload),
        )
        assert trusted.tag == "pipeline/example/1"
        assert trusted.filename == "model.bin"
        assert trusted.size == 3
        assert trusted.sha256 == _sha256(payload)


class TestGithubTimeout:
    def test_missing_or_blank_env_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_GITHUB_TIMEOUT", raising=False)
        assert download._get_github_timeout_seconds() == download.DEFAULT_GITHUB_TIMEOUT_SECONDS
        monkeypatch.setenv("LEXNLP_GITHUB_TIMEOUT", "   ")
        assert download._get_github_timeout_seconds() == download.DEFAULT_GITHUB_TIMEOUT_SECONDS

    def test_invalid_and_non_positive_values_use_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_GITHUB_TIMEOUT", "not-a-float")
        assert download._get_github_timeout_seconds() == download.DEFAULT_GITHUB_TIMEOUT_SECONDS
        monkeypatch.setenv("LEXNLP_GITHUB_TIMEOUT", "0")
        assert download._get_github_timeout_seconds() == download.DEFAULT_GITHUB_TIMEOUT_SECONDS
        monkeypatch.setenv("LEXNLP_GITHUB_TIMEOUT", "-3.5")
        assert download._get_github_timeout_seconds() == download.DEFAULT_GITHUB_TIMEOUT_SECONDS

    def test_positive_timeout_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_GITHUB_TIMEOUT", "12.25")
        assert download._get_github_timeout_seconds() == 12.25


class TestNormaliseRepoUrl:
    def test_https_url_gains_trailing_slash(self) -> None:
        assert (
            download._normalise_repo_url(
                "https://api.github.com/repos/reviewed/models/releases/tags"
            )
            == "https://api.github.com/repos/reviewed/models/releases/tags/"
        )

    def test_rejects_non_https_and_credentials(self) -> None:
        with pytest.raises(download.AssetTrustError, match="absolute HTTPS URL"):
            download._normalise_repo_url("http://api.github.com/repos/reviewed/models/releases/tags/")
        with pytest.raises(download.AssetTrustError, match="absolute HTTPS URL"):
            download._normalise_repo_url(
                "https://user:pass@api.github.com/repos/reviewed/models/releases/tags/"
            )


class TestConfiguredModelsRepo:
    def test_legacy_assignment_without_trailing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_MODELS_REPO", raising=False)
        monkeypatch.delenv("LEXNLP_MODELS_REPO_SLUG", raising=False)
        monkeypatch.setattr(
            download,
            "MODELS_REPO",
            "https://api.github.com/repos/legacy/download-module/releases/tags",
        )
        assert (
            download._configured_models_repo()
            == "https://api.github.com/repos/legacy/download-module/releases/tags/"
        )

    def test_legacy_assignment_already_slashed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_MODELS_REPO", raising=False)
        monkeypatch.delenv("LEXNLP_MODELS_REPO_SLUG", raising=False)
        slashed = "https://api.github.com/repos/legacy/download-module/releases/tags/"
        monkeypatch.setattr(download, "MODELS_REPO", slashed)
        assert download._configured_models_repo() == slashed


class TestLoadAssetManifest:
    def test_slug_without_owner_repo_is_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry()],
            models_repo_slug="not-a-pair",
        )
        with pytest.raises(download.AssetTrustError, match="models_repo_slug=owner/repository"):
            download.load_asset_manifest(manifest_path)

    def test_empty_slug_is_rejected_when_base_url_missing(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry()],
            models_repo_slug="",
        )
        with pytest.raises(download.AssetTrustError, match="models_repo_slug=owner/repository"):
            download.load_asset_manifest(manifest_path)

    def test_slug_builds_github_releases_url(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry()],
            models_repo_slug="reviewed/models",
        )
        manifest = download.load_asset_manifest(manifest_path)
        assert manifest.models_repo == (
            "https://api.github.com/repos/reviewed/models/releases/tags/"
        )
        trusted = manifest.get("pipeline/example/1")
        assert trusted.filename == "model.bin"
        assert trusted.size == len(b"reviewed")

    def test_non_positive_size_is_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry(size=0)],
        )
        with pytest.raises(download.AssetTrustError, match="size must be positive"):
            download.load_asset_manifest(manifest_path)

    def test_invalid_sha256_length_is_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry(sha256="abc")],
        )
        with pytest.raises(download.AssetTrustError, match="SHA-256 is invalid"):
            download.load_asset_manifest(manifest_path)

    def test_non_hex_sha256_is_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[_asset_entry(sha256="g" * 64)],
        )
        with pytest.raises(download.AssetTrustError, match="SHA-256 is invalid"):
            download.load_asset_manifest(manifest_path)

    def test_duplicate_tag_is_rejected(self, tmp_path: Path) -> None:
        entry = _asset_entry()
        manifest_path = _write_manifest(
            tmp_path / "manifest.json",
            assets=[entry, dict(entry)],
        )
        with pytest.raises(download.AssetTrustError, match="Duplicate release tag"):
            download.load_asset_manifest(manifest_path)


class TestRequireMatchingRepository:
    def test_matching_repository_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = "https://api.github.com/repos/reviewed/models/releases/tags/"
        monkeypatch.setenv("LEXNLP_MODELS_REPO", repo)
        manifest = download.AssetManifest(models_repo=repo, assets={})
        assert download._require_matching_repository(manifest) == repo

    def test_mismatch_names_both_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "LEXNLP_MODELS_REPO",
            "https://api.github.com/repos/other/models/releases/tags/",
        )
        manifest = download.AssetManifest(
            models_repo="https://api.github.com/repos/reviewed/models/releases/tags/",
            assets={},
        )
        with pytest.raises(download.AssetTrustError, match="not the repository") as caught:
            download._require_matching_repository(manifest)
        message = str(caught.value)
        assert "other/models" in message
        assert "reviewed/models" in message


class TestVerifyFileAndTrustedAsset:
    def test_missing_path_raises_checksum_error(self, tmp_path: Path) -> None:
        trusted = download.TrustedAsset(
            tag="pipeline/example/1",
            filename="model.bin",
            size=1,
            sha256=_sha256(b"x"),
        )
        missing = tmp_path / "model.bin"
        with pytest.raises(download.ChecksumError, match="does not exist"):
            download._verify_file(missing, trusted)

    def test_directory_is_not_a_file(self, tmp_path: Path) -> None:
        trusted = download.TrustedAsset(
            tag="pipeline/example/1",
            filename="model.bin",
            size=1,
            sha256=_sha256(b"x"),
        )
        with pytest.raises(download.ChecksumError, match="does not exist"):
            download._verify_file(tmp_path, trusted)

    def test_size_mismatch_names_received_and_expected(self, tmp_path: Path) -> None:
        payload = b"reviewed"
        path = tmp_path / "model.bin"
        path.write_bytes(payload)
        trusted = download.TrustedAsset(
            tag="pipeline/example/1",
            filename="model.bin",
            size=len(payload) + 4,
            sha256=_sha256(payload),
        )
        with pytest.raises(download.ChecksumError, match="size verification failed") as caught:
            download._verify_file(path, trusted)
        assert f"received={len(payload)}" in str(caught.value)
        assert f"expected={len(payload) + 4}" in str(caught.value)

    def test_digest_mismatch_names_received_and_expected(self, tmp_path: Path) -> None:
        payload = b"reviewed"
        path = tmp_path / "model.bin"
        path.write_bytes(payload)
        expected = _sha256(b"different")
        trusted = download.TrustedAsset(
            tag="pipeline/example/1",
            filename="model.bin",
            size=len(payload),
            sha256=expected,
        )
        with pytest.raises(download.ChecksumError, match="SHA-256 verification failed") as caught:
            download._verify_file(path, trusted)
        assert f"received={_sha256(payload)}" in str(caught.value)
        assert f"expected={expected}" in str(caught.value)

    def test_verify_trusted_asset_file_accepts_matching_artifact(self, tmp_path: Path) -> None:
        payload = b"reviewed model payload"
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry(payload=payload)])
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(payload)

        trusted = download.verify_trusted_asset_file(
            artifact,
            "pipeline/example/1",
            manifest_path=manifest_path,
        )
        assert trusted.filename == "model.bin"
        assert trusted.size == len(payload)
        assert trusted.sha256 == _sha256(payload)

    def test_verify_trusted_asset_file_rejects_wrong_filename(self, tmp_path: Path) -> None:
        payload = b"reviewed model payload"
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry(payload=payload)])
        artifact = tmp_path / "other.bin"
        artifact.write_bytes(payload)
        with pytest.raises(download.AssetTrustError, match="filename does not match") as caught:
            download.verify_trusted_asset_file(
                artifact,
                "pipeline/example/1",
                manifest_path=manifest_path,
            )
        assert "received='other.bin'" in str(caught.value)
        assert "expected='model.bin'" in str(caught.value)

    def test_verify_trusted_asset_payload_skips_filename_constraint(self, tmp_path: Path) -> None:
        payload = b"reviewed model payload"
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry(payload=payload)])
        artifact = tmp_path / "staging.bin"
        artifact.write_bytes(payload)
        trusted = download.verify_trusted_asset_payload(
            artifact,
            "pipeline/example/1",
            manifest_path=manifest_path,
        )
        assert trusted.filename == "model.bin"
        assert trusted.sha256 == _sha256(payload)
