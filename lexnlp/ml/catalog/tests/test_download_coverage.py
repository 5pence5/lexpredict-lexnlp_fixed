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
            download._normalise_repo_url("https://api.github.com/repos/reviewed/models/releases/tags")
            == "https://api.github.com/repos/reviewed/models/releases/tags/"
        )

    def test_rejects_non_https_and_credentials(self) -> None:
        with pytest.raises(download.AssetTrustError, match="absolute HTTPS URL"):
            download._normalise_repo_url("http://api.github.com/repos/reviewed/models/releases/tags/")
        with pytest.raises(download.AssetTrustError, match="absolute HTTPS URL"):
            download._normalise_repo_url("https://user:pass@api.github.com/repos/reviewed/models/releases/tags/")


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
            download._configured_models_repo() == "https://api.github.com/repos/legacy/download-module/releases/tags/"
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
        assert manifest.models_repo == ("https://api.github.com/repos/reviewed/models/releases/tags/")
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


REVIEWED_REPO = "https://api.github.com/repos/reviewed/models/releases/tags/"


def _use_reviewed_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LEXNLP_MODELS_REPO", REVIEWED_REPO)
    monkeypatch.delenv("LEXNLP_MODELS_REPO_SLUG", raising=False)


class TestCatalogDestinationDirectory:
    def test_happy_path_returns_nested_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        monkeypatch.setattr(download, "CATALOG", catalog)
        destination = download._catalog_destination_directory("pipeline/example/1")
        assert destination.relative_to(catalog.resolve()) == Path("pipeline/example/1")

    def test_symlink_to_catalog_root_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        (catalog / "loop").symlink_to(catalog, target_is_directory=True)
        monkeypatch.setattr(download, "CATALOG", catalog)
        with pytest.raises(download.AssetTrustError, match="beneath the catalog"):
            download._catalog_destination_directory("loop")


class TestLoadAssetManifestErrors:
    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(download.AssetTrustError, match="Unable to load"):
            download.load_asset_manifest(tmp_path / "nope.json")

    def test_corrupt_json_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(download.AssetTrustError, match="Unable to load"):
            download.load_asset_manifest(bad)

    def test_unsupported_schema_is_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry()], schema_version=2)
        with pytest.raises(download.AssetTrustError, match="Unsupported"):
            download.load_asset_manifest(manifest_path)

    def test_entry_missing_key_is_rejected(self, tmp_path: Path) -> None:
        entry = _asset_entry()
        del entry["sha256"]
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[entry])
        with pytest.raises(download.AssetTrustError, match="Malformed entry"):
            download.load_asset_manifest(manifest_path)

    def test_entry_with_non_integer_size_is_rejected(self, tmp_path: Path) -> None:
        entry = _asset_entry()
        entry["size"] = "huge"
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[entry])
        with pytest.raises(download.AssetTrustError, match="Malformed entry"):
            download.load_asset_manifest(manifest_path)

    def test_empty_assets_are_rejected(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[])
        with pytest.raises(download.AssetTrustError, match="does not contain any assets"):
            download.load_asset_manifest(manifest_path)


class TestBuildGithubHeaders:
    def test_github_token_adds_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "tok-123")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        headers = download._build_github_headers({"Accept": "application/octet-stream"})
        assert headers == {"Accept": "application/octet-stream", "Authorization": "Bearer tok-123"}

    def test_gh_token_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "fallback-456")
        assert download._build_github_headers({}) == {"Authorization": "Bearer fallback-456"}

    def test_github_token_wins_over_gh_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "primary")
        monkeypatch.setenv("GH_TOKEN", "secondary")
        assert download._build_github_headers({}) == {"Authorization": "Bearer primary"}

    def test_no_token_leaves_headers_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert download._build_github_headers({"Accept": "x"}) == {"Accept": "x"}


class TestDownloadReleaseToPathWiring:
    def test_wires_tag_asset_and_install(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        payload = b"reviewed"
        manifest_path = _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry(payload=payload)])
        _use_reviewed_repo(monkeypatch)
        calls: dict[str, object] = {}

        class FakeTagResponse:
            def raise_for_status(self) -> None:
                calls["status_checked"] = True

        asset = {"name": "model.bin", "size": len(payload), "url": f"{REVIEWED_REPO}assets/1"}
        sentinel = tmp_path / "model.bin"

        def fake_get_tag(tag: str, *, models_repo: str | None = None) -> FakeTagResponse:
            calls["tag"] = tag
            calls["models_repo"] = models_repo
            return FakeTagResponse()

        def fake_get_asset(response: object, *, filename: str | None = None) -> dict:
            calls["response"] = response
            calls["filename"] = filename
            return asset

        def fake_download_to_path(cls: type, asset_arg: object, destination: object, **kwargs: object) -> Path:
            calls["asset"] = asset_arg
            calls["destination"] = destination
            calls["trusted"] = kwargs.get("trusted")
            return sentinel

        monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag", fake_get_tag)
        monkeypatch.setattr(download.GitHubReleaseDownloader, "get_asset", fake_get_asset)
        monkeypatch.setattr(
            download.GitHubReleaseDownloader,
            "download_asset_to_path",
            classmethod(fake_download_to_path),
        )

        result = download.GitHubReleaseDownloader.download_release_to_path(
            "pipeline/example/1",
            manifest_path=manifest_path,
        )

        assert result == sentinel
        assert calls["tag"] == "pipeline/example/1"
        assert calls["models_repo"] == REVIEWED_REPO
        assert isinstance(calls["response"], FakeTagResponse)
        assert calls["status_checked"] is True
        assert calls["filename"] == "model.bin"
        assert calls["asset"] == asset
        expected_trusted = download.load_asset_manifest(manifest_path).get("pipeline/example/1")
        assert calls["trusted"] == expected_trusted
        assert Path(str(calls["destination"])).as_posix().endswith("pipeline/example/1")


class TestGetAssetErrors:
    @staticmethod
    def _json_response(payload: object) -> object:
        class _Response:
            @staticmethod
            def json() -> object:
                return payload

        return _Response()

    def test_missing_assets_without_filename_reraises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Available keys"):
            download.GitHubReleaseDownloader.get_asset(self._json_response({"other": []}))

    def test_missing_assets_with_filename_is_trust_error(self) -> None:
        with pytest.raises(download.AssetTrustError, match="asset list"):
            download.GitHubReleaseDownloader.get_asset(self._json_response({"other": []}), filename="model.bin")

    def test_unparsable_response_is_trust_error(self) -> None:
        class _Boom:
            @staticmethod
            def json() -> object:
                raise ValueError("not json")

        with pytest.raises(download.AssetTrustError, match="asset list"):
            download.GitHubReleaseDownloader.get_asset(_Boom(), filename="model.bin")

    def test_null_response_is_trust_error(self) -> None:
        with pytest.raises(download.AssetTrustError, match="asset list"):
            download.GitHubReleaseDownloader.get_asset(self._json_response(None), filename="model.bin")

    def test_zero_matches_are_rejected(self) -> None:
        response = self._json_response({"assets": [{"name": "other.bin"}]})
        with pytest.raises(download.AssetTrustError, match="found 0"):
            download.GitHubReleaseDownloader.get_asset(response, filename="model.bin")

    def test_multiple_matches_are_rejected(self) -> None:
        response = self._json_response({"assets": [{"name": "model.bin"}, {"name": "model.bin"}]})
        with pytest.raises(download.AssetTrustError, match="found 2"):
            download.GitHubReleaseDownloader.get_asset(response, filename="model.bin")


class TestYieldAsset:
    def test_non_positive_content_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="content_length must be positive"):
            list(download.GitHubReleaseDownloader.yield_asset(iter([b"x"]), 0, 8192))
        with pytest.raises(ValueError, match="content_length must be positive"):
            list(download.GitHubReleaseDownloader.yield_asset(iter([b"x"]), -5, 8192))

    def test_empty_chunks_are_skipped(self) -> None:
        result = list(download.GitHubReleaseDownloader.yield_asset(iter([b"", b"abc", b"", b"de"]), 5, 2))
        assert result == [b"abc", b"de"]


class _FakeStreamResponse:
    def __init__(self, payload: bytes, content_length: int | None = "unset") -> None:  # type: ignore[assignment]
        self._payload = payload
        self.headers: dict[str, str] = {} if content_length == "unset" else {"Content-Length": str(content_length)}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 8192):  # type: ignore[no-untyped-def]
        for offset in range(0, len(self._payload), chunk_size):
            yield self._payload[offset : offset + chunk_size]


def _patch_http_get(monkeypatch: pytest.MonkeyPatch, response: object) -> None:
    monkeypatch.setattr(download, "_session", lambda: _FakeSession(response))


class _FakeSession:
    def __init__(self, response: object) -> None:
        self._response = response

    def get(self, **kwargs: object) -> object:
        return self._response


def _reviewed_trusted(payload: bytes) -> download.TrustedAsset:
    return download.TrustedAsset(
        tag="pipeline/example/1",
        filename="model.bin",
        size=len(payload),
        sha256=_sha256(payload),
    )


class TestDownloadAssetLegacyResolution:
    def _manifest(self, *trusted_assets: download.TrustedAsset) -> download.AssetManifest:
        return download.AssetManifest(
            models_repo=REVIEWED_REPO,
            assets={asset.tag: asset for asset in trusted_assets},
        )

    def test_unknown_filename_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        manifest = self._manifest(_reviewed_trusted(b"reviewed"))
        monkeypatch.setattr(download, "load_asset_manifest", lambda *args, **kwargs: manifest)
        _use_reviewed_repo(monkeypatch)
        asset = {"name": "unknown.bin", "size": 8, "url": f"{REVIEWED_REPO}assets/9"}
        with pytest.raises(download.AssetTrustError, match="exactly one asset"):
            download.GitHubReleaseDownloader.download_asset_to_path(asset, tmp_path)

    def test_ambiguous_filename_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        payload = b"reviewed"
        first = _reviewed_trusted(payload)
        second = download.TrustedAsset(
            tag="pipeline/example/2", filename=first.filename, size=first.size, sha256=first.sha256
        )
        manifest = self._manifest(first, second)
        monkeypatch.setattr(download, "load_asset_manifest", lambda *args, **kwargs: manifest)
        _use_reviewed_repo(monkeypatch)
        asset = {"name": first.filename, "size": first.size, "url": f"{REVIEWED_REPO}assets/9"}
        with pytest.raises(download.AssetTrustError, match="exactly one asset"):
            download.GitHubReleaseDownloader.download_asset_to_path(asset, tmp_path)

    def test_name_mismatch_is_rejected(self, tmp_path: Path) -> None:
        trusted = _reviewed_trusted(b"reviewed")
        asset = {"name": "other.bin", "size": trusted.size, "url": f"{REVIEWED_REPO}assets/1"}
        with pytest.raises(download.AssetTrustError, match="does not match the trusted manifest"):
            download.GitHubReleaseDownloader.download_asset_to_path(
                asset, tmp_path, trusted=trusted, expected_host="api.github.com"
            )

    def test_size_mismatch_is_rejected(self, tmp_path: Path) -> None:
        trusted = _reviewed_trusted(b"reviewed")
        asset = {"name": trusted.filename, "size": trusted.size + 1, "url": f"{REVIEWED_REPO}assets/1"}
        with pytest.raises(download.AssetTrustError, match="does not match the trusted manifest"):
            download.GitHubReleaseDownloader.download_asset_to_path(
                asset, tmp_path, trusted=trusted, expected_host="api.github.com"
            )


class TestDownloadContentValidation:
    def test_content_length_mismatch_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        payload = b"reviewed"
        trusted = _reviewed_trusted(payload)
        asset = {"name": trusted.filename, "size": trusted.size, "url": f"{REVIEWED_REPO}assets/1"}
        _patch_http_get(monkeypatch, _FakeStreamResponse(payload, content_length=len(payload) - 1))
        with pytest.raises(download.AssetTrustError, match="Content-Length"):
            download.GitHubReleaseDownloader.download_asset_to_path(
                asset, tmp_path, trusted=trusted, expected_host="api.github.com"
            )

    def test_truncated_body_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        payload = b"reviewed"
        trusted = _reviewed_trusted(payload)
        asset = {"name": trusted.filename, "size": trusted.size, "url": f"{REVIEWED_REPO}assets/1"}
        _patch_http_get(monkeypatch, _FakeStreamResponse(payload[:3]))
        with pytest.raises(download.ChecksumError, match="truncated") as caught:
            download.GitHubReleaseDownloader.download_asset_to_path(
                asset, tmp_path, trusted=trusted, expected_host="api.github.com"
            )
        assert "received=3" in str(caught.value)
        assert f"expected={len(payload)}" in str(caught.value)


class TestDownloadGithubReleasePrompt:
    def _manifest_path(self, tmp_path: Path) -> Path:
        return _write_manifest(tmp_path / "manifest.json", assets=[_asset_entry()])

    def test_decline_returns_none_without_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        manifest_path = self._manifest_path(tmp_path)
        _use_reviewed_repo(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        calls: list[str] = []
        monkeypatch.setattr(
            download.GitHubReleaseDownloader,
            "download_release_to_path",
            classmethod(lambda cls, tag, **kwargs: calls.append(tag)),
        )
        assert download.download_github_release("pipeline/example/1", manifest_path=manifest_path) is None
        assert calls == []

    def test_invalid_answer_is_rejected(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        manifest_path = self._manifest_path(tmp_path)
        _use_reviewed_repo(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _prompt: "maybe")
        with pytest.raises(ValueError, match="User input must be"):
            download.download_github_release("pipeline/example/1", manifest_path=manifest_path)

    @pytest.mark.parametrize("answer", ["y", ""])
    def test_accept_delegates_to_downloader(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str) -> None:
        manifest_path = self._manifest_path(tmp_path)
        _use_reviewed_repo(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda _prompt: answer)
        calls: list[str] = []
        sentinel = tmp_path / "model.bin"
        monkeypatch.setattr(
            download.GitHubReleaseDownloader,
            "download_release_to_path",
            classmethod(lambda cls, tag, **kwargs: calls.append(tag) or sentinel),
        )
        assert download.download_github_release("pipeline/example/1", manifest_path=manifest_path) is None
        assert calls == ["pipeline/example/1"]


class TestBytesToHumanReadable:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (1, "1.00 B"),
            (512, "512.00 B"),
            (2048, "2.00 KiB"),
            (5 * 1024**2, "5.00 MiB"),
            (3 * 1024**3, "3.00 GiB"),
        ],
    )
    def test_binary_units(self, size: int, expected: str) -> None:
        assert download._bytes_to_human_readable(size) == expected

    def test_tebibytes_use_tib_suffix(self) -> None:
        assert download._bytes_to_human_readable(5 * 1024**4) == "5.0 TiB"
