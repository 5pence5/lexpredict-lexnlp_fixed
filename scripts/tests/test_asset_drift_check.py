"""Regression tests for scheduled release-asset drift checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from requests import HTTPError, Response, Timeout

from scripts import asset_drift_check


def test_force_download_requests_remote_bytes_and_uses_returned_path(
    monkeypatch,
    tmp_path: Path,
):
    from lexnlp.ml.catalog import download

    payload = b"fresh remote bytes"
    tag = "pipeline/example/1"
    filename = "model.bin"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models_repo_slug": "reviewed/models",
                "assets": [
                    {
                        "tag": tag,
                        "filename": filename,
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    remote_path = tmp_path / "remote" / filename
    remote_path.parent.mkdir()
    remote_path.write_bytes(payload)
    calls = []

    def fake_download(
        requested_tag,
        *,
        manifest_path,
        force,
    ):
        calls.append((requested_tag, manifest_path, force))
        return remote_path

    monkeypatch.setattr(
        download,
        "download_github_release_to_path",
        fake_download,
    )

    assert (
        asset_drift_check.main(
            [
                "--manifest",
                str(manifest_path),
                "--force-download",
            ]
        )
        == 0
    )
    assert calls == [(tag, manifest_path, True)]


def test_download_missing_ignores_private_local_candidate(
    monkeypatch,
    tmp_path: Path,
):
    from lexnlp.ml import catalog
    from lexnlp.ml.catalog import download

    payload = b"reviewed release bytes"
    tag = "pipeline/example/1"
    filename = "model.bin"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models_repo_slug": "reviewed/models",
                "assets": [
                    {
                        "tag": tag,
                        "filename": filename,
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog_root = tmp_path / "catalog"
    local_candidate = (
        catalog_root / catalog.get_local_candidate_tag(tag) / filename
    )
    local_candidate.parent.mkdir(parents=True)
    local_candidate.write_bytes(b"private candidate bytes")
    downloaded = catalog_root / tag / filename
    calls = []

    def fake_download(requested_tag, *, manifest_path):
        calls.append((requested_tag, manifest_path))
        downloaded.parent.mkdir(parents=True)
        downloaded.write_bytes(payload)
        catalog.invalidate_catalog_cache()
        return downloaded

    monkeypatch.setattr(catalog, "CATALOG", catalog_root)
    monkeypatch.setattr(
        download,
        "download_github_release_to_path",
        fake_download,
    )
    catalog.invalidate_catalog_cache()

    assert (
        asset_drift_check.main(
            [
                "--manifest",
                str(manifest_path),
                "--download-missing",
            ]
        )
        == 0
    )
    assert calls == [(tag, manifest_path)]
    catalog.invalidate_catalog_cache()


def _candidate_manifest(tmp_path, tag="pipeline/is-contract/0.2"):
    payload = b"reviewed runtime release bytes"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "models_repo_slug": "LexPredict/lexpredict-lexnlp",
        "assets": [{"tag": tag, "filename": "model.bin", "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")
    return path, payload


def _response(status, url):
    response = Response()
    response.status_code = status
    response.url = url
    response._content = b"{}"
    response._content_consumed = True
    return response


def _allow_candidate_args(manifest):
    return ["--manifest", str(manifest), "--force-download", "--allow-unpublished-runtime-candidates"]


@pytest.mark.parametrize("tag", sorted(asset_drift_check.UNPUBLISHED_RUNTIME_CANDIDATE_TAGS))
def test_only_canonical_unpublished_default_release_metadata_is_allowed(
    monkeypatch, tmp_path, capsys, tag,
):
    from lexnlp import DEFAULT_MODELS_REPO
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path, tag)
    calls = []

    def missing_metadata(requested_tag, *, models_repo):
        calls.append((requested_tag, models_repo))
        return _response(404, models_repo + requested_tag)

    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag", missing_metadata)
    monkeypatch.setattr(download, "download_github_release_to_path",
                        lambda *args, **kwargs: pytest.fail("unpublished release has no asset to download"))
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 0
    assert calls == [(tag, DEFAULT_MODELS_REPO)]
    assert "UNPUBLISHED " + tag in capsys.readouterr().out


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_release_metadata_authentication_and_server_errors_fail(monkeypatch, tmp_path, capsys, status):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path)
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda tag, *, models_repo: _response(status, models_repo + tag))
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1
    assert "ERROR" in capsys.readouterr().err


def test_release_metadata_network_failure_remains_visible(monkeypatch, tmp_path, capsys):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path)

    def unavailable(*args, **kwargs):
        raise Timeout("release metadata timeout")

    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag", unavailable)
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1
    assert "metadata timeout" in capsys.readouterr().err


def test_redirected_metadata_404_is_not_a_canonical_unpublished_release(monkeypatch, tmp_path):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path)
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda *args, **kwargs: _response(404, "https://example.test/missing"))
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1


@pytest.mark.parametrize("failure", ["missing-asset", "asset-404", "checksum"])
def test_published_release_asset_failures_are_never_unpublished(monkeypatch, tmp_path, capsys, failure):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path)
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda tag, *, models_repo: _response(200, models_repo + tag))

    def broken_asset(*args, **kwargs):
        if failure == "missing-asset":
            raise download.AssetTrustError("published release does not contain its pinned asset")
        if failure == "checksum":
            raise download.ChecksumError("published asset checksum mismatch")
        response = _response(404, "https://api.github.com/repos/LexPredict/lexpredict-lexnlp/releases/assets/1")
        raise HTTPError("asset download returned 404", response=response)

    monkeypatch.setattr(download, "download_github_release_to_path", broken_asset)
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1
    captured = capsys.readouterr()
    assert "UNPUBLISHED" not in captured.out
    assert "ERROR" in captured.err


def test_runtime_candidate_is_verified_automatically_after_publication(monkeypatch, tmp_path, capsys):
    from lexnlp.ml.catalog import download

    manifest, payload = _candidate_manifest(tmp_path)
    model = tmp_path / "model.bin"
    model.write_bytes(payload)
    calls = []
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda tag, *, models_repo: _response(200, models_repo + tag))

    def downloaded(tag, *, manifest_path, force):
        calls.append((tag, manifest_path, force))
        return model

    monkeypatch.setattr(download, "download_github_release_to_path", downloaded)
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 0
    assert calls == [("pipeline/is-contract/0.2", manifest, True)]
    assert "asset-drift: OK" in capsys.readouterr().out
    model.write_bytes(b"tampered runtime release bytes")
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1


@pytest.mark.parametrize("tag,use_allowance", [
    ("pipeline/is-contract/0.1", True), ("pipeline/custom/1", True),
    ("pipeline/is-contract/0.2", False),
])
def test_required_sources_and_default_strict_mode_reject_missing_tags(
    monkeypatch, tmp_path, capsys, tag, use_allowance,
):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path, tag)
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda *args, **kwargs: pytest.fail("strict required tag must use verified downloader"))

    def missing_release(*args, **kwargs):
        raise HTTPError("required release metadata returned 404", response=_response(404, "https://api.github.com"))

    monkeypatch.setattr(download, "download_github_release_to_path", missing_release)
    args = _allow_candidate_args(manifest)
    if not use_allowance:
        args.remove("--allow-unpublished-runtime-candidates")
    assert asset_drift_check.main(args) == 1
    assert "required release metadata returned 404" in capsys.readouterr().err


def test_candidate_allowance_never_overrides_manifest_repository_trust(monkeypatch, tmp_path):
    from lexnlp.ml.catalog import download

    manifest, _ = _candidate_manifest(tmp_path)
    monkeypatch.setenv("LEXNLP_MODELS_REPO_SLUG", "unreviewed/models")
    monkeypatch.setattr(download.GitHubReleaseDownloader, "get_tag",
                        lambda *args, **kwargs: pytest.fail("repository mismatch must fail before network access"))
    assert asset_drift_check.main(_allow_candidate_args(manifest)) == 1
