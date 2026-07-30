"""Regression tests for scheduled release-asset drift checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
