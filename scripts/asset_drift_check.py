#!/usr/bin/env python3
"""Verify that required release-tag assets exist and match pinned hashes.

This is intended for scheduled CI ("asset drift") to catch situations where
release assets disappear or are replaced under the same tag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


DEFAULT_MANIFEST = Path("test_data/model_quality/release_asset_manifest.json")
UNPUBLISHED_RUNTIME_CANDIDATE_TAGS = frozenset({
    "pipeline/is-contract/0.2",
    "pipeline/contract-type/0.2-runtime",
})


class DriftError(Exception):
    pass


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Manifest JSON path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing tags before verifying hashes.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always re-download tags before verifying hashes (recommended for scheduled CI).",
    )
    parser.add_argument(
        "--allow-unpublished-runtime-candidates",
        action="store_true",
        help=(
            "Report the two default upstream runtime candidates as unpublished only "
            "when their canonical release-tag metadata returns HTTP 404. Published "
            "release assets and all other errors remain mandatory checks."
        ),
    )
    return parser.parse_args(argv)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Manifest JSON must be an object")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("Manifest JSON must contain a non-empty 'assets' list")
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_assets(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for item in payload["assets"]:
        if not isinstance(item, dict):
            raise ValueError("Manifest assets must be objects")
        for key in ("tag", "filename", "sha256"):
            if key not in item:
                raise ValueError(f"Manifest asset missing key={key}")
        yield item


def ensure_tag_downloaded(tag: str, *, manifest_path: Path) -> Path:
    from lexnlp.ml.catalog import get_exact_path_from_catalog
    from lexnlp.ml.catalog.download import download_github_release_to_path

    try:
        return get_exact_path_from_catalog(tag)
    except FileNotFoundError:
        return download_github_release_to_path(
            tag,
            manifest_path=manifest_path,
        )


def runtime_candidate_is_unpublished(tag: str, *, manifest_path: Path) -> bool:
    """Accept only an absent default upstream release, never a broken asset."""
    if tag not in UNPUBLISHED_RUNTIME_CANDIDATE_TAGS:
        return False

    from lexnlp import DEFAULT_MODELS_REPO
    from lexnlp.ml.catalog.download import (
        GitHubReleaseDownloader,
        _require_matching_repository,
        load_asset_manifest,
    )

    manifest = load_asset_manifest(manifest_path)
    manifest.get(tag)
    models_repo = _require_matching_repository(manifest)
    if models_repo != DEFAULT_MODELS_REPO:
        # The exception is for these unpublished upstream releases, rather
        # than custom repositories or arbitrary caller-supplied manifests.
        return False

    response = GitHubReleaseDownloader.get_tag(tag, models_repo=models_repo)
    try:
        if response.status_code == 404 and response.url == models_repo + tag:
            return True
        response.raise_for_status()
        return False
    finally:
        response.close()


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    payload = load_manifest(args.manifest)

    failures: list[str] = []

    for asset in iter_assets(payload):
        tag = str(asset["tag"]).strip()
        expected_name = str(asset["filename"]).strip()
        expected_sha = str(asset["sha256"]).strip().lower()
        expected_size = asset.get("size")

        try:
            from lexnlp.ml.catalog import get_exact_path_from_catalog
            from lexnlp.ml.catalog.download import download_github_release_to_path

            if (args.allow_unpublished_runtime_candidates
                    and runtime_candidate_is_unpublished(tag, manifest_path=args.manifest)):
                print(f"asset-drift: UNPUBLISHED {tag} (canonical upstream release metadata returned HTTP 404)")
                continue

            if args.force_download:
                path = download_github_release_to_path(
                    tag,
                    manifest_path=args.manifest,
                    force=True,
                )
            elif args.download_missing:
                path = ensure_tag_downloaded(
                    tag,
                    manifest_path=args.manifest,
                )
            else:
                path = get_exact_path_from_catalog(tag)
        except Exception as exc:
            failures.append(f"{tag}: missing/unreadable ({exc})")
            continue

        if path.name != expected_name:
            failures.append(f"{tag}: unexpected filename {path.name!r} (expected {expected_name!r})")
            continue

        if expected_size is not None:
            try:
                expected_size_int = int(expected_size)
            except (TypeError, ValueError):
                failures.append(f"{tag}: invalid manifest size={expected_size!r}")
                continue
            actual_size = path.stat().st_size
            if actual_size != expected_size_int:
                failures.append(f"{tag}: size mismatch {actual_size} != {expected_size_int}")
                continue

        actual_sha = sha256_file(path)
        if actual_sha.lower() != expected_sha:
            failures.append(f"{tag}: sha256 mismatch {actual_sha} != {expected_sha}")
            continue

        print(f"asset-drift: OK {tag} ({path.name})")

    if failures:
        for failure in failures:
            print(f"asset-drift: ERROR {failure}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
