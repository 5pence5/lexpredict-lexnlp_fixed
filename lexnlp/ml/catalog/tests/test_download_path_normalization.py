"""Tests for :func:`GitHubReleaseDownloader.download_asset` Path handling.

PR #14 review noted ``destination_directory`` accepts ``Path | str`` but the
implementation called ``.mkdir`` on the raw argument, which would explode
with ``AttributeError`` when a ``str`` was passed. After the fix, the
helper normalizes the argument via ``Path(destination_directory)`` before
use.

We mock out the network-touching ``requests.get`` so the test does not hit
GitHub. The focus is on the filesystem branch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from lexnlp.ml.catalog.download import GitHubReleaseDownloader, TrustedAsset


def _trusted(name: str, payload: bytes) -> TrustedAsset:
    """Describe ``payload`` so the fail-closed download check can verify it.

    Downloads are now checked against a reviewed manifest, so a test asset has
    to carry its own trusted entry rather than relying on an unlisted name.
    """
    return TrustedAsset(
        tag="pipeline/example/1",
        filename=name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _fake_response(payload: bytes) -> MagicMock:
    """
    Create a mocked HTTP response that simulates streaming `payload` as a download.

    The mock supports the context-manager protocol so ``with _session().get(...)
    as response:`` in :mod:`lexnlp.ml.catalog.download` works as intended;
    ``__enter__`` returns the mock itself and ``__exit__`` is a no-op.

    Parameters:
        payload (bytes): The byte content to be yielded by the response's stream.

    Returns:
        MagicMock: A mock response with:
            - `raise_for_status()` as a no-op,
            - `headers["Content-Length"]` set to the length of `payload`,
            - `iter_content(chunk_size)` yielding a single chunk equal to `payload`,
            - context-manager protocol (``__enter__`` / ``__exit__``).
    """
    response = MagicMock()
    response.raise_for_status = lambda: None
    response.headers = {"Content-Length": str(len(payload))}
    response.iter_content = lambda chunk_size=8192: iter([payload])
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    return response


class TestDownloadAssetNormalizesPath:
    @patch("lexnlp.ml.catalog.download._session")
    def test_str_destination_does_not_raise(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_get.return_value.get.return_value = _fake_response(b"payload")
        asset = {"url": "https://api.github.com/repos/LexPredict/lexpredict-lexnlp/releases/assets/1", "name": "thing.bin", "size": 7}
        # Pass the directory as a *str* on purpose to exercise the fix.
        GitHubReleaseDownloader.download_asset(asset, str(tmp_path), trusted=_trusted("thing.bin", b"payload"))
        assert (tmp_path / "thing.bin").read_bytes() == b"payload"

    @patch("lexnlp.ml.catalog.download._session")
    def test_path_destination_still_works(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_get.return_value.get.return_value = _fake_response(b"abc")
        asset = {"url": "https://api.github.com/repos/LexPredict/lexpredict-lexnlp/releases/assets/1", "name": "abc.bin", "size": 3}
        GitHubReleaseDownloader.download_asset(asset, tmp_path, trusted=_trusted("abc.bin", b"abc"))
        assert (tmp_path / "abc.bin").read_bytes() == b"abc"

    @patch("lexnlp.ml.catalog.download._session")
    def test_nested_destination_is_created(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_get.return_value.get.return_value = _fake_response(b"ok")
        nested = tmp_path / "a" / "b" / "c"
        asset = {"url": "https://api.github.com/repos/LexPredict/lexpredict-lexnlp/releases/assets/1", "name": "n.bin", "size": 2}
        GitHubReleaseDownloader.download_asset(asset, str(nested), trusted=_trusted("n.bin", b"ok"))
        assert (nested / "n.bin").exists()
