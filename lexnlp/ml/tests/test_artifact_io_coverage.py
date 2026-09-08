"""Coverage tests for lexnlp.ml.artifact_io crash-safe artifact writers."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

from lexnlp.ml.artifact_io import atomic_output_path, atomic_pickle_dump


class TestAtomicOutputPathSymlink:
    def test_symlink_destination_is_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "real.bin"
        target.write_bytes(b"real")
        link = tmp_path / "link.bin"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            with atomic_output_path(link):
                pass
        assert target.read_bytes() == b"real"

    def test_broken_symlink_destination_is_rejected(self, tmp_path: Path) -> None:
        link = tmp_path / "dangling.bin"
        link.symlink_to(tmp_path / "missing.bin")
        with pytest.raises(ValueError, match="symlink"):
            with atomic_output_path(link):
                pass


class TestAtomicOutputPathEmpty:
    def test_untouched_temporary_file_is_rejected(self, tmp_path: Path) -> None:
        destination = tmp_path / "model.bin"
        with pytest.raises(RuntimeError, match="empty model artifact"):
            with atomic_output_path(destination):
                pass
        assert not destination.exists()

    def test_deleted_temporary_file_is_rejected(self, tmp_path: Path) -> None:
        destination = tmp_path / "model.bin"
        with pytest.raises(RuntimeError, match="empty model artifact"):
            with atomic_output_path(destination) as temporary_path:
                temporary_path.unlink()
        assert not destination.exists()

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        destination = tmp_path / "model.bin"
        with pytest.raises(RuntimeError, match="empty model artifact"):
            with atomic_output_path(destination) as temporary_path:
                temporary_path.write_bytes(b"")
        assert not destination.exists()


class TestAtomicOutputPathPublish:
    def test_content_is_published_and_temp_removed(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "model.bin"
        with atomic_output_path(destination) as temporary_path:
            temporary_path.write_bytes(b"payload-bytes")
        assert destination.read_bytes() == b"payload-bytes"
        assert list(tmp_path.rglob("*.tmp")) == []

    def test_directory_fsync_failure_still_publishes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        real_open = os.open

        def fail_dir_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
            if Path(os.fspath(path)).is_dir():
                raise OSError("directory handles unavailable")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", fail_dir_open)
        destination = tmp_path / "model.bin"
        with atomic_output_path(destination) as temporary_path:
            temporary_path.write_bytes(b"durable")
        assert destination.read_bytes() == b"durable"


class TestAtomicPickleDump:
    def test_default_protocol_round_trips(self, tmp_path: Path) -> None:
        destination = tmp_path / "model.pickle"
        value = {"weights": [1, 2, 3], "label": "contract"}
        atomic_pickle_dump(value, destination)
        with destination.open("rb") as handle:
            assert pickle.load(handle) == value

    def test_explicit_protocol_round_trips(self, tmp_path: Path) -> None:
        destination = tmp_path / "model.pickle"
        value = ["a", "b", "c"]
        atomic_pickle_dump(value, destination, protocol=2)
        with destination.open("rb") as handle:
            assert pickle.load(handle) == value
