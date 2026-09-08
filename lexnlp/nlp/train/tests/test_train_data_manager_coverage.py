"""Coverage tests for lexnlp.nlp.train.train_data_manager."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from lexnlp.nlp.train.train_data_manager import ensure_documents_in_folder


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestEnsureDocumentsInFolder:
    def test_empty_mapping_returns_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        assert ensure_documents_in_folder({}, str(target), OrderedDict()) == {}

    def test_target_hit_needs_no_alias(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        existing = _write(target / "doc.txt", "already here")
        result = ensure_documents_in_folder({"/somewhere/else/doc.txt": {"label": 1}}, str(target), OrderedDict())
        assert result == {str(existing): {"label": 1}}
        assert existing.read_text(encoding="utf-8") == "already here"

    def test_alias_copy_preserves_bytes_and_metadata(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write(src_dir / "doc.txt", "source bytes")
        target = tmp_path / "target"
        target.mkdir()
        aliases: OrderedDict[str, str] = OrderedDict([("/alias/", str(src_dir) + "/")])
        result = ensure_documents_in_folder({"/alias/doc.txt": [1, 2]}, str(target), aliases)
        expected = str(target / "doc.txt")
        assert result == {expected: [1, 2]}
        assert Path(expected).read_text(encoding="utf-8") == "source bytes"

    def test_non_matching_alias_is_skipped(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write(src_dir / "doc.txt", "source bytes")
        target = tmp_path / "target"
        target.mkdir()
        aliases: OrderedDict[str, str] = OrderedDict([("/other/", str(src_dir) + "/")])
        result = ensure_documents_in_folder({"/alias/doc.txt": "meta"}, str(target), aliases)
        assert result == {}
        assert not (target / "doc.txt").exists()

    def test_missing_alias_source_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "target"
        target.mkdir()
        aliases: OrderedDict[str, str] = OrderedDict([("/alias/", str(tmp_path / "absent") + "/")])
        result = ensure_documents_in_folder({"/alias/doc.txt": "meta"}, str(target), aliases)
        assert result == {}
        assert not (target / "doc.txt").exists()

    def test_first_usable_alias_wins(self, tmp_path: Path) -> None:
        good = tmp_path / "good" / "sub"
        good.mkdir(parents=True)
        _write(good / "doc.txt", "good bytes")
        target = tmp_path / "target"
        target.mkdir()
        aliases: OrderedDict[str, str] = OrderedDict(
            [
                ("/alias/sub/", str(tmp_path / "absent") + "/"),
                ("/alias/", str(tmp_path / "good") + "/"),
            ]
        )
        result = ensure_documents_in_folder({"/alias/sub/doc.txt": 7}, str(target), aliases)
        expected = str(target / "doc.txt")
        assert result == {expected: 7}
        assert Path(expected).read_text(encoding="utf-8") == "good bytes"

    def test_mixed_batch(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        _write(src_dir / "copied.txt", "copied")
        target = tmp_path / "target"
        target.mkdir()
        present = _write(target / "present.txt", "present")
        aliases: OrderedDict[str, str] = OrderedDict([("/alias/", str(src_dir) + "/")])
        result = ensure_documents_in_folder(
            {
                "/anywhere/present.txt": "m1",
                "/alias/copied.txt": "m2",
                "/alias/ghost.txt": "m3",
            },
            str(target),
            aliases,
        )
        assert result == {str(present): "m1", str(target / "copied.txt"): "m2"}
        assert (target / "copied.txt").read_text(encoding="utf-8") == "copied"
        assert not (target / "ghost.txt").exists()
