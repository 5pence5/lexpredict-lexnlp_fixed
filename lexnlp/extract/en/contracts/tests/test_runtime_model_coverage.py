__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import tarfile
from pathlib import Path

import pytest


def _make_tar(archive_path: Path, members: dict[str, str]) -> None:
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, content in members.items():
            import io

            payload = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _add_dir_entry(archive_path: Path) -> None:
    with tarfile.open(archive_path, mode="a") as archive:
        info = tarfile.TarInfo(name="CONTRACT_TYPES")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)


class TestEnsureTagDownloaded:
    def test_returns_catalog_path_without_download(self, monkeypatch, tmp_path) -> None:
        from lexnlp.extract.en.contracts import runtime_model

        cached = tmp_path / "model.skops"
        cached.write_bytes(b"x")
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: cached)

        calls: list[str] = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append(tag),
        )
        assert runtime_model.ensure_tag_downloaded("some/tag") == cached
        assert calls == []

    def test_downloads_on_catalog_miss(self, monkeypatch, tmp_path) -> None:
        from lexnlp.extract.en.contracts import runtime_model

        cached = tmp_path / "model.skops"
        cached.write_bytes(b"x")
        state = {"tries": 0}

        def fake_get(tag: str):
            state["tries"] += 1
            if state["tries"] == 1:
                raise FileNotFoundError(tag)
            return cached

        seen: dict[str, object] = {}

        def fake_download(tag: str, prompt_user: bool = False):
            seen["tag"] = tag
            seen["prompt_user"] = prompt_user

        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", fake_get)
        monkeypatch.setattr("lexnlp.ml.catalog.download.download_github_release", fake_download)
        assert runtime_model.ensure_tag_downloaded("some/tag") == cached
        assert seen == {"tag": "some/tag", "prompt_user": False}


class TestLoadPipelineForTag:
    def test_loads_model_from_tag_path(self, monkeypatch, tmp_path) -> None:
        from lexnlp.extract.en.contracts import runtime_model

        sentinel_path = tmp_path / "pipe.skops"
        sentinel_path.write_bytes(b"x")
        sentinel_pipeline = object()
        seen: dict[str, object] = {}
        monkeypatch.setattr(runtime_model, "ensure_tag_downloaded", lambda tag: sentinel_path)

        def fake_load(path, trusted: bool = False):
            seen["path"] = path
            seen["trusted"] = trusted
            return sentinel_pipeline

        monkeypatch.setattr(runtime_model, "load_model", fake_load)
        assert runtime_model.load_pipeline_for_tag("some/tag") is sentinel_pipeline
        assert seen == {"path": sentinel_path, "trusted": True}


class TestExtractLabel:
    def test_extracts_parent_directory(self) -> None:
        from lexnlp.extract.en.contracts.runtime_model import _extract_label

        assert _extract_label("CONTRACT_TYPES/NDA/doc1.txt") == "NDA"

    def test_short_path_raises(self) -> None:
        from lexnlp.extract.en.contracts.runtime_model import _extract_label

        with pytest.raises(ValueError, match="Unexpected corpus member path"):
            _extract_label("doc1.txt")


class TestCollectSamples:
    def test_rejects_non_positive_limits(self, tmp_path) -> None:
        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        archive = tmp_path / "corpus.tar.gz"
        archive.write_bytes(b"x")
        with pytest.raises(ValueError, match="max_docs_per_label"):
            collect_contract_type_samples(archive, max_docs_per_label=0, head_character_n=100)
        with pytest.raises(ValueError, match="head_character_n"):
            collect_contract_type_samples(archive, max_docs_per_label=1, head_character_n=0)

    def test_happy_path_with_caps_and_truncation(self, tmp_path) -> None:
        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        archive = tmp_path / "corpus.tar.gz"
        long_text = "alpha beta gamma " * 50
        _make_tar(
            archive,
            {
                "CONTRACT_TYPES/AAA/a1.txt": long_text,
                "CONTRACT_TYPES/AAA/a2.txt": "second aaa doc here",
                "CONTRACT_TYPES/AAA/a3.txt": "third aaa doc here",
                "CONTRACT_TYPES/BBB/b1.txt": "bbb doc one two three",
                "CONTRACT_TYPES/BBB/b2.txt": "   ",
                "notes.md": "top-level notes without label dirs",
            },
        )
        texts, labels, counts = collect_contract_type_samples(archive, max_docs_per_label=2, head_character_n=20)
        assert counts == {"AAA": 2, "BBB": 1}
        assert labels == ["AAA", "AAA", "BBB"]
        assert all(len(t) <= 20 for t in texts)
        assert texts[0] == long_text[:20]
        assert "second aaa doc here"[:20] in texts

    def test_skips_non_txt_and_directories(self, tmp_path) -> None:
        import io

        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        archive = tmp_path / "corpus.tar"
        with tarfile.open(archive, mode="w") as tar:
            dir_info = tarfile.TarInfo(name="CONTRACT_TYPES")
            dir_info.type = tarfile.DIRTYPE
            tar.addfile(dir_info)
            payload = b"binary content here"
            info = tarfile.TarInfo(name="CONTRACT_TYPES/AAA/blob.bin")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            for name, content in [
                ("CONTRACT_TYPES/AAA/ok.txt", "aaa text here"),
                ("CONTRACT_TYPES/BBB/ok.txt", "bbb text here"),
            ]:
                raw = content.encode()
                item = tarfile.TarInfo(name=name)
                item.size = len(raw)
                tar.addfile(item, io.BytesIO(raw))
        texts, labels, counts = collect_contract_type_samples(archive, max_docs_per_label=5, head_character_n=4000)
        assert sorted(labels) == ["AAA", "BBB"]
        assert sum(counts.values()) == 2
        assert len(texts) == 2

    def test_unreadable_member_is_skipped(self, monkeypatch, tmp_path) -> None:
        from types import SimpleNamespace

        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        good_a = SimpleNamespace(name="CONTRACT_TYPES/AAA/ok.txt", isfile=lambda: True)
        good_b = SimpleNamespace(name="CONTRACT_TYPES/BBB/ok.txt", isfile=lambda: True)
        bad = SimpleNamespace(name="CONTRACT_TYPES/BBB/broken.txt", isfile=lambda: True)

        class FakeArchive:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def getmembers(self):
                return [good_a, good_b, bad]

            def extractfile(self, member):
                if member is bad:
                    return None
                import io

                return io.BytesIO(b"sample text here")

        monkeypatch.setattr(tarfile, "open", lambda *args, **kwargs: FakeArchive())
        texts, labels, counts = collect_contract_type_samples(
            tmp_path / "corpus.tar.gz", max_docs_per_label=5, head_character_n=4000
        )
        assert sorted(labels) == ["AAA", "BBB"]
        assert counts == {"AAA": 1, "BBB": 1}
        assert texts == ["sample text here", "sample text here"]

    def test_empty_archive_raises(self, tmp_path) -> None:
        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        archive = tmp_path / "empty.tar.gz"
        _make_tar(archive, {"CONTRACT_TYPES/AAA/blank.txt": "   "})
        with pytest.raises(RuntimeError, match="No samples collected"):
            collect_contract_type_samples(archive, max_docs_per_label=5, head_character_n=100)

    def test_single_label_raises(self, tmp_path) -> None:
        from lexnlp.extract.en.contracts.runtime_model import collect_contract_type_samples

        archive = tmp_path / "single.tar.gz"
        _make_tar(
            archive,
            {
                "CONTRACT_TYPES/AAA/a1.txt": "only one label present",
                "CONTRACT_TYPES/AAA/a2.txt": "another doc same label",
            },
        )
        with pytest.raises(RuntimeError, match="at least two labels"):
            collect_contract_type_samples(archive, max_docs_per_label=5, head_character_n=100)


class TestTrainPipeline:
    def test_length_mismatch_raises(self) -> None:
        from lexnlp.extract.en.contracts.runtime_model import train_contract_type_pipeline

        with pytest.raises(ValueError, match="length mismatch"):
            train_contract_type_pipeline(["a", "b"], ["A"], random_state=7)


class TestEnsureRuntimeModelFallbacks:
    def test_returns_downloaded_tag_when_cache_misses(self, monkeypatch, tmp_path) -> None:
        from lexnlp.extract.en.contracts import runtime_model

        downloaded = tmp_path / "downloaded.skops"
        downloaded.write_bytes(b"x")

        import lexnlp.ml.catalog as catalog_mod

        monkeypatch.setattr(
            catalog_mod, "get_path_from_catalog", lambda tag: (_ for _ in ()).throw(FileNotFoundError(tag))
        )
        monkeypatch.setattr(runtime_model, "ensure_tag_downloaded", lambda tag: downloaded)
        result = runtime_model.ensure_runtime_contract_type_model(force=False)
        assert result == downloaded

    def test_falls_back_to_training_when_download_fails(self, monkeypatch, tmp_path) -> None:
        from lexnlp.extract.en.contracts import runtime_model

        trained = tmp_path / "trained.skops"
        trained.write_bytes(b"x")

        import lexnlp.ml.catalog as catalog_mod

        def raise_missing(tag: str):
            raise FileNotFoundError(tag)

        corpus_archive = tmp_path / "corpus.tar.gz"
        corpus_archive.write_bytes(b"corpus")

        def fake_download(tag: str):
            if tag == runtime_model.RUNTIME_CONTRACT_TYPE_TAG:
                raise RuntimeError("network down")
            assert tag == runtime_model.CONTRACT_TYPE_CORPUS_TAG
            return corpus_archive

        monkeypatch.setattr(catalog_mod, "get_path_from_catalog", raise_missing)
        monkeypatch.setattr(runtime_model, "ensure_tag_downloaded", fake_download)
        monkeypatch.setattr(
            runtime_model,
            "collect_contract_type_samples",
            lambda archive, *, max_docs_per_label, head_character_n: (["t1", "t2"], ["A", "B"], {"A": 1, "B": 1}),
        )
        monkeypatch.setattr(
            runtime_model,
            "train_contract_type_pipeline",
            lambda texts, labels, *, random_state: object(),
        )
        monkeypatch.setattr(
            runtime_model,
            "write_pipeline_to_catalog",
            lambda *, pipeline, target_tag, force: (trained, True),
        )
        result = runtime_model.ensure_runtime_contract_type_model(force=False)
        assert result == trained
