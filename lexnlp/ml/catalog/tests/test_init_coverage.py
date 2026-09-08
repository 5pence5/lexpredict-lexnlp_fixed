"""Coverage tests for lexnlp.ml.catalog.__init__ missing lines."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestCatalogInitCoverage:
    def test_resolve_skips_non_directory_candidate(self, tmp_path, monkeypatch) -> None:
        import nltk.data

        import lexnlp.ml.catalog as catalog

        busy_file = tmp_path / "afile"
        busy_file.write_bytes(b"x")
        monkeypatch.setattr(nltk.data, "path", [str(busy_file)])
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        assert catalog._resolve_nltk_data_dir() == tmp_path / "home" / "nltk_data"

    def test_resolve_falls_back_to_cwd(self, monkeypatch) -> None:
        import nltk.data

        import lexnlp.ml.catalog as catalog

        monkeypatch.setattr(nltk.data, "path", [])
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/definitely/missing/home")))
        assert catalog._resolve_nltk_data_dir() == Path.cwd() / "nltk_data"

    def test_get_path_refreshes_cache_on_miss(self, tmp_path) -> None:
        import lexnlp.ml.catalog as catalog

        original = catalog.CATALOG
        catalog.CATALOG = tmp_path
        try:
            first_dir = tmp_path / "pipeline" / "x" / "0.1"
            first_dir.mkdir(parents=True)
            (first_dir / "model.pkl").write_bytes(b"m")
            catalog.invalidate_catalog_cache()
            assert catalog.get_path_from_catalog("pipeline/x/0.1") == (first_dir / "model.pkl")
            # Added after the cache was built: miss triggers a refresh.
            second_dir = tmp_path / "pipeline" / "y" / "0.2"
            second_dir.mkdir(parents=True)
            (second_dir / "m.pkl").write_bytes(b"m")
            assert catalog.get_path_from_catalog("pipeline/y/0.2") == (second_dir / "m.pkl")
        finally:
            catalog.CATALOG = original
            catalog.invalidate_catalog_cache()

    def test_get_path_refreshes_stale_entry_then_raises(self, tmp_path) -> None:
        import lexnlp.ml.catalog as catalog

        original = catalog.CATALOG
        catalog.CATALOG = tmp_path
        try:
            tag_dir = tmp_path / "pipeline" / "z" / "0.3"
            tag_dir.mkdir(parents=True)
            model = tag_dir / "m.pkl"
            model.write_bytes(b"m")
            catalog.invalidate_catalog_cache()
            assert catalog.get_path_from_catalog("pipeline/z/0.3") == model
            model.unlink()
            with pytest.raises(FileNotFoundError):
                catalog.get_path_from_catalog("pipeline/z/0.3")
        finally:
            catalog.CATALOG = original
            catalog.invalidate_catalog_cache()

    def test_get_path_missing_tag_raises(self, tmp_path) -> None:
        import lexnlp.ml.catalog as catalog

        original = catalog.CATALOG
        catalog.CATALOG = tmp_path
        try:
            catalog.invalidate_catalog_cache()
            with pytest.raises(FileNotFoundError):
                catalog.get_path_from_catalog("nope/none/0")
        finally:
            catalog.CATALOG = original
            catalog.invalidate_catalog_cache()
