"""Coverage tests for lexnlp/utils/unicode/unicode_lookup.py.

Covers _load_table's OSError paths and build_lookup_tables end to end
against the bundled UnicodeData.txt fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.utils.unicode import unicode_lookup
from lexnlp.utils.unicode.unicode_lookup import _load_table, build_lookup_tables

UNICODE_DATA = os.path.join(lexnlp_test_path, "lexnlp/utils/unicode_data.txt")


class TestLoadTableErrors:
    def test_missing_file_with_ignore_error_returns_none(self, tmp_path: Path, capsys) -> None:
        missing = str(tmp_path / "does-not-exist.pickle")
        assert _load_table(missing, True) is None
        assert "Unable to load unicode lookup table" in capsys.readouterr().out

    def test_missing_file_without_ignore_error_reraises(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "does-not-exist.pickle")
        with pytest.raises(OSError):
            _load_table(missing, False)

    def test_bundled_tables_loaded_at_import(self) -> None:
        assert unicode_lookup.UNICODE_CHAR_CATEGORIES is not None
        assert unicode_lookup.UNICODE_CHAR_CATEGORY_MAPPING is not None
        assert unicode_lookup.UNICODE_CHAR_TOP_CATEGORY_MAPPING is not None
        assert unicode_lookup.UNICODE_CHAR_CATEGORY_MAPPING["a"] == "Ll"


class TestBuildLookupTables:
    def test_builds_all_tables_from_unicode_data(self, tmp_path: Path) -> None:
        fn_categories = str(tmp_path / "categories.pickle")
        fn_mapping = str(tmp_path / "mapping.pickle")
        fn_top = str(tmp_path / "top.pickle")

        build_lookup_tables(fn_categories, fn_mapping, fn_top, table_source=UNICODE_DATA)

        categories = _load_table(fn_categories, False)
        assert "," in categories["punctuation"]
        assert "[" in categories["punctuation_start"]
        assert "]" in categories["punctuation_end"]
        assert "^" in categories["symbol"]
        assert "$" in categories["symbol_currency"]
        assert "+" in categories["symbol_math"]
        assert " " in categories["whitespace"]
        assert " " in categories["space"]
        assert " " in categories["line"]

        mapping = _load_table(fn_mapping, False)
        assert mapping["a"] == "Ll"
        assert mapping["A"] == "Lu"
        assert mapping["0"] == "Nd"

        top = _load_table(fn_top, False)
        assert top["a"] == "L"
        assert top["0"] == "N"
        assert top[" "] == "Z"
        assert top["$"] == "S"

    def test_output_files_are_non_empty_pickles(self, tmp_path: Path) -> None:
        outputs = [tmp_path / name for name in ("c.pickle", "m.pickle", "t.pickle")]
        build_lookup_tables(*(str(p) for p in outputs), table_source=UNICODE_DATA)
        for path in outputs:
            assert path.stat().st_size > 0

    def test_category_lists_cover_expected_groups(self, tmp_path: Path) -> None:
        fn_categories = str(tmp_path / "categories.pickle")
        fn_mapping = str(tmp_path / "mapping.pickle")
        fn_top = str(tmp_path / "top.pickle")
        build_lookup_tables(fn_categories, fn_mapping, fn_top, table_source=UNICODE_DATA)
        categories = _load_table(fn_categories, False)
        assert set(categories) == {
            "punctuation",
            "punctuation_start",
            "punctuation_end",
            "symbol",
            "symbol_currency",
            "symbol_math",
            "whitespace",
            "space",
            "line",
        }
        assert len(categories["punctuation"]) > len(categories["punctuation_start"])
        assert len(categories["whitespace"]) >= len(categories["space"])
