"""Final-coverage tests for lexnlp.utils.unicode.unicode_lookup missing lines."""

from __future__ import annotations

import io
import os
import pickle
import runpy
from unittest import mock

import pytest

from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.utils.unicode import unicode_lookup

UNICODE_DATA = os.path.join(lexnlp_test_path, "lexnlp/utils/unicode_data.txt")
FTP_URL = "ftp://ftp.unicode.org/Public/11.0.0/ucd/UnicodeData.txt"


class KeepOpenBytesIO(io.BytesIO):
    """BytesIO that survives the `with open(...)` close in build_lookup_tables."""

    def close(self) -> None:
        super().flush()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_main_guard_builds_and_saves_lookup_tables(capsys) -> None:
    """Lines 196-201: running the module as __main__ builds tables from UnicodeData.txt."""
    targets = {
        unicode_lookup._FN_UNICODE_CHAR_CATEGORIES,
        unicode_lookup._FN_UNICODE_CHAR_CATEGORY_MAPPING,
        unicode_lookup._FN_UNICODE_CHAR_TOP_CATEGORY_MAPPING,
    }
    buffers: dict[str, KeepOpenBytesIO] = {}
    real_open = open
    seen_urls: list[str] = []

    import pandas

    real_read_csv = pandas.read_csv

    def fake_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) in targets and "w" in mode:
            buffers[str(path)] = KeepOpenBytesIO()
            return buffers[str(path)]
        return real_open(path, mode, *args, **kwargs)

    def fake_read_csv(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_urls.append(url)
        return real_read_csv(UNICODE_DATA, *args, **kwargs)

    with (
        mock.patch("builtins.open", fake_open),
        mock.patch("pandas.read_csv", fake_read_csv),
    ):
        runpy.run_module("lexnlp.utils.unicode.unicode_lookup", run_name="__main__")

    assert seen_urls == [FTP_URL]
    assert set(buffers) == targets
    categories = pickle.loads(buffers[unicode_lookup._FN_UNICODE_CHAR_CATEGORIES].getvalue())
    assert "," in categories["punctuation"]
    assert "[" in categories["punctuation_start"]
    mapping = pickle.loads(buffers[unicode_lookup._FN_UNICODE_CHAR_CATEGORY_MAPPING].getvalue())
    assert mapping["a"] == "Ll"
    top = pickle.loads(buffers[unicode_lookup._FN_UNICODE_CHAR_TOP_CATEGORY_MAPPING].getvalue())
    assert top["a"] == "L"
    out = capsys.readouterr().out
    assert "Building and saving unicode lookup tables..." in out
    assert "Done" in out
