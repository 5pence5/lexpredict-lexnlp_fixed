"""Final-coverage tests for lexnlp.extract.en.addresses.address_features missing lines."""

from __future__ import annotations

import io
import json
import runpy
from unittest import mock

import nltk
import pytest

from lexnlp.extract.en.addresses.address_features import POS_TAG_SET_INDEX_FN


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_main_guard_rebuilds_pos_tagset_index() -> None:
    """Line 192: running the module as __main__ rebuilds the POS tagset index file."""
    buffer = io.StringIO()
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(POS_TAG_SET_INDEX_FN) and "w" in mode:
            return buffer
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(nltk.help, "load", return_value={"VB": "verb", "NN": "noun"}, create=True),
        mock.patch("builtins.open", fake_open),
    ):
        runpy.run_module("lexnlp.extract.en.addresses.address_features", run_name="__main__")
    assert json.loads(buffer.getvalue()) == {"NN": 1, "VB": 2}
