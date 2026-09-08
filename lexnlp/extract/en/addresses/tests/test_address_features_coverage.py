"""Coverage tests for address feature helpers."""

from __future__ import annotations

import io
import json
from unittest import mock

import nltk

from lexnlp.extract.en.addresses.address_features import (
    FEATURE_WORD_LEN,
    POS_TAG_SET_INDEX_FN,
    ZERO_FEATURES,
    _load_set_from_lines,
    get_word_features,
    is_datetime,
    is_email,
    is_url,
    prepare_pos_tagset_index_file,
)


def test_load_set_without_normalize_preserves_case() -> None:
    raw = _load_set_from_lines("provinces.txt")
    normalized = _load_set_from_lines("provinces.txt", normalize=True)
    assert isinstance(raw, set)
    assert len(raw) > 0
    assert all(isinstance(item, str) for item in raw)
    assert {item.upper() for item in raw} == normalized


def test_is_datetime_rejects_out_of_range_years() -> None:
    assert is_datetime("15 June 1500") is False
    assert is_datetime("15 June 2500") is False


def test_is_datetime_accepts_modern_date() -> None:
    assert is_datetime("15 June 2020") is True


def test_is_url() -> None:
    assert is_url("https://example.com/path?q=1") is True
    assert is_url("not a url") is False


def test_is_email() -> None:
    assert bool(is_email("user@example.com")) is True
    assert bool(is_email("no-at-sign-here")) is False


def test_get_word_features_empty_word_returns_zero_vector() -> None:
    features = get_word_features("", "NN")
    assert features == [0] * FEATURE_WORD_LEN
    assert features == ZERO_FEATURES
    assert len(features) == 21


def test_prepare_pos_tagset_index_file_writes_sorted_index() -> None:
    buffer = io.StringIO()
    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(POS_TAG_SET_INDEX_FN) and "w" in mode:
            return buffer
        return real_open(path, mode, *args, **kwargs)

    # NOTE: with the installed nltk version, `nltk.help` has no `load`
    # attribute (the working call is `nltk.data.load`), so the real
    # function raises AttributeError before line 187. The shim below only
    # stands in for that external resource load; the sorted-index build and
    # json.dump on lines 187-188 execute for real. See report Notes.
    with (
        mock.patch.object(nltk.help, "load", return_value={"VB": "verb", "NN": "noun"}, create=True),
        mock.patch("builtins.open", fake_open),
    ):
        prepare_pos_tagset_index_file()
    assert json.loads(buffer.getvalue()) == {"NN": 1, "VB": 2}
