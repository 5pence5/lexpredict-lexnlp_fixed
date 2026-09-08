"""Coverage tests for lexnlp.extract.en.addresses.addresses."""

from __future__ import annotations

import pytest

import lexnlp.extract.en.addresses.addresses as addresses_mod
from lexnlp.extract.en.addresses.addresses import (
    Address,
    _safe_index,
    align_tokens,
    get_address_annotations,
    get_address_spans,
    get_addresses,
    load_classifier,
)

ADDRESS_TEXT = "Please send mail to 123 Main Street, Los Angeles, CA 90001 for processing of this matter."
EXPECTED_SPAN_TEXT = "123 Main Street, Los Angeles, CA"


class TestAddressValueObject:
    def test_init_collapses_matching_addr2_to_none(self) -> None:
        address = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "123 Main Street")
        assert address.zip_code == "90001"
        assert address.country == "USA"
        assert address.state == "CA"
        assert address.city == "Los Angeles"
        assert address.addr1 == "123 Main Street"
        assert address.addr2 is None

    def test_init_keeps_distinct_addr2(self) -> None:
        address = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")
        assert address.addr2 == "Apt 4"

    def test_str_format(self) -> None:
        address = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")
        assert str(address) == "123 Main Street, Apt 4, Los Angeles, CA, USA, 90001"

    def test_members_tuple(self) -> None:
        address = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")
        assert address.members() == ("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")

    def test_equality_and_hash(self) -> None:
        first = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")
        second = Address("90001", "USA", "CA", "Los Angeles", "123 Main Street", "Apt 4")
        other = Address("20500", "USA", "DC", "Washington", "1600 Pennsylvania Ave", "NW")
        assert first == second
        assert hash(first) == hash(second)
        assert len({first, second, other}) == 2
        assert first != other
        assert (first == "not an address") is False


class TestSafeIndex:
    def test_safe_miss_returns_none(self) -> None:
        assert _safe_index("hello world", "missing", 0, True) is None

    def test_unsafe_miss_raises_with_context(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            _safe_index("hello world", "missing", 0)


class TestAlignTokens:
    def test_plain_tokens_produce_exact_offsets(self) -> None:
        sentence = "hello world"
        assert align_tokens(["hello", "world"], sentence) == [(0, 5), (6, 11)]

    def test_nltk_quote_tokens_map_to_double_quote(self) -> None:
        sentence = '"hello"'
        assert align_tokens(["``", "hello", "''"], sentence) == [(0, 1), (1, 6), (6, 7)]

    def test_nltk_quote_tokens_fall_back_to_single_quote(self) -> None:
        sentence = "'hello'"
        assert align_tokens(["``", "hello", "''"], sentence) == [(0, 1), (1, 6), (6, 7)]

    def test_quote_tokens_without_any_quote_raise(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            align_tokens(["``"], "no quotes here")


class FakeBinaryFile:
    def __init__(self, payload: str = "fake-bytes") -> None:
        self.payload = payload

    def __enter__(self) -> str:
        return self.payload

    def __exit__(self, *args: object) -> bool:
        return False


class TestLoadClassifier:
    def test_pickle_fallback_uses_renamed_load(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(addresses_mod.os.path, "exists", lambda _path: False)
        monkeypatch.setattr(addresses_mod, "renamed_load", lambda _fh: sentinel)
        monkeypatch.setattr("builtins.open", lambda *args, **kwargs: FakeBinaryFile())
        assert load_classifier() is sentinel

    def test_pickle_fallback_reads_classifier_file(self, monkeypatch) -> None:
        seen: dict[str, object] = {}
        sentinel = object()

        def fake_open(path: str, mode: str) -> FakeBinaryFile:
            seen["path"] = path
            seen["mode"] = mode
            return FakeBinaryFile()

        def fake_load(fh: object) -> object:
            seen["fh"] = fh
            return sentinel

        monkeypatch.setattr(addresses_mod.os.path, "exists", lambda _path: False)
        monkeypatch.setattr("builtins.open", fake_open)
        monkeypatch.setattr(addresses_mod, "renamed_load", fake_load)
        assert load_classifier() is sentinel
        assert str(seen["path"]).endswith("addresses_clf.pickle")
        assert seen["mode"] == "rb"
        assert seen["fh"] == "fake-bytes"


class TestAddressGenerators:
    def test_get_addresses_yields_real_span_text(self) -> None:
        assert list(get_addresses(ADDRESS_TEXT)) == [EXPECTED_SPAN_TEXT]

    def test_get_address_spans_yields_text_and_offsets(self) -> None:
        spans = list(get_address_spans(ADDRESS_TEXT))
        assert spans == [(EXPECTED_SPAN_TEXT, 20, 52)]
        start, end = spans[0][1], spans[0][2]
        assert ADDRESS_TEXT[start:end] == EXPECTED_SPAN_TEXT

    def test_get_address_annotations_propagates_constructor_type_error(self) -> None:
        # The wrapper reaches its yield lines, but AddressAnnotation takes no
        # `name` keyword, so construction raises TypeError. Recorded here as
        # observed behaviour; the wrapper likely needs a source fix.
        with pytest.raises(TypeError, match="name"):
            list(get_address_annotations(ADDRESS_TEXT))
