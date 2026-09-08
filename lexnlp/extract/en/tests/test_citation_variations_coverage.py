"""Coverage tests for lexnlp.extract.en.citation_variations missing lines."""

from lexnlp.extract.en import citation_variations
from lexnlp.extract.en.citation_variations import (
    _normalized_variation_index,
    normalize_reporter,
    variation_map,
)


def test_variation_map_handles_str_value():
    # All shipped VARIATIONS_ONLY values are lists, so the `str` branch
    # (lines 59-60) needs a patched raw map. Restore state afterwards.
    raw = citation_variations._VARIATIONS_RAW
    assert all(isinstance(v, list) for v in raw.values())
    citation_variations._VARIATIONS_RAW = {"FakeVariant": "FakeCanon"}
    variation_map.cache_clear()
    _normalized_variation_index.cache_clear()
    try:
        assert variation_map() == {"FakeVariant": ("FakeCanon",)}
        assert _normalized_variation_index() == {"FakeVariant": ("FakeCanon",)}
    finally:
        citation_variations._VARIATIONS_RAW = raw
        variation_map.cache_clear()
        _normalized_variation_index.cache_clear()


def test_variation_map_list_values_stay_tuples():
    vm = variation_map()
    assert len(vm) > 0
    key = next(iter(vm))
    assert isinstance(vm[key], tuple)
    assert len(vm[key]) > 0


def test_normalize_reporter_returns_canonical_member_unchanged():
    from reporters_db import EDITIONS

    name = next(iter(EDITIONS))
    # Line 110: a name already in EDITIONS is returned unchanged.
    assert normalize_reporter(name) == name
    assert isinstance(name, str) and len(name) > 0
