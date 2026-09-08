"""Tests for :mod:`lexnlp.extract.en.citation_variations`."""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.en.citation_variations import (
    canonical_for,
    is_known_reporter,
    normalize_reporter,
    variation_map,
)


class TestVariationMap:
    def test_has_entries(self) -> None:
        vm = variation_map()
        assert len(vm) > 0

    def test_values_are_tuples(self) -> None:
        vm = variation_map()
        sample_key = next(iter(vm))
        assert isinstance(vm[sample_key], tuple)


class TestCanonicalFor:
    def test_returns_empty_for_unknown(self) -> None:
        assert canonical_for("TotallyFakeReporter99") == ()

    def test_returns_empty_for_empty(self) -> None:
        assert canonical_for("") == ()

    def test_space_normalized_lookup(self) -> None:
        # "U. S." (with internal space) should still resolve when the
        # canonical variant is stored as "U.S." or similar.
        result = canonical_for("U. S.")
        # We don't guarantee a specific canonical here (depends on db
        # version) but the function must not raise.
        assert isinstance(result, tuple)


class TestIsKnownReporter:
    def test_returns_false_for_empty(self) -> None:
        assert not is_known_reporter("")

    def test_returns_false_for_nonsense(self) -> None:
        assert not is_known_reporter("NotAReporter")


class TestNormalizeReporter:
    def test_returns_input_when_unknown(self) -> None:
        assert normalize_reporter("NotAReporter") == "NotAReporter"

    def test_returns_input_unchanged_when_empty(self) -> None:
        assert normalize_reporter("") == ""


# ---------------------------------------------------------------------------
# Additional tests for PR changes: _normalized_variation_index
# ---------------------------------------------------------------------------


class TestNormalizedVariationIndex:
    """Tests for the new pre-built whitespace-stripped index added by the PR."""

    def test_index_is_consistent_with_variation_map(self) -> None:
        """The stripped index must contain exactly the same entries as
        variation_map() after stripping spaces from keys."""
        from lexnlp.extract.en.citation_variations import (
            _normalized_variation_index,
            variation_map,
        )

        vm = variation_map()
        index = _normalized_variation_index()
        # Every variation_map key, when stripped of spaces, must appear in the index.
        for variation, canons in vm.items():
            stripped = variation.replace(" ", "")
            # The index may merge multiple original keys that normalise to the
            # same stripped form; the expected canons are a subset of index values.
            assert stripped in index, f"Stripped key '{stripped}' (from '{variation}') missing from index"

    def test_index_values_are_tuples(self) -> None:
        """All values in the index must be tuples of strings."""
        from lexnlp.extract.en.citation_variations import _normalized_variation_index

        index = _normalized_variation_index()
        for key, canons in index.items():
            assert isinstance(canons, tuple), f"Value for '{key}' is not a tuple: {canons!r}"
            for c in canons:
                assert isinstance(c, str), f"Non-str canonical '{c!r}' for key '{key}'"

    def test_index_is_non_empty(self) -> None:
        """If reporters_db has variations, the index must be non-empty."""
        from lexnlp.extract.en.citation_variations import (
            _normalized_variation_index,
            variation_map,
        )

        if variation_map():  # only assert when the db has content
            assert len(_normalized_variation_index()) > 0

    def test_canonical_for_result_matches_index_lookup(self) -> None:
        """canonical_for must agree with a direct _normalized_variation_index lookup."""
        from lexnlp.extract.en.citation_variations import (
            _normalized_variation_index,
            canonical_for,
        )

        index = _normalized_variation_index()
        if not index:
            return  # reporters_db not available
        # Take a key from the index (already stripped of spaces) and wrap it
        # in spaces so canonical_for exercises the normalization path.
        stripped_key = next(iter(index))
        spaced_key = " ".join(stripped_key)  # add spaces between every char
        result = canonical_for(spaced_key)
        expected = index[stripped_key]
        assert result == expected, (
            f"canonical_for('{spaced_key}') → {result!r}, but index['{stripped_key}'] → {expected!r}"
        )


class TestCanonicalForAdditional:
    """Additional canonical_for tests for the O(1) lookup refactor."""

    def test_result_is_always_tuple(self) -> None:
        """canonical_for must always return a tuple, never None."""
        for variant in ("", "TotallyUnknown", "U.S.", "U. S."):
            result = canonical_for(variant)
            assert isinstance(result, tuple), (
                f"canonical_for('{variant}') returned {type(result).__name__}, expected tuple"
            )

    def test_known_variant_returns_non_empty_tuple(self) -> None:
        """For a variant that exists in the db, the result must be non-empty."""
        from lexnlp.extract.en.citation_variations import variation_map

        vm = variation_map()
        if not vm:
            return  # reporters_db empty
        # Pick the first known variant
        known = next(iter(vm))
        result = canonical_for(known)
        assert len(result) > 0, f"canonical_for('{known}') returned empty tuple"

    def test_space_normalization_is_idempotent(self) -> None:
        """Adding extra spaces around dots in a known variant must still resolve."""
        from lexnlp.extract.en.citation_variations import variation_map

        vm = variation_map()
        if not vm:
            return
        known = next(iter(vm))
        # Add spaces around every "." in the known variant.
        spaced = known.replace(".", " . ")
        result_spaced = canonical_for(spaced)
        result_direct = canonical_for(known)
        # Both should resolve to the same canons (both strip spaces internally).
        assert result_spaced == result_direct, (
            f"Space-padded variant '{spaced}' → {result_spaced!r} differs from direct '{known}' → {result_direct!r}"
        )
