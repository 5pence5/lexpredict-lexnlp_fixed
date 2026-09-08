"""Coverage tests for lexnlp.extract.common.text_beautifier missing lines."""

from __future__ import annotations

from unittest import TestCase

from lexnlp.extract.common.text_beautifier import TextBeautifier


class TestTextBeautifierCoverage(TestCase):
    def test_normalize_empty(self) -> None:
        assert TextBeautifier.normalize_smb_preserve_len("") == ""
        assert TextBeautifier.normalize_smb_preserve_len("“a”") == '"a"'

    def test_strip_pair_symbols_falsy(self) -> None:
        assert TextBeautifier.strip_pair_symbols("") == ""
        assert TextBeautifier.strip_pair_symbols(None) is None

    def test_strip_pair_symbols_whitespace_coords(self) -> None:
        assert TextBeautifier.strip_pair_symbols(("   ", 0, 3)) == ("", 3, 3)

    def test_unify_quotes_braces_exception(self) -> None:
        assert TextBeautifier.unify_quotes_braces(None) is None

    def test_unify_quotes_braces_coords_exception(self) -> None:
        assert TextBeautifier.unify_quotes_braces_coords(None, 0, 5) == (None, 0, 5)

    def test_find_transformed_word_far_returns_none(self) -> None:
        assert TextBeautifier.find_transformed_word('" hello', "''", 0) == ('"', 0)
        assert TextBeautifier.find_transformed_word('ab    " cd', "''", 0) is None
