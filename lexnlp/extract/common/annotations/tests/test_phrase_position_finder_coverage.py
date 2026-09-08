"""Coverage tests for the early-exit branch in PhrasePositionFinder."""

from __future__ import annotations

from lexnlp.extract.common.annotations.phrase_position_finder import PhrasePositionFinder


class TestPhrasePositionFinderExhaustion:
    def test_break_when_start_exhausts_text(self) -> None:
        result = PhrasePositionFinder.find_phrase_in_source_text("ab", ["ab", "cd"])
        assert result[0] == ("ab", 0, 2)
        # Second phrase is never searched: the loop breaks and the
        # (phrase, 0, 0) sentinel is left untouched.
        assert result[1] == ("cd", 0, 0)

    def test_trailing_phrase_unreached_after_full_consumption(self) -> None:
        result = PhrasePositionFinder.find_phrase_in_source_text("hello world", ["hello", "world", "extra"])
        assert result[0] == ("hello", 0, 5)
        assert result[1] == ("world", 6, 11)
        assert result[2] == ("extra", 0, 0)
