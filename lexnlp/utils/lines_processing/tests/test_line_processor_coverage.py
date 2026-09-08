"""Coverage tests for lexnlp.utils.lines_processing.line_processor."""

from lexnlp.utils.lines_processing.line_processor import (
    LineOrPhrase,
    LineProcessor,
    SingleWord,
)


class TestLineOrPhraseCoverage:
    def test_repr_default_ending(self) -> None:
        assert repr(LineOrPhrase("abc", 10)) == "abc->"

    def test_repr_with_ending(self) -> None:
        phrase = LineOrPhrase("abc", 10)
        phrase.ending = "\n"
        assert repr(phrase) == "abc->\n"


class TestSingleWordCoverage:
    def test_get_end(self) -> None:
        assert SingleWord("hello", 5).get_end() == 10

    def test_get_end_defaults(self) -> None:
        assert SingleWord().get_end() == 0

    def test_repr(self) -> None:
        assert repr(SingleWord("hi", 3)) == "hi"


class TestDetermineLineLengthCoverage:
    def test_tab_counts_as_whitespace(self) -> None:
        proc = LineProcessor()
        proc.determine_line_length("xx" * 20 + "\t" + "yy" * 20 + "\n" + "z" * 80 + "\n")
        assert proc.line_length == 81
        assert proc.tail_length == int(81 * LineProcessor.line_tail_percent / 100)

    def test_empty_text_uses_default_length(self) -> None:
        proc = LineProcessor()
        proc.determine_line_length("")
        assert proc.line_length == LineProcessor.default_length
        assert proc.tail_length == int(LineProcessor.default_length * LineProcessor.line_tail_percent / 100)

    def test_single_char_uses_default_length(self) -> None:
        proc = LineProcessor()
        proc.determine_line_length("a")
        assert proc.line_length == LineProcessor.default_length


class TestWordsToLowercaseCoverage:
    def test_words_lowered_separators_kept(self) -> None:
        proc = LineProcessor()
        words = proc.split_text_on_words("Hello, WORLD")
        assert [(w.text, w.is_separator) for w in words] == [
            ("Hello", False),
            (", ", True),
            ("WORLD", False),
        ]
        proc.words_to_lowercase(words)
        assert [(w.text, w.is_separator) for w in words] == [
            ("hello", False),
            (", ", True),
            ("world", False),
        ]
