"""Coverage tests for uncovered branches in lexnlp.nlp.en.tokens."""

from __future__ import annotations

from lexnlp.nlp.en.tokens import get_tokens, get_tokens_by_regex


class TestGetTokensByRegexLowercase:
    def test_lowercase_true_lowercases_tokens(self) -> None:
        tokens = list(get_tokens_by_regex("Hello WORLD Foo", lowercase=True, preserve_line=True))
        assert tokens == ["hello", "world", "foo"]

    def test_lowercase_false_preserves_case(self) -> None:
        tokens = list(get_tokens_by_regex("Hello WORLD Foo", lowercase=False, preserve_line=True))
        assert tokens == ["Hello", "WORLD", "Foo"]


class TestGetTokensByRegexPreserveLineFalse:
    def test_period_only_token_is_dropped(self) -> None:
        kept = list(get_tokens_by_regex("Hello . World", lowercase=False, preserve_line=False))
        assert kept == ["Hello", "World"]

    def test_period_only_token_is_kept_when_preserve_line(self) -> None:
        kept = list(get_tokens_by_regex("Hello . World", lowercase=False, preserve_line=True))
        assert kept == ["Hello", ".", "World"]

    def test_all_periods_input_yields_nothing_without_preserve_line(self) -> None:
        assert list(get_tokens_by_regex("...", lowercase=False, preserve_line=False)) == []
        assert list(get_tokens_by_regex("...", lowercase=False, preserve_line=True)) == [".", ".", "."]

    def test_lowercase_and_no_preserve_line_combined(self) -> None:
        tokens = list(get_tokens_by_regex("Hello . WORLD", lowercase=True, preserve_line=False))
        assert tokens == ["hello", "world"]


class TestGetTokensStopwordLowercase:
    def test_stopword_with_lowercase(self) -> None:
        tokens = list(get_tokens("This is a Test.", stopword=True, lowercase=True))
        assert tokens == ["test", "."]

    def test_stopword_without_lowercase_preserves_case(self) -> None:
        tokens = list(get_tokens("This is a Test.", stopword=True, lowercase=False))
        assert tokens == ["Test", "."]

    def test_stopword_lowercase_without_stopwords_present(self) -> None:
        tokens = list(get_tokens("Hello World", stopword=True, lowercase=True))
        assert tokens == ["hello", "world"]
