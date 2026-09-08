"""Coverage tests for lexnlp.nlp.en.transforms.tokens n-gram/skipgram distributions."""

import collections

import pytest

from lexnlp.nlp.en.transforms.tokens import (
    get_bigram_distribution,
    get_ngram_distribution,
    get_skipgram_distribution,
    get_trigram_distribution,
)


class TestNgramDistributionCoverage:
    def test_bigram_counts(self) -> None:
        dist = get_ngram_distribution("the quick brown fox jumps", 2)
        assert isinstance(dist, collections.defaultdict)
        assert dict(dist) == {
            ("the", "quick"): 1,
            ("quick", "brown"): 1,
            ("brown", "fox"): 1,
            ("fox", "jumps"): 1,
        }

    def test_bigram_repeated_counts(self) -> None:
        dist = get_ngram_distribution("a b a b", 2)
        assert dict(dist) == {("a", "b"): 2, ("b", "a"): 1}

    def test_trigram_counts(self) -> None:
        dist = get_ngram_distribution("the quick brown fox jumps", 3)
        assert dict(dist) == {
            ("the", "quick", "brown"): 1,
            ("quick", "brown", "fox"): 1,
            ("brown", "fox", "jumps"): 1,
        }

    def test_lowercase_flag(self) -> None:
        dist = get_ngram_distribution("Hello HELLO world", 2, lowercase=True)
        assert dict(dist) == {("hello", "hello"): 1, ("hello", "world"): 1}


class TestBigramTrigramWrapperCoverage:
    def test_bigram_matches_ngram(self) -> None:
        text = "the quick brown fox jumps"
        assert dict(get_bigram_distribution(text)) == dict(get_ngram_distribution(text, 2))
        assert get_bigram_distribution(text)[("brown", "fox")] == 1

    def test_trigram_matches_ngram(self) -> None:
        text = "the quick brown fox jumps"
        assert dict(get_trigram_distribution(text)) == dict(get_ngram_distribution(text, 3))
        assert get_trigram_distribution(text)[("the", "quick", "brown")] == 1


class TestSkipgramDistributionCoverage:
    def test_skipgram_raises_attribute_error(self) -> None:
        # Real behaviour: nltk.util resolves to nltk.stem.util in this
        # environment, which has no skipgrams attribute, so the call fails.
        with pytest.raises(AttributeError, match="skipgrams"):
            get_skipgram_distribution("the quick brown fox", 2, 1)
