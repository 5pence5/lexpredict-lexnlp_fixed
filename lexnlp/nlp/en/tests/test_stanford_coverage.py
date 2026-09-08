"""Coverage tests for lexnlp.nlp.en.stanford (no Java/Stanford jars required)."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import pytest

import lexnlp.nlp.en.stanford as stanford


@pytest.fixture(scope="module", autouse=True)
def _restore_stanford_module():
    """Reload after this file so later Stanford tests see the real import-time init."""
    yield
    importlib.reload(stanford)


class FakeTokenizer:
    def __init__(self, path_to_jar: str | None = None) -> None:
        self.path_to_jar = path_to_jar

    def tokenize(self, text: str) -> list[str]:
        return [part for part in text.split() if part]


class FakeTagger:
    def __init__(self, model: str | None = None, jar: str | None = None) -> None:
        self.model = model
        self.jar = jar
        self.calls: list[list[str]] = []

    def tag(self, tokens: Sequence[str]) -> list[tuple[str, str]]:
        self.calls.append(list(tokens))
        tagged: list[tuple[str, str]] = []
        for token in tokens:
            low = token.lower()
            if low in {"run", "runs", "ran", "eat", "eats", "jump"}:
                tagged.append((token, "VBZ"))
            elif low in {"cat", "cats", "dog", "dogs", "court", "courts"}:
                tagged.append((token, "NNS"))
            else:
                tagged.append((token, "DT"))
        return tagged


@pytest.fixture
def enabled_stanford(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stanford, "is_stanford_enabled", lambda: True)
    monkeypatch.setattr(stanford, "STANFORD_TOKENIZER", FakeTokenizer())
    monkeypatch.setattr(stanford, "STANFORD_TAGGER", FakeTagger())


class TestCheckStanford:
    def test_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stanford, "is_stanford_enabled", lambda: False)
        with pytest.raises(RuntimeError, match="USE_STANFORD is set to False"):
            stanford.check_stanford()

    def test_missing_tokenizer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stanford, "is_stanford_enabled", lambda: True)
        monkeypatch.setattr(stanford, "STANFORD_TOKENIZER", None)
        monkeypatch.setattr(stanford, "STANFORD_TAGGER", None)
        with pytest.raises(RuntimeError, match="POS tagger jar file is not found"):
            stanford.check_stanford()

    def test_missing_tagger_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stanford, "is_stanford_enabled", lambda: True)
        monkeypatch.setattr(stanford, "STANFORD_TOKENIZER", FakeTokenizer())
        monkeypatch.setattr(stanford, "STANFORD_TAGGER", None)
        with pytest.raises(RuntimeError, match="tagger model file is not found"):
            stanford.check_stanford()

    def test_ready_components_pass(self, enabled_stanford: None) -> None:
        stanford.check_stanford()


class TestImportTimeInit:
    def test_successful_construction_sets_tokenizer_and_tagger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nltk.tag
        import nltk.tokenize.stanford as nltk_stanford_tok

        monkeypatch.setattr(nltk_stanford_tok, "StanfordTokenizer", FakeTokenizer)
        monkeypatch.setattr(nltk.tag, "StanfordPOSTagger", FakeTagger)
        reloaded = importlib.reload(stanford)
        assert isinstance(reloaded.STANFORD_TOKENIZER, FakeTokenizer)
        assert isinstance(reloaded.STANFORD_TAGGER, FakeTagger)
        assert reloaded.STANFORD_POS_FILE.endswith("stanford-postagger.jar")
        assert reloaded.STANFORD_DEFAULT_TAG_MODEL.endswith("english-bidirectional-distsim.tagger")
        assert reloaded.STANFORD_TOKENIZER.path_to_jar == reloaded.STANFORD_POS_FILE
        assert reloaded.STANFORD_TAGGER.model == reloaded.STANFORD_DEFAULT_TAG_MODEL
        assert reloaded.STANFORD_TAGGER.jar == reloaded.STANFORD_POS_FILE

    def test_lookup_error_leaves_tokenizer_and_tagger_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nltk.tag
        import nltk.tokenize.stanford as nltk_stanford_tok

        class Boom:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise LookupError("stanford jar missing")

        monkeypatch.setattr(nltk_stanford_tok, "StanfordTokenizer", Boom)
        monkeypatch.setattr(nltk.tag, "StanfordPOSTagger", Boom)
        reloaded = importlib.reload(stanford)
        assert reloaded.STANFORD_TOKENIZER is None
        assert reloaded.STANFORD_TAGGER is None


class TestGetTokens:
    def test_preserves_case_without_stopword_filter(self, enabled_stanford: None) -> None:
        tokens = stanford.get_tokens_list("The Cat Run", lowercase=False, stopword=False)
        assert tokens == ["The", "Cat", "Run"]

    def test_lowercase_without_stopword_filter(self, enabled_stanford: None) -> None:
        tokens = stanford.get_tokens_list("The Cat Run", lowercase=True, stopword=False)
        assert tokens == ["the", "cat", "run"]

    def test_stopword_filter_keeps_content_words(self, enabled_stanford: None) -> None:
        tokens = stanford.get_tokens_list("The Cat Run", lowercase=False, stopword=True)
        assert "The" not in tokens
        assert tokens == ["Cat", "Run"]

    def test_stopword_filter_with_lowercase(self, enabled_stanford: None) -> None:
        tokens = stanford.get_tokens_list("The Cat Run", lowercase=True, stopword=True)
        assert tokens == ["cat", "run"]

    def test_stopword_only_sentence_yields_empty(self, enabled_stanford: None) -> None:
        tokens = stanford.get_tokens_list("The the THE", lowercase=False, stopword=True)
        assert tokens == []


class TestGetVerbs:
    def test_original_case(self, enabled_stanford: None) -> None:
        verbs = list(stanford.get_verbs("The cats RUN", lowercase=False, lemmatize=False))
        assert verbs == ["RUN"]

    def test_lowercase(self, enabled_stanford: None) -> None:
        verbs = list(stanford.get_verbs("The cats RUN", lowercase=True, lemmatize=False))
        assert verbs == ["run"]

    def test_lemmatize_uses_lemma_list(
        self, enabled_stanford: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            stanford,
            "get_lemma_list",
            lambda text, lowercase=False: ["the", "cat", "run"],
        )
        verbs = list(stanford.get_verbs("The cats RUN", lowercase=True, lemmatize=True))
        assert verbs == ["run"]

    def test_no_verbs_yields_empty(self, enabled_stanford: None) -> None:
        assert list(stanford.get_verbs("The cat", lowercase=False, lemmatize=False)) == []


class TestGetNouns:
    def test_original_case(self, enabled_stanford: None) -> None:
        nouns = list(stanford.get_nouns("The CATS run", lowercase=False, lemmatize=False))
        assert nouns == ["CATS"]

    def test_lowercase(self, enabled_stanford: None) -> None:
        nouns = list(stanford.get_nouns("The CATS run", lowercase=True, lemmatize=False))
        assert nouns == ["cats"]

    def test_lemmatize_uses_lemma_list(
        self, enabled_stanford: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            stanford,
            "get_lemma_list",
            lambda text, lowercase=False: ["the", "cat", "run"],
        )
        nouns = list(stanford.get_nouns("The CATS run", lowercase=False, lemmatize=True))
        assert nouns == ["cat"]

    def test_no_nouns_yields_empty(self, enabled_stanford: None) -> None:
        assert list(stanford.get_nouns("The run", lowercase=False, lemmatize=False)) == []
