"""Coverage tests for lexnlp.extract.en.entities.stanford_ner."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import pytest

import lexnlp.extract.en.entities.stanford_ner as ner


@pytest.fixture(scope="module", autouse=True)
def _restore_stanford_ner_module():
    """Reload after this file so later Stanford tests see the real tagger init."""
    yield
    importlib.reload(ner)


class _FakeTagger:
    def __init__(self, tags: Sequence[tuple[str, str]] | None = None) -> None:
        self.tags = list(tags or [])
        self.calls: list[list[str]] = []

    def tag(self, tokens: Sequence[str]) -> list[tuple[str, str]]:
        self.calls.append(list(tokens))
        return list(self.tags)


@pytest.fixture
def patched_ner(monkeypatch: pytest.MonkeyPatch) -> tuple[object, _FakeTagger]:
    tagger = _FakeTagger()
    monkeypatch.setattr(ner, "STANFORD_NER_TAGGER", tagger)
    monkeypatch.setattr(ner, "get_tokens_list", lambda text: [token for token, _label in tagger.tags])
    return ner, tagger


class TestGetModelFile:
    def test_english_and_english7_paths(self) -> None:
        english = ner.get_model_file("english")
        english7 = ner.get_model_file("english7")
        assert english.endswith("english.all.3class.distsim.crf.ser.gz")
        assert english7.endswith("english.muc.7class.distsim.crf.ser.gz")
        assert "classifiers" in english
        assert english.startswith(ner.STANFORD_NER_PATH) or ner.STANFORD_NER_PATH in english

    def test_unknown_language_raises(self) -> None:
        with pytest.raises(KeyError):
            ner.get_model_file("klingon")


class TestTaggerInit:
    def test_lookup_error_leaves_tagger_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Boom:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise LookupError("stanford jar missing")

        monkeypatch.setattr("nltk.tag.StanfordNERTagger", Boom)
        reloaded = importlib.reload(ner)
        assert reloaded.STANFORD_NER_TAGGER is None
        assert reloaded.STANFORD_NER_FILE.endswith("stanford-ner.jar")
        assert "english" in reloaded.STANFORD_NER_MODEL_MAP

    def test_successful_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: dict[str, object] = {}

        class Fake:
            def __init__(self, model_file: str, jar: str, encoding: str = "utf-8") -> None:
                created["model"] = model_file
                created["jar"] = jar
                created["encoding"] = encoding

        monkeypatch.setattr("nltk.tag.StanfordNERTagger", Fake)
        reloaded = importlib.reload(ner)
        assert reloaded.STANFORD_NER_TAGGER is not None
        assert created["encoding"] == "utf-8"
        assert created["jar"] == reloaded.STANFORD_NER_FILE
        assert str(created["model"]).endswith("english.all.3class.distsim.crf.ser.gz")


class TestGetPersons:
    def test_joins_adjacent_person_tokens(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("John", "PERSON"), ("Smith", "PERSON"), ("works", "O")]
        names = list(module.get_persons("John Smith works here."))
        assert names == ["John Smith"]
        assert tagger.calls

    def test_strict_keeps_person_tokens_separate(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("John", "PERSON"), ("Smith", "PERSON")]
        assert list(module.get_persons("John Smith is here.", strict=True)) == ["John", "Smith"]

    def test_punctuation_is_joined_inside_window(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("John", "PERSON"), (",", "O"), ("Jr", "PERSON")]
        assert list(module.get_persons("John, Jr is here.")) == ["John, Jr"]

    def test_window_gap_starts_new_name(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("John", "PERSON"), ("the", "O"), ("Smith", "PERSON")]
        assert list(module.get_persons("John the Smith is here.", window=2)) == ["John", "Smith"]

    def test_return_source_includes_sentence(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        text = "Alice Johnson works here."
        tagger.tags = [("Alice", "PERSON"), ("Johnson", "PERSON")]
        rows = list(module.get_persons(text, return_source=True))
        assert len(rows) == 1
        name, sentence = rows[0]
        assert name == "Alice Johnson"
        assert "Alice Johnson" in sentence or sentence.endswith(".")

    def test_strips_trailing_punctuation(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Alice", "PERSON"), (".", "O")]
        assert list(module.get_persons("Alice.")) == ["Alice"]

    def test_empty_when_no_person_tags(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Hello", "O"), ("world", "O")]
        assert list(module.get_persons("Hello world.")) == []

    def test_empty_text_yields_nothing(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Alice", "PERSON")]
        assert list(module.get_persons("")) == []


class TestGetOrganizations:
    def test_joins_adjacent_org_tokens(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Acme", "ORGANIZATION"), ("Corp", "ORGANIZATION")]
        assert list(module.get_organizations("Acme Corp is here.")) == ["Acme Corp"]

    def test_strict_and_punctuation_and_source(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Acme", "ORGANIZATION"), (",", "O"), ("Inc", "ORGANIZATION")]
        assert list(module.get_organizations("Acme, Inc is here.", strict=True)) == ["Acme", "Inc"]
        tagger.tags = [("Acme", "ORGANIZATION"), (",", "O"), ("Inc", "ORGANIZATION")]
        joined = list(module.get_organizations("Acme, Inc is here."))
        assert joined == ["Acme, Inc"]
        sourced = list(module.get_organizations("Acme, Inc is here.", return_source=True))
        assert sourced[0][0] == "Acme, Inc"
        assert isinstance(sourced[0][1], str)

    def test_non_org_tokens_are_ignored(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("the", "O"), ("Acme", "ORGANIZATION")]
        assert list(module.get_organizations("the Acme is here.")) == ["Acme"]


class TestGetLocations:
    def test_joins_with_space_and_apostrophe(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("New", "LOCATION"), ("York", "LOCATION")]
        assert list(module.get_locations("New York is here.")) == ["New York"]
        tagger.tags = [("London", "LOCATION"), ("'s", "LOCATION")]
        assert list(module.get_locations("London's is here.")) == ["London's"]

    def test_strict_splits_and_return_source(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("New", "LOCATION"), ("York", "LOCATION")]
        assert list(module.get_locations("New York is here.", strict=True)) == ["New", "York"]
        tagger.tags = [("Boston", "LOCATION")]
        sourced = list(module.get_locations("Boston is here.", return_source=True))
        assert sourced[0][0] == "Boston"
        assert "Boston" in sourced[0][1] or sourced[0][1].endswith(".")

    def test_punctuation_inside_window(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Washington", "LOCATION"), (",", "O"), ("D", "LOCATION")]
        assert list(module.get_locations("Washington, D is here.")) == ["Washington, D"]

    def test_gap_outside_window_starts_new_location(self, patched_ner: tuple[object, _FakeTagger]) -> None:
        module, tagger = patched_ner
        tagger.tags = [("Paris", "LOCATION"), ("and", "O"), ("Rome", "LOCATION")]
        assert list(module.get_locations("Paris and Rome are here.", window=2)) == ["Paris", "Rome"]
