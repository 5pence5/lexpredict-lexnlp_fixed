"""Coverage tests for the deprecated pt _get_courts helper and list wrapper."""

import warnings

import pytest

from lexnlp.extract.common.annotations.court_annotation import CourtAnnotation
from lexnlp.extract.en.dict_entities import DictionaryEntry
from lexnlp.extract.pt.courts import _get_courts, get_court_annotation_list, get_court_annotations


def _entries() -> list[DictionaryEntry]:
    return [DictionaryEntry(id=1, name="Supremo Tribunal Federal")]


def test_get_courts_warns_and_yields_match() -> None:
    text = "A decisão do Supremo Tribunal Federal foi publicada ontem."
    with pytest.warns(DeprecationWarning, match="removed in a future version"):
        result = list(_get_courts(text, _entries()))
    assert len(result) == 1
    entry, alias = result[0]
    assert entry.name == "Supremo Tribunal Federal"
    assert entry.id == 1
    assert alias.alias == "Supremo Tribunal Federal"


def test_get_courts_no_match_yields_nothing() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert list(_get_courts("nothing here", _entries())) == []


def test_get_courts_priority_and_language_flags() -> None:
    text = "A decisão do Supremo Tribunal Federal foi publicada ontem."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        plain = list(_get_courts(text, _entries()))
        flagged = list(
            _get_courts(
                text,
                _entries(),
                priority=True,
                text_languages=["pt"],
                simplified_normalization=True,
            )
        )
    assert len(plain) == 1
    assert len(flagged) == 1
    assert flagged[0][0].name == plain[0][0].name


def test_get_court_annotation_list() -> None:
    text = "A decisão do Supremo Tribunal Federal foi publicada ontem."
    result = get_court_annotation_list(text)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], CourtAnnotation)
    assert result[0].name == "Supremo Tribunal Federal"
    assert result[0].locale == "pt"
    assert result[0].coords == (12, 37)
    gen_names = [a.name for a in get_court_annotations(text)]
    assert [a.name for a in result] == gen_names
    assert get_court_annotation_list("") == []
    assert get_court_annotation_list("texto sem tribunal mencionado") == []
