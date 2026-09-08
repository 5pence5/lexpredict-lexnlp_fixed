"""Coverage tests for geoentity list-wrapper helpers."""

import os

from lexnlp.extract.common.annotations.geo_annotation import GeoAnnotation
from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.extract.en.dict_entities import DictionaryEntry
from lexnlp.extract.en.geoentities import (
    get_geoentities,
    get_geoentity_annotation_list,
    get_geoentity_annotations,
    get_geoentity_list,
)


def _load_config() -> list[DictionaryEntry]:
    base_path = os.path.join(lexnlp_test_path, "lexnlp/extract/en/tests/test_geoentities")
    entities_fn = os.path.join(base_path, "geoentities.csv")
    aliases_fn = os.path.join(base_path, "geoaliases.csv")
    return list(DictionaryEntry.load_entities_from_files(entities_fn, aliases_fn))


_CONFIG = _load_config()


def test_get_geoentity_list_returns_matches():
    text = "I live in France."
    expected = list(get_geoentities(text, geo_config_list=_CONFIG))
    actual = get_geoentity_list(text, geo_config_list=_CONFIG)
    assert isinstance(actual, list)
    assert [(entry.name, alias.alias) for entry, alias in actual] == [
        (entry.name, alias.alias) for entry, alias in expected
    ]
    assert [(entry.name, alias.alias) for entry, alias in actual] == [("France", "France")]


def test_get_geoentity_list_empty_without_match():
    assert get_geoentity_list("no geo xyzqqq", geo_config_list=_CONFIG) == []


def test_get_geoentity_annotation_list_returns_annotations():
    text = "I live in France."
    expected = list(get_geoentity_annotations(text, geo_config_list=_CONFIG))
    actual = get_geoentity_annotation_list(text, geo_config_list=_CONFIG)
    assert isinstance(actual, list)
    assert len(actual) == len(expected) == 1
    assert all(isinstance(item, GeoAnnotation) for item in actual)
    annotation = actual[0]
    assert annotation.coords == (10, 16)
    assert annotation.name == "France"
    assert annotation.alias == "France"
    assert annotation.locale == "en"


def test_get_geoentity_annotation_list_empty_without_match():
    assert get_geoentity_annotation_list("no geo xyzqqq", geo_config_list=_CONFIG) == []
