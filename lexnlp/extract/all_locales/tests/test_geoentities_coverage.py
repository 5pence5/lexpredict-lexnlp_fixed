"""Coverage tests for lexnlp.extract.all_locales.geoentities dispatch."""

from __future__ import annotations

import inspect
import os

from lexnlp.extract.all_locales import geoentities as dispatcher
from lexnlp.extract.all_locales.geoentities import (
    ROUTINE_BY_LOCALE,
    get_geoentity_annotations,
)
from lexnlp.extract.all_locales.languages import DEFAULT_LANGUAGE, LANG_DE, LANG_EN
from lexnlp.extract.common.annotations.geo_annotation import GeoAnnotation
from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.extract.de.geoentities import (
    get_geoentity_annotations as get_geoentity_annotations_de,
)
from lexnlp.extract.en.dict_entities import DictionaryEntry
from lexnlp.extract.en.geoentities import (
    get_geoentity_annotations as get_geoentity_annotations_en,
)

_BASE = os.path.join(lexnlp_test_path, "lexnlp/extract/en/tests/test_geoentities")
_CONFIG = list(
    DictionaryEntry.load_entities_from_files(
        os.path.join(_BASE, "geoentities.csv"), os.path.join(_BASE, "geoaliases.csv")
    )
)
_TEXT = "I live in Berlin."


class TestDispatchTable:
    def test_table_maps_both_locales(self) -> None:
        assert ROUTINE_BY_LOCALE[LANG_EN.code] is get_geoentity_annotations_en
        assert ROUTINE_BY_LOCALE[LANG_DE.code] is get_geoentity_annotations_de

    def test_default_language_is_english(self) -> None:
        assert DEFAULT_LANGUAGE.code == LANG_EN.code

    def test_dispatcher_is_a_generator_function(self) -> None:
        assert inspect.isgeneratorfunction(get_geoentity_annotations)


class TestLocaleRouting:
    def test_en_locale(self) -> None:
        annotations = list(get_geoentity_annotations("en", _TEXT, _CONFIG))
        assert len(annotations) == 1
        first = annotations[0]
        assert isinstance(first, GeoAnnotation)
        assert first.coords == (10, 16)
        assert first.name == "Berlin"
        assert _TEXT[first.coords[0] : first.coords[1]] == "Berlin"

    def test_en_us_locale_uses_language_prefix(self) -> None:
        annotations = list(get_geoentity_annotations("en_US", _TEXT, _CONFIG))
        assert [(a.coords, a.name) for a in annotations] == [((10, 16), "Berlin")]

    def test_de_locale(self) -> None:
        annotations = list(get_geoentity_annotations("de", _TEXT, _CONFIG))
        assert len(annotations) == 1
        assert isinstance(annotations[0], GeoAnnotation)
        assert (annotations[0].coords, annotations[0].name) == ((10, 16), "Berlin")

    def test_unknown_locale_falls_back_to_default(self) -> None:
        annotations = list(get_geoentity_annotations("fr_FR", _TEXT, _CONFIG))
        assert [(a.coords, a.name) for a in annotations] == [((10, 16), "Berlin")]

    def test_en_and_fallback_agree(self) -> None:
        expected = [(a.coords, a.name, a.alias) for a in get_geoentity_annotations("en", _TEXT, _CONFIG)]
        actual = [(a.coords, a.name, a.alias) for a in get_geoentity_annotations("xx", _TEXT, _CONFIG)]
        assert actual == expected

    def test_no_match_yields_empty(self) -> None:
        assert list(get_geoentity_annotations("en", "nothing geographic here", _CONFIG)) == []

    def test_kwargs_are_forwarded(self) -> None:
        annotations = list(
            get_geoentity_annotations(
                "en",
                _TEXT,
                _CONFIG,
                conflict_resolving_field="id",
                priority_direction="asc",
                text_languages=["en"],
                min_alias_len=2,
                prepared_alias_ban_list=None,
                simplified_normalization=True,
            )
        )
        assert [(a.coords, a.name) for a in annotations] == [((10, 16), "Berlin")]

    def test_routing_matches_direct_routines(self, monkeypatch) -> None:
        calls: list[str] = []
        en_direct = list(get_geoentity_annotations_en(_TEXT, _CONFIG))
        de_direct = list(get_geoentity_annotations_de(_TEXT, _CONFIG))

        def _spy(name: str, expected):
            def _inner(*args, **kwargs):
                calls.append(name)
                yield from expected

            return _inner

        monkeypatch.setitem(dispatcher.ROUTINE_BY_LOCALE, "en", _spy("en", en_direct))
        monkeypatch.setitem(dispatcher.ROUTINE_BY_LOCALE, "de", _spy("de", de_direct))
        assert [a.name for a in get_geoentity_annotations("en", _TEXT, _CONFIG)] == [a.name for a in en_direct]
        assert [a.name for a in get_geoentity_annotations("de", _TEXT, _CONFIG)] == [a.name for a in de_direct]
        assert calls == ["en", "de"]
