"""Coverage tests for uncovered helpers in lexnlp.extract.en.dict_entities."""

from __future__ import annotations

from lexnlp.extract.en.dict_entities import (
    AliasBanRecord,
    DictionaryEntity,
    DictionaryEntry,
    DictionaryEntryAlias,
    SearchResultPosition,
    _find_entity_positions,
    alias_is_banlisted,
    normalize_text,
    normalize_text_with_map,
    prepare_alias_banlist_dict,
)


class TestDictionaryEntryAliasReprAndFactory:
    def test_repr_includes_alias_id_when_set(self) -> None:
        alias = DictionaryEntryAlias(alias="Mississippi", language="en", alias_id=1051)
        assert repr(alias) == "Mississippi, lang: en, id: 1051"

    def test_repr_omits_id_when_missing(self) -> None:
        alias = DictionaryEntryAlias(alias="Mississippi", language="fr")
        assert repr(alias) == "Mississippi, lang: fr"

    def test_entity_alias_normalizes_lowercase_by_default(self) -> None:
        alias = DictionaryEntryAlias.entity_alias("Mississippi", language="en", alias_id=7)
        assert alias.alias == "Mississippi"
        assert alias.language == "en"
        assert alias.alias_id == 7
        assert alias.is_abbreviation is False
        assert alias.normalized_alias == normalize_text("Mississippi", lowercase=True)

    def test_entity_alias_preserves_case_for_abbreviations(self) -> None:
        alias = DictionaryEntryAlias.entity_alias("MS", language="en", is_abbreviation=True)
        assert alias.is_abbreviation is True
        assert alias.normalized_alias == normalize_text("MS", lowercase=False)
        assert "MS" in alias.normalized_alias


class TestHasCloserLocale:
    def test_falsy_language_list_is_never_closer(self) -> None:
        left = DictionaryEntryAlias(alias="A", language="en")
        right = DictionaryEntryAlias(alias="B", language="fr")
        assert left.has_closer_locale(right, None) is False
        assert left.has_closer_locale(right, []) is False

    def test_specified_language_loses_to_empty_default(self) -> None:
        specified = DictionaryEntryAlias(alias="A", language="en")
        unspecified = DictionaryEntryAlias(alias="B", language="")
        assert specified.has_closer_locale(unspecified, ["en", "fr"]) is False

    def test_empty_self_language_beats_specified_other(self) -> None:
        unspecified = DictionaryEntryAlias(alias="A", language="")
        specified = DictionaryEntryAlias(alias="B", language="en")
        assert unspecified.has_closer_locale(specified, ["en"]) is True

    def test_self_wins_when_it_appears_first_in_order(self) -> None:
        english = DictionaryEntryAlias(alias="A", language="en")
        french = DictionaryEntryAlias(alias="B", language="fr")
        assert english.has_closer_locale(french, ["en", "fr"]) is True

    def test_self_loses_when_other_appears_first_in_order(self) -> None:
        english = DictionaryEntryAlias(alias="A", language="en")
        french = DictionaryEntryAlias(alias="B", language="fr")
        assert english.has_closer_locale(french, ["fr", "en"]) is False

    def test_neither_language_in_order_returns_false(self) -> None:
        english = DictionaryEntryAlias(alias="A", language="en")
        french = DictionaryEntryAlias(alias="B", language="fr")
        assert english.has_closer_locale(french, ["de"]) is False


class TestDictionaryEntryStr:
    def test_str_matches_name_and_id(self) -> None:
        entry = DictionaryEntry(id=9, name="Delaware", name_is_alias=False)
        assert str(entry) == '"Delaware": #9'
        assert str(entry) == repr(entry)


class TestAliasBanRecordRepr:
    def test_abbreviation_suffix(self) -> None:
        record = AliasBanRecord(alias="AM", lang="en", is_abbrev=True)
        assert repr(record) == "AM, en abbr."

    def test_non_abbreviation_has_no_suffix(self) -> None:
        record = AliasBanRecord(alias="alias", lang="fr", is_abbrev=False)
        assert repr(record) == "alias, fr"


class TestSearchResultPositionRepr:
    def test_repr_includes_entity_map_and_span(self) -> None:
        entry = DictionaryEntry(id=1, name="Court", name_is_alias=False)
        alias = DictionaryEntryAlias(alias="Court", language="en")
        position = SearchResultPosition(entry, alias, start=4, end=9, source_text="Court")
        text = repr(position)
        assert 'alias="Court"' in text
        assert "@[4, 9]" in text
        assert "1" in text


class TestNormalizeTextWithMapSimpleTokenization:
    def test_lowercases_space_split_tokens(self) -> None:
        dst, src_map = normalize_text_with_map(
            "hello world",
            simple_tokenization=True,
            lowercase=True,
            use_stemmer=False,
        )
        assert "hello" in dst
        assert "world" in dst
        assert dst == dst.lower()
        assert len(src_map) == len("hello world")
        assert src_map[0] >= 0

    def test_simple_tokenization_without_lowercase_preserves_case(self) -> None:
        dst, _src_map = normalize_text_with_map(
            "Hello World",
            simple_tokenization=True,
            lowercase=False,
            use_stemmer=False,
        )
        assert "Hello" in dst
        assert "World" in dst


class TestFindEntityPositions:
    def test_empty_aliases_leave_context_untouched(self) -> None:
        entry = DictionaryEntry(id=1, name="Ghost", name_is_alias=False, aliases=[])
        context: dict[int, SearchResultPosition] = {}
        _find_entity_positions(
            " ghost ",
            " ghost ",
            entry,
            text_languages=None,
            alias_language_order=["en"],
            context=context,
        )
        assert context == {}

    def test_none_context_and_empty_aliases_is_a_no_op(self) -> None:
        entry = DictionaryEntry(id=1, name="Ghost", name_is_alias=False, aliases=[])
        result = _find_entity_positions(
            " ghost ",
            " ghost ",
            entry,
            text_languages=None,
            alias_language_order=["en"],
            context=None,
        )
        assert result is None

    def test_banlisted_alias_is_skipped(self) -> None:
        alias = DictionaryEntryAlias.entity_alias("Court", language="en")
        entry = DictionaryEntry(id=2, name="Court", name_is_alias=False, aliases=[alias])
        ban_list = prepare_alias_banlist_dict([AliasBanRecord("Court", "en", False)])
        assert ban_list is not None
        assert alias_is_banlisted(ban_list, alias.normalized_alias, "en", False) is True

        context: dict[int, SearchResultPosition] = {}
        _find_entity_positions(
            " Court ",
            " court ",
            entry,
            text_languages=None,
            alias_language_order=["en"],
            context=context,
            alias_ban_list=ban_list,
        )
        assert context == {}

        unbanned: dict[int, SearchResultPosition] = {}
        _find_entity_positions(
            " Court ",
            " court ",
            entry,
            text_languages=None,
            alias_language_order=["en"],
            context=unbanned,
            alias_ban_list=None,
        )
        assert len(unbanned) == 1
        found = next(iter(unbanned.values()))
        assert found.alias_text == "Court"

    def test_none_context_still_searches_when_aliases_exist(self) -> None:
        alias = DictionaryEntryAlias.entity_alias("Court", language="en")
        entry = DictionaryEntry(id=3, name="Court", name_is_alias=False, aliases=[alias])
        result = _find_entity_positions(
            " Court sits ",
            " court sits ",
            entry,
            text_languages=None,
            alias_language_order=["en"],
            context=None,
        )
        assert result is None


class TestDictionaryEntityRepr:
    def test_none_entity_and_missing_coords(self) -> None:
        wrapped = DictionaryEntity(entity=None, coords=None)
        assert repr(wrapped) == "None, -"

    def test_single_item_entity_with_coords(self) -> None:
        entry = DictionaryEntry(id=4, name="Texas", name_is_alias=False)
        wrapped = DictionaryEntity(entity=(entry,), coords=(10, 15))
        assert repr(wrapped) == '"Texas": #4, @[10, 15]'

    def test_pair_entity_appends_second_member(self) -> None:
        entry = DictionaryEntry(id=5, name="Texas", name_is_alias=False)
        alias = DictionaryEntryAlias(alias="TX", language="en")
        wrapped = DictionaryEntity(entity=(entry, alias), coords=(1, 2))
        text = repr(wrapped)
        assert '"Texas": #5' in text
        assert "TX, lang: en" in text
        assert "@[1, 2]" in text
