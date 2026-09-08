"""Coverage tests for get_court_annotations extra_columns branch."""

from __future__ import annotations

from lexnlp.extract.all_locales.courts import get_court_annotations
from lexnlp.extract.en.dict_entities import DictionaryEntry


def _make_entry() -> DictionaryEntry:
    return DictionaryEntry(
        id=7,
        name="Supreme Court of Testland",
        priority=1,
        entity_name="Supreme Court of Testland EN",
        category="supreme",
        extra_columns={"jurisdiction": "Testland", "court_type": "supreme"},
    )


class TestCourtExtraColumns:
    def test_extra_columns_copied_onto_annotation(self) -> None:
        entry = _make_entry()
        text = "The Supreme Court of Testland decided the case."
        ants = list(get_court_annotations("en", text, [entry]))
        assert len(ants) == 1
        ant = ants[0]
        assert ant.entity_id == 7
        assert ant.entity_category == "supreme"
        assert ant.entity_priority == 1
        assert ant.name_en == "Supreme Court of Testland EN"
        assert ant.name == "Supreme Court of Testland"
        assert ant.alias == "Supreme Court of Testland"
        assert ant.locale == "en"
        assert ant.jurisdiction == "Testland"
        assert ant.court_type == "supreme"
        assert ant.coords == (4, 29)
        assert text[ant.coords[0] : ant.coords[1]] == "Supreme Court of Testland"

    def test_extra_columns_with_priority_flag(self) -> None:
        entry = _make_entry()
        text = "The Supreme Court of Testland decided the case."
        ants = list(get_court_annotations("en", text, [entry], priority=True))
        assert len(ants) == 1
        assert ants[0].jurisdiction == "Testland"
        assert ants[0].court_type == "supreme"
