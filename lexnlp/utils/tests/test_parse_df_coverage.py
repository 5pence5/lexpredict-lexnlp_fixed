"""Coverage tests for lexnlp.utils.parse_df missing lines."""

from __future__ import annotations

import pandas as pd

from lexnlp.utils.parse_df import DataframeEntityParser, get_entities, get_entity_list


def _dup_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Peppa", "Peppa"],
            "title": ["Low", "High"],
            "prio": [2, 1],
        }
    )


class TestParseDfCoverage:
    def test_empty_collection_returns_none(self) -> None:
        df = pd.DataFrame({"name": ["A"], "empty": [""]})
        parser = DataframeEntityParser(dataframe=df, parse_columns=["empty"])
        assert parser.get_collection_ptn(["", ""]) is None
        assert parser.collection_patterns == {}
        assert parser.get_entity_list("A") == []

    def test_priority_sort_column_picks_first(self) -> None:
        parser = DataframeEntityParser(
            dataframe=_dup_df(),
            parse_columns=["name"],
            result_columns={"title": "label"},
            priority_sort_column="prio",
        )
        ents = parser.get_entity_list("Peppa is here")
        assert len(ents) == 1
        assert ents[0]["label"] == "High"
        assert ents[0]["source"] == "Peppa"

    def test_non_unique_column_values_returns_entities(self) -> None:
        parser = DataframeEntityParser(
            dataframe=_dup_df(),
            parse_columns=["name"],
            result_columns={"title": "label"},
            unique_column_values=False,
        )
        ents = parser.get_entity_list("Peppa is here")
        assert len(ents) == 1
        assert ents[0]["source"] == "Peppa"
        assert ents[0]["entities"] == [{"label": "Low"}, {"label": "High"}]

    def test_module_get_entities(self) -> None:
        df = _dup_df()
        ents = list(get_entities("Peppa is here", df, ["name"], {"title": "label"}))
        assert len(ents) == 1
        assert ents[0]["source"] == "Peppa"
        assert ents[0]["label"] == "Low"

    def test_module_get_entity_list(self) -> None:
        df = _dup_df()
        ents = get_entity_list("Peppa is here", df, ["name"], {"title": "label"})
        assert isinstance(ents, list)
        assert len(ents) == 1
        assert ents[0]["source"] == "Peppa"
        assert ents[0]["label"] == "Low"
