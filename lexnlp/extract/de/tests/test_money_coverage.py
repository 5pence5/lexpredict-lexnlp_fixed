"""Coverage tests for the module-level helpers in ``lexnlp.extract.de.money``.

The existing suite only exercises ``get_money_annotations``; lines 62
(``get_money``), 70 (``get_money_list``) and 85
(``get_money_annotation_list``) need direct calls asserted on real
return values.
"""

from __future__ import annotations

import types
from decimal import Decimal

from lexnlp.extract.common.annotations.money_annotation import MoneyAnnotation
from lexnlp.extract.de.money import (
    get_money,
    get_money_annotation_list,
    get_money_annotations,
    get_money_list,
)

TEXT = "100 Pfunde, 45 Dollars"


class TestGetMoney:
    def test_default_tuples(self) -> None:
        result = get_money(TEXT)
        assert isinstance(result, types.GeneratorType)
        assert list(result) == [(Decimal("100.0"), "GBP"), (Decimal("45.0"), "USD")]

    def test_return_sources_adds_text(self) -> None:
        assert list(get_money(TEXT, return_sources=True)) == [
            (Decimal("100.0"), "GBP", "100 Pfunde"),
            (Decimal("45.0"), "USD", "45 Dollars"),
        ]

    def test_empty_text(self) -> None:
        assert list(get_money("")) == []


class TestGetMoneyList:
    def test_list_matches_generator(self) -> None:
        result = get_money_list(TEXT)
        assert isinstance(result, list)
        assert result == list(get_money(TEXT))
        assert result == [(Decimal("100.0"), "GBP"), (Decimal("45.0"), "USD")]

    def test_empty_text(self) -> None:
        assert get_money_list("") == []


class TestGetMoneyAnnotationList:
    def test_annotations_pinned(self) -> None:
        ants = get_money_annotation_list(TEXT)
        assert isinstance(ants, list)
        assert len(ants) == 2
        assert all(isinstance(a, MoneyAnnotation) for a in ants)
        assert ants[0].amount == Decimal("100.0")
        assert ants[0].currency == "GBP"
        assert ants[0].locale == "de"
        assert ants[0].coords == (0, 11)
        assert ants[0].text == "100 Pfunde"
        assert ants[1].amount == Decimal("45.0")
        assert ants[1].currency == "USD"
        assert ants[1].coords == (12, 22)
        assert ants[1].text == "45 Dollars"

    def test_matches_generator(self) -> None:
        ants = get_money_annotation_list(TEXT)
        gen_ants = list(get_money_annotations(TEXT))
        assert [(a.amount, a.currency, a.coords) for a in ants] == [(a.amount, a.currency, a.coords) for a in gen_ants]

    def test_empty_text(self) -> None:
        assert get_money_annotation_list("") == []
