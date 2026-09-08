"""Coverage tests for money list-wrapper helpers."""

from decimal import Decimal

from lexnlp.extract.common.annotations.money_annotation import MoneyAnnotation
from lexnlp.extract.en.money import (
    get_money,
    get_money_annotation_list,
    get_money_annotations,
    get_money_list,
)


def test_get_money_list_returns_amounts():
    text = "The cost is $25.50."
    assert get_money_list(text) == [(Decimal("25.50"), "USD")]
    assert get_money_list(text) == list(get_money(text))


def test_get_money_list_with_sources():
    actual = get_money_list("The cost is $25.50.", return_sources=True)
    assert actual == [(Decimal("25.50"), "USD", "cost is $25.50")]
    assert actual == list(get_money("The cost is $25.50.", return_sources=True))


def test_get_money_list_empty_without_match():
    assert get_money_list("no money here xyz") == []


def test_get_money_annotation_list_returns_annotations():
    text = "The cost is $25.50."
    expected = list(get_money_annotations(text))
    actual = get_money_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == len(expected) == 1
    assert all(isinstance(item, MoneyAnnotation) for item in actual)
    annotation = actual[0]
    assert annotation.amount == Decimal("25.50")
    assert annotation.currency == "USD"
    assert annotation.coords == (3, 18)
    assert annotation.text == "cost is $25.50"


def test_get_money_annotation_list_empty_without_match():
    assert get_money_annotation_list("no money here xyz") == []
