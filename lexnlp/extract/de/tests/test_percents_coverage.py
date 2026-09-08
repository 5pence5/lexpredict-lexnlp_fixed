"""Coverage tests for lexnlp.extract.de.percents missing lines."""

from decimal import Decimal

from lexnlp.extract.common.annotations.percent_annotation import PercentAnnotation
from lexnlp.extract.de.percents import (
    get_percent_annotation_list,
    get_percent_annotations,
    get_percent_list,
    get_percents,
)


def test_unparseable_amount_match_is_skipped():
    # "1,2,3" matches the percent pattern but yields no single amount,
    # so the match is skipped (line 65 `continue`).
    assert list(get_percent_annotations("1,2,3 Prozent")) == []
    assert list(get_percents("1,2,3 Prozent")) == []
    # Control: a normal value still extracts so the test is not vacuous.
    control = list(get_percent_annotations("15 Prozent"))
    assert len(control) == 1
    assert control[0].amount == Decimal("15.0")


def test_get_percent_list_wrapper():
    text = "15 Prozent"
    actual = get_percent_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    entry = actual[0]
    assert entry["source_text"] == "15 Prozent"
    assert entry["unit_name"] == "prozent"
    assert entry["amount"] == Decimal("15.0")
    assert entry["real_amount"] == Decimal("15.0000")
    assert entry["location_start"] == 0
    assert entry["location_end"] == 10
    assert actual == list(get_percents(text))


def test_get_percent_list_empty_without_match():
    assert get_percent_list("kein Prozentwert hier") == []


def test_get_percent_annotation_list_wrapper():
    text = "15 Prozent"
    actual = get_percent_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, PercentAnnotation)
    assert ant.coords == (0, 10)
    assert ant.sign == "prozent"
    assert ant.amount == Decimal("15.0")
    assert ant.locale == "de"
    expected = list(get_percent_annotations(text))
    assert len(expected) == 1
    assert actual[0].coords == expected[0].coords
    assert actual[0].text == expected[0].text


def test_get_percent_annotation_list_empty_without_match():
    assert get_percent_annotation_list("kein Prozentwert hier") == []
