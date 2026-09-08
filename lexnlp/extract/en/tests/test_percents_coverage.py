"""Coverage tests for lexnlp.extract.en.percents missing lines."""

from decimal import Decimal

from lexnlp.extract.common.annotations.percent_annotation import PercentAnnotation
from lexnlp.extract.en.percents import (
    get_percent_annotation_list,
    get_percent_annotations,
    get_percent_list,
    get_percents,
)


def test_unparseable_number_match_is_skipped():
    # "1,2,3" matches the percent pattern but parses to neither a single
    # amount nor a single ratio, so the match is skipped (line 104).
    assert list(get_percent_annotations("1,2,3 percent")) == []
    assert list(get_percents("1,2,3 percent")) == []
    # Control: a normal value still extracts so the test is not vacuous.
    control = list(get_percent_annotations("5 percent"))
    assert len(control) == 1
    assert control[0].amount == Decimal("5.0")


def test_ratio_based_percent_value():
    # "1/2" parses as a ratio (not a plain amount), exercising the
    # ratio-fallback branch of get_percent_annotations.
    actual = get_percent_annotations("1/2 percent")
    actual = list(actual)
    assert len(actual) == 1
    assert actual[0].amount == Decimal("0.5")
    assert actual[0].fraction == Decimal("0.005")
    assert actual[0].sign == "percent"


def test_get_percent_list_wrapper():
    text = "5 percent"
    actual = get_percent_list(text)
    assert actual == [("percent", Decimal("5.0"), Decimal("0.050"))]
    assert actual == list(get_percents(text))


def test_get_percent_list_with_sources():
    actual = get_percent_list("5 percent", return_sources=True)
    assert actual == [("percent", Decimal("5.0"), Decimal("0.050"), "5 percent")]
    assert actual == list(get_percents("5 percent", return_sources=True))


def test_get_percent_list_empty_without_match():
    assert get_percent_list("no percent here") == []


def test_get_percent_annotation_list_wrapper():
    text = "5 percent"
    actual = get_percent_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, PercentAnnotation)
    assert ant.amount == Decimal("5.0")
    assert ant.fraction == Decimal("0.050")
    assert ant.sign == "percent"
    assert ant.coords == (0, 9)
    expected = list(get_percent_annotations(text))
    assert len(expected) == 1
    assert actual[0].coords == expected[0].coords
    assert actual[0].text == expected[0].text


def test_get_percent_annotation_list_empty_without_match():
    assert get_percent_annotation_list("no percent here") == []
