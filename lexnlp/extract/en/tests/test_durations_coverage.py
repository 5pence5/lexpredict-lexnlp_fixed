"""Coverage tests for lexnlp.extract.en.durations line 92."""

from decimal import Decimal

from lexnlp.extract.en.durations import get_duration_list, get_durations


def test_get_duration_list_simple_day():
    result = get_duration_list("The lease is for 30 days.")
    assert result == [("day", Decimal("30.0"), Decimal("30.0"))]
    assert isinstance(result, list)


def test_get_duration_list_matches_generator():
    text = "The term is 2 years."
    assert get_duration_list(text) == list(get_durations(text))
    assert get_duration_list(text) == [("year", Decimal("2.0"), Decimal("730.0"))]


def test_get_duration_list_with_sources():
    result = get_duration_list("The lease is for 30 days.", return_sources=True)
    assert len(result) == 1
    duration_type, amount, days, source = result[0]
    assert duration_type == "day"
    assert amount == Decimal("30.0")
    assert days == Decimal("30.0")
    assert source == "30 days"


def test_get_duration_list_empty():
    assert get_duration_list("No duration here.") == []
    assert get_duration_list("") == []
