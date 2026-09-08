"""Coverage tests for lexnlp.extract.en.distances missing lines."""

from decimal import Decimal

from lexnlp.extract.common.annotations.distance_annotation import DistanceAnnotation
from lexnlp.extract.en.distances import (
    get_distance_annotation_list,
    get_distance_annotations,
    get_distance_list,
    get_distances,
)


def test_get_distance_list_basic() -> None:
    actual = get_distance_list("Today I ran 8 miles.")
    assert actual == [(Decimal("8.0"), "mile")]
    assert actual == list(get_distances("Today I ran 8 miles."))


def test_get_distance_list_symbol_and_sources() -> None:
    actual = get_distance_list("The road is 5 km long.")
    assert actual == [(Decimal("5.0"), "kilometer")]
    with_sources = get_distance_list("The road is 5 km long.", return_sources=True)
    assert with_sources == [(Decimal("5.0"), "kilometer", "5 km")]
    assert with_sources == list(get_distances("The road is 5 km long.", return_sources=True))


def test_get_distance_list_empty() -> None:
    assert get_distance_list("No distance here.") == []
    assert get_distance_list("") == []
    assert get_distance_list("km") == []
    assert get_distance_annotation_list("No distance here.") == []
    assert get_distance_annotation_list("") == []


def test_get_distance_annotation_list_fields() -> None:
    text = "The road is 5 km long."
    actual = get_distance_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, DistanceAnnotation)
    assert ant.coords == (12, 17)
    assert ant.amount == Decimal("5.0")
    assert ant.distance_type == "kilometer"
    assert ant.text == "5 km"
    assert ant.locale == "en"
    expected = list(get_distance_annotations(text))
    assert len(expected) == 1
    assert actual[0].coords == expected[0].coords
    assert actual[0].text == expected[0].text


def test_get_distance_list_word_number_and_float_digits() -> None:
    assert get_distance_list("Ran eight kilometers today.") == [(Decimal("8.0"), "kilometer")]
    assert get_distance_list("The road is 5 km long.", float_digits=0) == [(Decimal("5"), "kilometer")]
