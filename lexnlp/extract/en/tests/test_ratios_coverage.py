"""Coverage tests for ratio list-wrapper helpers."""

from decimal import Decimal

from lexnlp.extract.common.annotations.ratio_annotation import RatioAnnotation
from lexnlp.extract.en.ratios import (
    get_ratio_annotation_list,
    get_ratio_annotations,
    get_ratio_list,
    get_ratios,
)


def test_get_ratio_list_returns_ratios():
    text = "The ratio is 3 to 1."
    assert get_ratio_list(text) == [(Decimal("3.0"), Decimal("1.0"), Decimal("3"))]
    assert get_ratio_list(text) == list(get_ratios(text))


def test_get_ratio_list_with_sources():
    actual = get_ratio_list("The ratio is 3 to 1.", return_sources=True)
    assert actual == [(Decimal("3.0"), Decimal("1.0"), Decimal("3"), "3 to 1.")]
    assert actual == list(get_ratios("The ratio is 3 to 1.", return_sources=True))


def test_get_ratio_list_empty_without_match():
    assert get_ratio_list("no ratio here") == []


def test_get_ratio_annotation_list_returns_annotations():
    text = "The ratio is 3 to 1."
    expected = list(get_ratio_annotations(text))
    actual = get_ratio_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == len(expected) == 1
    assert all(isinstance(item, RatioAnnotation) for item in actual)
    annotation = actual[0]
    assert annotation.left == Decimal("3.0")
    assert annotation.right == Decimal("1.0")
    assert annotation.ratio == Decimal("3")
    assert annotation.coords == (13, 20)


def test_get_ratio_annotation_list_empty_without_match():
    assert get_ratio_annotation_list("no ratio here") == []
