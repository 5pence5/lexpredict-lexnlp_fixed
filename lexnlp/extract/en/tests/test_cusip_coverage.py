"""Coverage tests for lexnlp.extract.en.cusip missing lines."""

from lexnlp.extract.common.annotations.cusip_annotation import CusipAnnotation
from lexnlp.extract.en.cusip import (
    get_cusip_annotation_list,
    get_cusip_annotations,
    is_cusip_valid,
)


def test_is_cusip_valid_rejects_unknown_char():
    # Lowercase "a" is neither a digit nor in CHECKSUM_BASE (line 68).
    assert is_cusip_valid("12a456781") is False
    # Control: a known-good CUSIP validates.
    assert is_cusip_valid("837649128") is True


def test_is_cusip_valid_return_checksum():
    # Line 74: return the computed checksum instead of comparing.
    assert is_cusip_valid("837649128", return_checksum=True) == 8
    assert is_cusip_valid("392690QT3", return_checksum=True) == 3


def test_get_cusip_annotation_list_wrapper():
    text = "This is 837649128"
    actual = get_cusip_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, CusipAnnotation)
    assert ant.code == "837649128"
    assert ant.issuer_id == "837649"
    assert ant.issue_id == "12"
    assert ant.checksum == 8
    assert ant.internal is False
    assert ant.ppn is False
    assert ant.tba is None
    assert ant.coords == (8, 17)
    expected = list(get_cusip_annotations(text))
    assert len(expected) == 1
    assert actual[0].code == expected[0].code
    assert actual[0].coords == expected[0].coords


def test_get_cusip_annotation_list_empty_without_match():
    assert get_cusip_annotation_list("no cusip here") == []
