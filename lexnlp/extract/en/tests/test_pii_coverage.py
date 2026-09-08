"""Coverage tests for :mod:`lexnlp.extract.en.pii` list wrappers."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.en.pii import (
    get_pii_annotation_list,
    get_pii_list,
    get_ssn_annotation_list,
    get_ssn_list,
    get_us_phone_annotation_list,
    get_us_phone_list,
)

TEXT = "My ID is 078-05-1120 and my phone number is 212-212-2121"


def test_get_ssn_list() -> None:
    assert get_ssn_list(TEXT) == ["078-05-1120"]
    assert get_ssn_list(TEXT, return_sources=True) == [("078-05-1120", "078-05-1120")]
    assert get_ssn_list("no identifiers here") == []


def test_get_ssn_annotation_list() -> None:
    annotations = get_ssn_annotation_list(TEXT)
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.number == "078-05-1120"
    assert annotation.text == "078-05-1120"
    assert annotation.coords == (9, 20)


def test_get_us_phone_list() -> None:
    assert get_us_phone_list(TEXT) == ["(212) 212-2121"]
    assert get_us_phone_list(TEXT, return_sources=True) == [("(212) 212-2121", "212-212-2121")]
    assert get_us_phone_list("no identifiers here") == []


def test_get_us_phone_annotation_list() -> None:
    annotations = get_us_phone_annotation_list(TEXT)
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.phone == "(212) 212-2121"
    assert annotation.text == "212-212-2121"
    assert annotation.coords == (44, 56)


def test_get_pii_list() -> None:
    assert get_pii_list(TEXT) == [("ssn", "078-05-1120"), ("us_phone", "(212) 212-2121")]
    assert get_pii_list(TEXT, return_sources=True) == [
        ("ssn", "078-05-1120", "078-05-1120"),
        ("us_phone", "(212) 212-2121", "212-212-2121"),
    ]
    assert get_pii_list("no identifiers here") == []


def test_get_pii_annotation_list() -> None:
    annotations = get_pii_annotation_list(TEXT)
    assert len(annotations) == 2
    assert [ant.record_type for ant in annotations] == ["ssn", "phone"]
    assert annotations[0].coords == (9, 20)
    assert annotations[1].coords == (44, 56)
