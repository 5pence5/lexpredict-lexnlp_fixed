"""Coverage tests for lexnlp.extract.en.regulations list wrappers."""

from lexnlp.extract.common.annotations.regulation_annotation import RegulationAnnotation
from lexnlp.extract.en.regulations import (
    get_regulation_annotation_list,
    get_regulation_annotations,
    get_regulation_list,
    get_regulations,
)


def test_regulation_list_returns_tuples() -> None:
    text = "test 123 U.S.C § 456, code"
    result = get_regulation_list(text)
    assert result == [("United States Code", "123 USC § 456")]
    assert result == list(get_regulations(text))


def test_regulation_list_return_source() -> None:
    text = "test 123 U.S.C § 456, code"
    result = get_regulation_list(text, return_source=True)
    assert result == [("United States Code", "123 USC § 456", "123 U.S.C § 456")]
    assert result == list(get_regulations(text, return_source=True))


def test_regulation_list_as_dict() -> None:
    text = "test 123 U.S.C § 456, code"
    result = get_regulation_list(text, as_dict=True)
    assert result == [
        {
            "regulation_type": "United States Code",
            "regulation_code": "123 USC § 456",
            "regulation_text": "123 U.S.C § 456",
        }
    ]


def test_regulation_list_empty() -> None:
    assert get_regulation_list("no regulations here xyz") == []
    assert get_regulation_list("") == []


def test_regulation_annotation_list() -> None:
    text = "test 123 U.S.C § 456, code"
    result = get_regulation_annotation_list(text)
    assert len(result) == 1
    ant = result[0]
    assert isinstance(ant, RegulationAnnotation)
    assert ant.source == "United States Code"
    assert ant.name == "123 USC § 456"
    assert ant.text == "123 U.S.C § 456"
    assert ant.locale == "en"
    assert ant.coords == (5, 20)
    expected = list(get_regulation_annotations(text))
    assert len(expected) == 1
    assert result[0].coords == expected[0].coords
    assert result[0].name == expected[0].name
    assert result[0].source == expected[0].source


def test_regulation_annotation_list_empty() -> None:
    assert get_regulation_annotation_list("no regulations here xyz") == []
    assert get_regulation_annotation_list("") == []
