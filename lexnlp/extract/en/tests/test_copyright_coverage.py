"""Coverage tests for lexnlp.extract.en.copyright missing lines."""

from lexnlp.extract.common.annotations.copyright_annotation import CopyrightAnnotation
from lexnlp.extract.en.copyright import (
    get_copyright_annotation_list,
    get_copyright_annotations,
    get_copyright_list,
    get_copyrights,
)


def test_get_copyright_list_basic() -> None:
    text = "Copyright 2020 Maverick International, Inc."
    actual = get_copyright_list(text)
    assert actual == [("Copyright", "2020", "Maverick International, Inc")]
    assert actual == list(get_copyrights(text))


def test_get_copyright_list_with_sources() -> None:
    text = "Copyright 2020 Maverick International, Inc."
    actual = get_copyright_list(text, return_sources=True)
    assert actual == [
        ("Copyright", "2020", "Maverick International, Inc", "Copyright 2020 Maverick International, Inc")
    ]
    assert actual == list(get_copyrights(text, return_sources=True))


def test_get_copyright_list_empty() -> None:
    assert get_copyright_list("No copyright here.") == []
    assert get_copyright_list("") == []
    assert get_copyright_annotation_list("No copyright here.") == []
    assert get_copyright_annotation_list("") == []


def test_get_copyright_annotation_list_fields() -> None:
    text = "Copyright 2020 Maverick International, Inc."
    actual = get_copyright_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, CopyrightAnnotation)
    assert ant.sign == "Copyright"
    assert ant.date == "2020"
    assert ant.name == "Maverick International, Inc"
    assert ant.coords == (0, 42)
    assert ant.locale == "en"
    expected = list(get_copyright_annotations(text))
    assert len(expected) == 1
    assert actual[0].sign == expected[0].sign
    assert actual[0].coords == expected[0].coords


def test_get_copyright_annotation_list_sources_flag() -> None:
    text = "Copyright 2020 Maverick International, Inc."
    without = get_copyright_annotation_list(text)[0]
    assert without.text == ""
    with_sources = get_copyright_annotation_list(text, return_sources=True)[0]
    assert with_sources.text == "Copyright 2020 Maverick International, Inc"
    assert with_sources.sign == without.sign
    assert with_sources.coords == without.coords
