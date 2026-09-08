"""Coverage tests for lexnlp.extract.en.urls list wrappers."""

from lexnlp.extract.common.annotations.url_annotation import UrlAnnotation
from lexnlp.extract.en.urls import (
    get_url_annotation_list,
    get_url_annotations,
    get_url_list,
    get_urls,
)


def test_url_list_returns_urls() -> None:
    text = "Visit https://example.com/page and www.google.com now"
    result = get_url_list(text)
    assert result == ["https://example.com/page", "www.google.com"]
    assert result == list(get_urls(text))


def test_url_list_empty() -> None:
    assert get_url_list("no urls here, just words") == []
    assert get_url_list("") == []


def test_url_annotation_list() -> None:
    text = "Visit https://example.com/page and www.google.com now"
    result = get_url_annotation_list(text)
    assert len(result) == 2
    assert all(isinstance(ant, UrlAnnotation) for ant in result)
    assert result[0].url == "https://example.com/page"
    assert result[0].coords == (6, 30)
    assert result[0].locale == "en"
    assert result[1].url == "www.google.com"
    assert result[1].coords == (35, 49)
    assert result[1].locale == "en"
    expected = list(get_url_annotations(text))
    assert [ant.url for ant in result] == [ant.url for ant in expected]
    assert [ant.coords for ant in result] == [ant.coords for ant in expected]


def test_url_annotation_list_empty() -> None:
    assert get_url_annotation_list("no urls here, just words") == []
    assert get_url_annotation_list("") == []
