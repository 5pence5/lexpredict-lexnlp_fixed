"""Coverage tests for :mod:`lexnlp.extract.pt.citations` missing lines."""

from lexnlp.extract.pt.citations import (
    _normalise_year,
    _parse_int,
    get_case_citation_annotations,
    get_citation_list,
    get_citations,
)


def test_parse_int_none_returns_none() -> None:
    assert _parse_int(None) is None
    assert _parse_int("12.345") == 12345
    assert _parse_int("999") == 999
    assert _parse_int("abc") is None


def test_normalise_year_fallthrough_returns_full_value() -> None:
    assert _normalise_year("2019") == 2019
    assert _normalise_year("202") == 202
    assert _normalise_year(None) is None
    assert _normalise_year("85") == 1985
    assert _normalise_year("20") == 2020


def test_four_digit_year_annotation() -> None:
    ants = list(get_case_citation_annotations("Vide RE 123.456/2019 julgado."))
    assert len(ants) == 1
    assert ants[0].text == "RE 123.456/2019"
    assert ants[0].reporter == "RE"
    assert ants[0].volume == 123456
    assert ants[0].year == 2019


def test_get_citations_yields_surface_forms() -> None:
    text = "Vide REsp 12.345/SP e processo 1234567-56.2020.5.04.0001."
    assert list(get_citations(text)) == ["REsp 12.345/SP", "1234567-56.2020.5.04.0001"]
    assert list(get_citations("Sem citacoes aqui.")) == []


def test_get_citation_list_returns_list() -> None:
    text = "Vide REsp 12.345/SP e processo 1234567-56.2020.5.04.0001."
    result = get_citation_list(text)
    assert result == ["REsp 12.345/SP", "1234567-56.2020.5.04.0001"]
    assert result == list(get_citations(text))
    assert get_citation_list("Sem citacoes aqui.") == []
