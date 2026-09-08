"""Coverage tests for lexnlp.extract.de.amounts."""

from decimal import Decimal

import pytest

from lexnlp.extract.de.amounts import (
    AmountParserDE,
    get_amount_annotation_list,
    get_amount_list,
)


def test_text2num_unknown_word_raises_runtime_error():
    parser = AmountParserDE()
    with pytest.raises(RuntimeError, match="Unknown number: blah"):
        parser.text2num("blah")


def test_parse_with_extended_sources_returns_value_text_and_coords():
    parser = AmountParserDE()
    assert list(parser.parse("dreißig", return_sources=True, extended_sources=True)) == [
        (Decimal("30"), "dreißig", (0, 7))
    ]


def test_parse_with_simple_sources_returns_value_and_text():
    parser = AmountParserDE()
    assert list(parser.parse("dreißig", return_sources=True, extended_sources=False)) == [(Decimal("30"), "dreißig")]


def test_parse_without_sources_returns_values_only():
    parser = AmountParserDE()
    assert list(parser.parse("dreißig", return_sources=False)) == [Decimal("30")]


def test_parse_annotations_skips_unparseable_match_and_prints_error(capsys):
    # "1.2.3" matches the numeric pattern but Decimal() rejects it, so the
    # annotation is skipped after printing the conversion error.
    parser = AmountParserDE()
    assert list(parser.parse_annotations("1.2.3")) == []
    captured = capsys.readouterr()
    assert "ConversionSyntax" in captured.out


def test_parse_annotations_skips_none_amount(monkeypatch):
    # Defensive guard: text2num never returns None in practice (it returns a
    # Decimal or raises), so force None to exercise the skip branch.
    parser = AmountParserDE()
    monkeypatch.setattr(AmountParserDE, "text2num", lambda self, s: None)
    assert list(parser.parse_annotations("dreißig")) == []


def test_parse_annotations_picks_up_preceding_currency_symbol_with_space():
    parser = AmountParserDE()
    ants = list(parser.parse_annotations("€ 100"))
    assert len(ants) == 1
    assert ants[0].value == Decimal("100")
    assert ants[0].text == "€ 100"
    assert ants[0].coords == (2, 5)


def test_parse_annotations_picks_up_attached_currency_symbol():
    parser = AmountParserDE()
    ants = list(parser.parse_annotations("€100"))
    assert len(ants) == 1
    assert ants[0].value == Decimal("100")
    assert ants[0].text == "€100"
    assert ants[0].coords == (1, 4)


def test_parse_annotations_picks_up_currency_prefix_word():
    parser = AmountParserDE()
    ants = list(parser.parse_annotations("CHF 100"))
    assert len(ants) == 1
    assert ants[0].value == Decimal("100")
    assert ants[0].text == "CHF 100"


def test_get_amount_list_without_sources():
    assert get_amount_list("dreißig") == [Decimal("30")]


def test_get_amount_list_with_sources():
    assert get_amount_list("dreißig", return_sources=True) == [(Decimal("30"), "dreißig", (0, 7))]


def test_get_amount_annotation_list():
    ants = get_amount_annotation_list("dreißig")
    assert len(ants) == 1
    assert ants[0].value == Decimal("30")
    assert ants[0].text == "dreißig"
    assert ants[0].coords == (0, 7)
