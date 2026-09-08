"""Coverage tests for lexnlp.extract.pt.money list wrapper."""

from decimal import Decimal

from lexnlp.extract.pt.money import get_money, get_money_list


def test_money_list_prefix_real() -> None:
    result = get_money_list("Valor: R$ 50,00.")
    assert len(result) == 1
    row = result[0]
    assert row["source_text"] == "R$ 50,00"
    assert row["amount"] == Decimal("50.0000")
    assert row["currency"] == "BRL"
    assert row["location_start"] == 7
    assert row["location_end"] == 15
    assert result == list(get_money("Valor: R$ 50,00."))


def test_money_list_suffix_form() -> None:
    result = get_money_list("O custo é de 1.500,00 reais por mês.")
    assert len(result) == 1
    assert result[0]["amount"] == Decimal("1500.0000")
    assert result[0]["currency"] == "BRL"
    assert result[0]["source_text"] == "1.500,00 reais"


def test_money_list_empty() -> None:
    assert get_money_list("Sem valores aqui.") == []
    assert get_money_list("") == []


def test_money_list_float_digits() -> None:
    result = get_money_list("Valor: R$ 12,7.", float_digits=0)
    assert len(result) == 1
    assert result[0]["amount"] == Decimal("13")
    assert result[0]["currency"] == "BRL"
