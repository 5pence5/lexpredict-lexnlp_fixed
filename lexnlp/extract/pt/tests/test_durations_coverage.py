"""Coverage tests for lexnlp.extract.pt.durations line 138."""

from decimal import Decimal

from lexnlp.extract.pt.durations import _is_continuation, get_duration_annotation_list


def test_ascending_pair_not_grouped():
    """Second unit longer than the first hits `curr_days >= prev_days`."""
    ants = get_duration_annotation_list("Prazo de 6 meses e 2 anos.")
    assert len(ants) == 2
    assert [a.is_complex for a in ants] == [False, False]
    assert ants[0].duration_type_en == "month"
    assert ants[1].duration_type_en == "year"
    assert ants[0].text == "6 meses"
    assert ants[1].text == "2 anos"
    assert ants[0].duration_days == Decimal("180")
    assert ants[1].duration_days == Decimal("730")


def test_equal_units_not_grouped():
    """Equal units also hit the `>=` guard (curr == prev)."""
    ants = get_duration_annotation_list("Prazo de 2 dias e 30 dias.")
    assert len(ants) == 2
    assert ants[0].duration_type_en == "day"
    assert ants[1].duration_type_en == "day"
    assert ants[0].amount == Decimal("2")
    assert ants[1].amount == Decimal("30")


def test_descending_pair_groups_contrast():
    """Descending pair merges, proving the ascending tests exercise the guard."""
    ants = get_duration_annotation_list("Vigência de 2 anos e 6 meses, prorrogável.")
    assert len(ants) == 1
    assert ants[0].is_complex is True
    assert ants[0].duration_days == Decimal("910")


def test_is_continuation_directly():
    ants = get_duration_annotation_list("Prazo de 6 meses e 2 anos.")
    text = "Prazo de 6 meses e 2 anos."
    assert _is_continuation(ants[0], ants[1], text) is False
