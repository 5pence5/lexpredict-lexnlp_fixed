"""Coverage tests for lexnlp.extract.pt.distances line 103."""

from decimal import Decimal

from lexnlp.extract.pt.distances import get_distance_list, get_distances


def test_get_distances_default_branch_kilometers():
    result = list(get_distances("A obra cobre 12,5 quilômetros de estrada."))
    assert result == [(Decimal("12.5000"), "kilometer")]
    assert len(result[0]) == 2


def test_get_distances_default_branch_km_symbol():
    result = get_distance_list("Distância máxima: 100 km do litoral.")
    assert result == [(Decimal("100.0000"), "kilometer")]


def test_get_distances_list_matches_generator():
    text = "Recuo mínimo de 5 metros do alinhamento."
    assert get_distance_list(text) == list(get_distances(text))
    amount, distance_type = get_distance_list(text)[0]
    assert amount == Decimal("5.0000")
    assert distance_type == "meter"


def test_get_distances_with_sources_contrast():
    result = list(get_distances("100 km", return_sources=True))
    assert len(result) == 1
    assert result[0][0] == Decimal("100.0000")
    assert result[0][1] == "kilometer"
    assert "100" in result[0][2]


def test_get_distances_empty():
    assert list(get_distances("Sem distâncias citadas.")) == []
    assert get_distance_list("") == []
