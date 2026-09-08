"""Coverage tests for :mod:`lexnlp.extract.pt.ratios` default branch."""

from __future__ import annotations

from decimal import Decimal
from unittest import TestCase

from lexnlp.extract.pt.ratios import get_ratio_list, get_ratios


class TestPtRatiosDefaultBranch(TestCase):
    def test_get_ratios_without_sources_yields_triples(self) -> None:
        results = list(get_ratios("A proporção é 3:1 entre as partes."))
        self.assertEqual(1, len(results))
        self.assertEqual((Decimal("3"), Decimal("1"), Decimal("3")), results[0])
        self.assertEqual(3, len(results[0]))

    def test_get_ratio_list_default_has_no_source(self) -> None:
        results = get_ratio_list("Margem de 5/4 no contrato.")
        self.assertEqual(1, len(results))
        left, right, ratio = results[0]
        self.assertEqual(Decimal("5"), left)
        self.assertEqual(Decimal("4"), right)
        self.assertEqual(Decimal("1.25"), ratio)

    def test_explicit_false_matches_default(self) -> None:
        default = get_ratio_list("Razão de 2 para 1 na composição.")
        explicit = list(get_ratios("Razão de 2 para 1 na composição.", return_sources=False))
        self.assertEqual(default, explicit)
        self.assertEqual((Decimal("2"), Decimal("1"), Decimal("2")), explicit[0])
