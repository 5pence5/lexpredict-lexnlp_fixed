"""Coverage tests for :mod:`lexnlp.extract.pt.amounts` edge branches."""

from __future__ import annotations

from decimal import Decimal
from unittest import TestCase

from lexnlp.extract.pt.amounts import (
    get_amount_annotations,
    get_amount_list,
    get_amounts,
    text_to_number,
)


class TestTextToNumberEdges(TestCase):
    def test_no_word_chars_returns_none(self) -> None:
        self.assertIsNone(text_to_number("!!!"))
        self.assertIsNone(text_to_number("   "))

    def test_fraction_with_connector_before_multiplier(self) -> None:
        self.assertEqual(Decimal("500000.0"), text_to_number("meio e milhão"))
        self.assertEqual(Decimal("500000.0"), text_to_number("meio e milhao"))

    def test_connector_only_phrase_returns_none(self) -> None:
        self.assertIsNone(text_to_number("e e"))


class TestWordPhraseSkips(TestCase):
    def test_single_e_yields_nothing(self) -> None:
        self.assertEqual([], list(get_amount_annotations("e")))

    def test_double_e_yields_nothing(self) -> None:
        # "e e" passes the single-"e" guard but text_to_number returns
        # None, exercising the value-is-None skip.
        self.assertIsNone(text_to_number("e e"))
        self.assertEqual([], list(get_amount_annotations("e e")))


class TestAmountValueHelpers(TestCase):
    def test_get_amounts_yields_values(self) -> None:
        self.assertEqual(
            [Decimal("100.0000")], list(get_amounts("pagou 100 reais"))
        )

    def test_get_amounts_empty_text(self) -> None:
        self.assertEqual([], list(get_amounts("")))

    def test_get_amount_list_returns_list(self) -> None:
        self.assertEqual(
            [Decimal("100.0000")], get_amount_list("pagou 100 reais")
        )
        self.assertEqual([], get_amount_list(""))
