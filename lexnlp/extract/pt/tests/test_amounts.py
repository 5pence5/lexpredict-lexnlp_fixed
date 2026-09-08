"""Tests for :mod:`lexnlp.extract.pt.amounts`."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from decimal import Decimal
from unittest import TestCase

from lexnlp.extract.pt.amounts import (
    get_amount_annotation_list,
    text_to_number,
)


class TestTextToNumber(TestCase):
    def test_units(self):
        self.assertEqual(Decimal(1), text_to_number("um"))
        self.assertEqual(Decimal(2), text_to_number("dois"))
        self.assertEqual(Decimal(2), text_to_number("duas"))
        self.assertEqual(Decimal(9), text_to_number("nove"))

    def test_teens(self):
        self.assertEqual(Decimal(11), text_to_number("onze"))
        self.assertEqual(Decimal(15), text_to_number("quinze"))
        self.assertEqual(Decimal(19), text_to_number("dezenove"))

    def test_tens_and_units(self):
        self.assertEqual(Decimal(21), text_to_number("vinte e um"))
        self.assertEqual(Decimal(99), text_to_number("noventa e nove"))

    def test_hundreds(self):
        self.assertEqual(Decimal(100), text_to_number("cem"))
        self.assertEqual(Decimal(125), text_to_number("cento e vinte e cinco"))
        self.assertEqual(Decimal(500), text_to_number("quinhentos"))

    def test_thousands(self):
        self.assertEqual(Decimal(1000), text_to_number("mil"))
        self.assertEqual(Decimal(1250), text_to_number("mil duzentos e cinquenta"))
        self.assertEqual(Decimal(2500), text_to_number("dois mil e quinhentos"))
        self.assertEqual(Decimal(5430), text_to_number("cinco mil quatrocentos e trinta"))

    def test_millions(self):
        self.assertEqual(Decimal(1_000_000), text_to_number("um milhão"))
        self.assertEqual(Decimal(3_000_000), text_to_number("três milhões"))
        self.assertEqual(
            Decimal(1_250_000),
            text_to_number("um milhão duzentos e cinquenta mil"),
        )

    def test_billions(self):
        self.assertEqual(Decimal(1_000_000_000), text_to_number("um bilhão"))
        self.assertEqual(Decimal(2_500_000_000), text_to_number("dois bilhões e quinhentos milhões"))

    def test_trillions(self):
        self.assertEqual(Decimal(1_000_000_000_000), text_to_number("um trilhão"))
        self.assertEqual(
            Decimal(1_500_000_000_000),
            text_to_number("um trilhão e quinhentos bilhões"),
        )
        self.assertEqual(
            Decimal(7_000_000_000_000),
            text_to_number("sete trilhões"),
        )

    def test_meio_after_multiplier(self):
        # "um milhão e meio" should resolve to 1,500,000
        self.assertEqual(Decimal("1500000"), text_to_number("um milhão e meio"))

    def test_leading_fraction_with_multiplier(self):
        """``meio milhão`` / ``meia bilhão`` should consume the trailing
        multiplier rather than apply ``last_mult`` (which is 1 by default).
        Mixed ``um e meio milhão`` must combine the whole part with the
        fraction before multiplying.
        """
        self.assertEqual(Decimal(500_000), text_to_number("meio milhão"))
        self.assertEqual(Decimal(500_000_000), text_to_number("meio bilhão"))
        self.assertEqual(Decimal(500_000_000), text_to_number("meia bilhão"))
        # Whole + fraction + multiplier composition.
        self.assertEqual(Decimal(1_500_000), text_to_number("um e meio milhão"))
        self.assertEqual(Decimal(2_500_000_000), text_to_number("dois e meio bilhão"))

    def test_bare_fraction_returns_half(self):
        """``meio`` alone (no following multiplier) is just 0.5."""
        self.assertEqual(Decimal("0.5"), text_to_number("meio"))

    def test_unknown_token_returns_none(self):
        self.assertIsNone(text_to_number("xyzabc não é número"))

    def test_empty_returns_none(self):
        self.assertIsNone(text_to_number(""))
        self.assertIsNone(text_to_number(None))

    def test_pt_pt_spellings(self):
        # pt-PT spellings ("dezasseis", "dezassete", "dezanove") accepted.
        self.assertEqual(Decimal(16), text_to_number("dezasseis"))
        self.assertEqual(Decimal(19), text_to_number("dezanove"))

    def test_pre_acordo_ortografico_cinquenta(self):
        """``cinqüenta`` (with trema) is the pre-1990-Acordo spelling."""
        self.assertEqual(Decimal(50), text_to_number("cinqüenta"))
        # Combined with units it still composes correctly.
        self.assertEqual(Decimal(57), text_to_number("cinqüenta e sete"))
        # Inside a larger phrase.
        self.assertEqual(
            Decimal(2_350_000),
            text_to_number("dois milhões trezentos e cinqüenta mil"),
        )


class TestGetAmountAnnotations(TestCase):
    def test_extracts_numeric_with_multiplier(self):
        text = "O fundo soma 2,5 milhões de reais."
        ants = get_amount_annotation_list(text)
        self.assertGreaterEqual(len(ants), 1)
        # First should be the "2,5 milhões" combined match.
        self.assertEqual(Decimal("2500000"), ants[0].value)

    def test_extracts_word_amount(self):
        text = "Foram aplicados cento e vinte e cinco testes."
        ants = get_amount_annotation_list(text)
        # Pure word match for "cento e vinte e cinco" = 125
        values = [a.value for a in ants]
        self.assertIn(Decimal(125), values)

    def test_extracts_plain_numeric(self):
        text = "Total: 1.234,56 unidades."
        ants = get_amount_annotation_list(text)
        self.assertGreaterEqual(len(ants), 1)
        values = [a.value for a in ants]
        self.assertIn(Decimal("1234.56"), values)

    def test_locale_is_pt(self):
        ants = get_amount_annotation_list("100 dias")
        self.assertTrue(all(a.locale == "pt" for a in ants))
