"""Coverage tests for lexnlp.extract.en.amounts missing lines."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import lexnlp.extract.en.amounts as amounts
from lexnlp.extract.en.amounts import (
    get_amount_annotation_list,
    get_amount_annotations,
    get_amount_list,
    get_amounts,
    quantize_by_float_digit,
    text2num,
)


class TestAmountsCoverage:
    def test_fraction_zero_denominator_falls_back(self, monkeypatch) -> None:
        real = amounts.FRACTION_EXTRACT_PTN_RE

        def fake_search(s, *args, **kwargs):
            match = real.search(s, *args, **kwargs)
            if match is not None and "third" in s:

                class ZeroDenominatorMatch:
                    def groups(self) -> tuple[str, str]:
                        return ("one", "zeroth")

                    def group(self, *a, **k):
                        return match.group(*a, **k)

                return ZeroDenominatorMatch()
            return match

        monkeypatch.setattr(
            amounts,
            "FRACTION_EXTRACT_PTN_RE",
            SimpleNamespace(search=fake_search, sub=real.sub),
        )
        # "zeroth" evaluates to Decimal(0), so the real Decimal division
        # raises (caught) ZeroDivisionError and the amount falls back to 0.
        assert text2num("one-third") == Decimal(0)
        assert text2num("zeroth", search_fraction=False) == Decimal(0)

    def test_fraction_real_value_unaffected(self) -> None:
        assert text2num("one-third") == Decimal("0.3333333333333333333333333333")

    def test_half_after_magnitude(self) -> None:
        assert text2num("one million half") == Decimal("1500000.0")

    def test_quantize_trailing_zeros(self) -> None:
        assert quantize_by_float_digit(Decimal("1.00000"), 4) == Decimal("1.0")

    def test_quantize_no_rounding_needed(self) -> None:
        assert quantize_by_float_digit(Decimal("1.23"), 4) == Decimal("1.23")

    def test_quantize_invalid_operation_returns_amount(self) -> None:
        assert quantize_by_float_digit(Decimal("1E+30"), 4) == Decimal("1E+30")

    def test_get_amount_list_wrapper(self) -> None:
        text = "I paid 25 dollars and 3.5 tons"
        assert get_amount_list(text) == [Decimal("25.0"), Decimal("3.5")]
        assert get_amount_list(text) == list(get_amounts(text))

    def test_get_amount_annotations_no_extended_sources(self) -> None:
        text = "I paid 25 dollars and 3.5 tons"
        ants = list(get_amount_annotations(text, extended_sources=False))
        assert [(a.coords, a.value, a.text) for a in ants] == [
            ((7, 10), Decimal("25.0"), "25 "),
            ((22, 26), Decimal("3.5"), "3.5 "),
        ]

    def test_get_amount_annotation_list_wrapper(self) -> None:
        text = "I paid 25 dollars and 3.5 tons"
        ants = get_amount_annotation_list(text)
        expected = list(get_amount_annotations(text))
        assert [(a.coords, a.value, a.text) for a in ants] == [(a.coords, a.value, a.text) for a in expected]
        assert ants[0].text == "25 dollars"
        assert ants[1].text == "3.5 tons"
