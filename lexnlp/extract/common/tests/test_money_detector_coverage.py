"""Coverage tests for lexnlp.extract.common.money_detector missing lines."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from unittest import TestCase
from unittest.mock import patch

from lexnlp.extract.de.money import money_detector


class _FakeMatch:
    """Minimal stand-in for a ``regex`` match object."""

    def __init__(self, captures: dict[str, list[str]]) -> None:
        self._captures = captures

    def capturesdict(self) -> dict[str, list[str]]:
        return self._captures

    def span(self) -> tuple[int, int]:
        return (0, 3)


class _FakeFinder:
    """Stand-in for the compiled currency pattern; yields canned matches."""

    def __init__(self, matches: list[_FakeMatch]) -> None:
        self._matches = matches

    def finditer(self, text: str) -> Iterator[_FakeMatch]:
        return iter(self._matches)


class TestMoneyDetectorCoverage(TestCase):
    def test_match_without_currency_info_is_skipped(self) -> None:
        # The production pattern always captures a prefix, postfix or trigger
        # word, so line 68 (``continue``) is only reachable with a stubbed
        # pattern whose match carries none of them. Such a match must yield
        # no annotations.
        bare = _FakeMatch(
            {
                "text": ["100"],
                "prefix": [],
                "postfix": [],
                "trigger_word": [],
                "amount": ["100"],
            }
        )
        with patch.object(money_detector, "currency_ptn_re", _FakeFinder([bare])):
            self.assertEqual([], list(money_detector.get_money_annotations("100")))
            self.assertEqual([], list(money_detector.get_money("100")))

    def test_prefix_symbol_sets_currency(self) -> None:
        annotations = list(money_detector.get_money_annotations("$100"))
        self.assertEqual(1, len(annotations))
        self.assertEqual("USD", annotations[0].currency)
        self.assertEqual(Decimal("100"), annotations[0].amount)
        self.assertEqual("de", annotations[0].locale)

    def test_postfix_token_sets_currency(self) -> None:
        self.assertEqual([(Decimal("100"), "GBP")], list(money_detector.get_money("100 Pfunde")))

    def test_trigger_word_falls_back_to_default_currency(self) -> None:
        # "Preis" is a trigger word with no prefix/postfix, so the detector
        # falls back to the default currency (EUR for the German detector).
        found = list(money_detector.get_money("Preis 100"))
        self.assertEqual(1, len(found))
        self.assertEqual((Decimal("100"), "EUR"), found[0])

    def test_return_sources_includes_matched_text(self) -> None:
        found = list(money_detector.get_money("100 Pfunde", return_sources=True))
        self.assertEqual(1, len(found))
        amount, currency, source = found[0]
        self.assertEqual((Decimal("100"), "GBP"), (amount, currency))
        self.assertIn("100", source)
        self.assertIn("Pfunde", source)
