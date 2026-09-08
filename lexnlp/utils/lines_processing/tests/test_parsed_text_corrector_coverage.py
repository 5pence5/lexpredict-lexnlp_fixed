"""Coverage tests for uncovered lines in parsed_text_corrector."""

from unittest import TestCase

from lexnlp.utils.lines_processing.parsed_text_corrector import ParsedTextCorrector
from lexnlp.utils.lines_processing.parsed_text_quality_estimator import (
    ParsedTextQualityEstimator,
)

CORRUPTED_TEXT = """1.1 Etymology

Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical
Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at

Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a
Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered

the undoubtable source."""


class TestCorrectIfCorruptedCoverage(TestCase):
    def test_corrupted_text_is_corrected(self):
        estimator = ParsedTextQualityEstimator()
        estimate = estimator.estimate_text(CORRUPTED_TEXT)
        self.assertGreaterEqual(estimate.corrupted_prob, 50)
        self.assertGreater(estimate.extra_line_breaks_prob, 50)

        corrector = ParsedTextCorrector()
        fixed = corrector.correct_if_corrupted(CORRUPTED_TEXT)
        # Real correction: the two spurious blank lines are collapsed,
        # the header blank line is kept (3 -> 1 double breaks).
        self.assertNotEqual(CORRUPTED_TEXT, fixed)
        self.assertLess(len(fixed), len(CORRUPTED_TEXT))
        self.assertEqual(3, CORRUPTED_TEXT.count("\n\n"))
        self.assertEqual(1, fixed.count("\n\n"))
        self.assertTrue(fixed.startswith("1.1 Etymology\n\n"))
        self.assertEqual(fixed, corrector.correct_line_breaks(CORRUPTED_TEXT))

    def test_clean_text_returns_unchanged(self):
        text = "Short clean line.\nSecond clean line.\n"
        corrector = ParsedTextCorrector()
        self.assertEqual(text, corrector.correct_if_corrupted(text))
