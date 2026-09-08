__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
from tempfile import NamedTemporaryFile
from unittest import TestCase

from lexnlp.extract.common.ocr_rating.lang_vector_distribution_builder import (
    LangVectorDistributionBuilder,
)

LONG_TEXT = "The quick brown fox jumps over the lazy dog. Legal contracts contain many provisions. " * 5


class TestLangVectorDistributionBuilderCoverage(TestCase):
    def test_empty_iterable_returns_none(self):
        builder = LangVectorDistributionBuilder()
        self.assertIsNone(builder.build_texts_reference_distribution([]))

    def test_all_short_texts_return_none(self):
        builder = LangVectorDistributionBuilder()
        self.assertIsNone(builder.build_texts_reference_distribution(["", "short", "x" * 99]))

    def test_short_texts_are_skipped(self):
        builder = LangVectorDistributionBuilder()
        mixed = builder.build_texts_reference_distribution(["short", LONG_TEXT])
        only_long = builder.build_texts_reference_distribution([LONG_TEXT])
        self.assertIsNotNone(mixed)
        self.assertIsNotNone(only_long)
        self.assertEqual(list(mixed.index), list(only_long.index))
        for key in only_long.index:
            self.assertAlmostEqual(float(mixed[key]), float(only_long[key]))

    def test_long_text_builds_normalized_distribution(self):
        builder = LangVectorDistributionBuilder()
        distr = builder.build_texts_reference_distribution([LONG_TEXT])
        self.assertIsNotNone(distr)
        self.assertGreater(len(distr), 0)
        self.assertAlmostEqual(float(distr.sum()), 1.0)
        self.assertTrue((distr >= 0).all())

    def test_files_with_only_short_texts_return_none(self):
        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("short")
            path = tmp.name
        try:
            builder = LangVectorDistributionBuilder()
            self.assertIsNone(builder.build_files_reference_distribution([path]))
        finally:
            os.unlink(path)
