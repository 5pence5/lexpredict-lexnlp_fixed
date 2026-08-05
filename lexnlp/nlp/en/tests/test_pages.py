#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.nlp.en.segments import pages
from lexnlp.nlp.en.segments.pages import (
    build_page_break_features,
    get_page_break_feature_names,
    get_pages,
)
from lexnlp.nlp.en.segments.utils import (
    build_document_distribution,
    has_compatible_feature_width,
    resolve_model_feature_width,
)
from lexnlp.tests import lexnlp_tests


class TestPages(TestCase):
    TEST_PATH = os.path.join(lexnlp_test_path, 'lexnlp/nlp/en/tests/test_pages/')

    def test_empty_page_input_returns_none_without_model_call(self):
        with patch.object(
            pages.PAGE_SEGMENTER_MODEL,
            'predict_proba',
        ) as predictor:
            self.assertEqual([], list(get_pages('')))
        predictor.assert_not_called()

    def test_short_documents_keep_the_full_page_feature_schema(self):
        for text in ('one', 'one\ntwo', 'one\ntwo\nthree'):
            with self.subTest(text=text):
                lines = text.splitlines()
                distribution = build_document_distribution(text)
                columns = get_page_break_feature_names(
                    lines_count=len(lines),
                    line_window_pre=3,
                    line_window_post=3,
                    include_doc=distribution,
                )
                self.assertEqual(244, len(columns))
                self.assertEqual(
                    resolve_model_feature_width(pages.PAGE_SEGMENTER_MODEL),
                    len(columns),
                )
                self.assertTrue(
                    has_compatible_feature_width(
                        pages.PAGE_SEGMENTER_MODEL,
                        len(columns),
                    )
                )

                # Exercise the fitted artifact: sklearn must accept every short
                # document with the complete 244-column training schema.
                list(get_pages(text))

    def test_short_page_rows_clip_only_unavailable_feature_values(self):
        lines = ['one', 'two', 'three']
        first = build_page_break_features(lines, 0, 3, 3)
        middle = build_page_break_features(lines, 1, 3, 3)
        last = build_page_break_features(lines, 2, 3, 3)

        self.assertIn('line_len_0', first)
        self.assertIn('line_len_2', first)
        self.assertNotIn('line_len_-1', first)
        self.assertNotIn('line_len_3', first)

        self.assertIn('line_len_-1', middle)
        self.assertIn('line_len_0', middle)
        self.assertIn('line_len_1', middle)
        self.assertNotIn('line_len_-2', middle)
        self.assertNotIn('line_len_2', middle)

        self.assertIn('line_len_-2', last)
        self.assertIn('line_len_0', last)
        self.assertNotIn('line_len_-3', last)
        self.assertNotIn('line_len_1', last)

    def test_short_page_scores_preserve_public_break_behavior(self):
        cases = (
            (
                'one-line positive',
                'only',
                [[0.0, 1.0]],
                ['', 'only'],
            ),
            (
                'two-line split',
                'one\ntwo',
                [[1.0, 0.0], [0.0, 1.0]],
                ['one', 'two'],
            ),
            (
                'two-line negative',
                'one\ntwo',
                [[1.0, 0.0], [1.0, 0.0]],
                [],
            ),
        )
        for name, text, scores, expected in cases:
            with self.subTest(name=name):
                with patch.object(
                    pages.PAGE_SEGMENTER_MODEL,
                    'predict_proba',
                    return_value=scores,
                ) as predictor:
                    self.assertEqual(expected, list(get_pages(text)))
                predictor.assert_called_once()
                feature_matrix = predictor.call_args.args[0]
                self.assertEqual(
                    (len(text.splitlines()), 244),
                    feature_matrix.shape,
                )

    def test_huge_incompatible_page_window_returns_immediately(self):
        text = 'one\ntwo\nthree\nfour'
        with patch.object(
            pages,
            'build_document_distribution',
        ) as distribution_builder, patch.object(
            pages,
            'build_page_break_features',
        ) as feature_builder, patch.object(
            pages.PAGE_SEGMENTER_MODEL,
            'predict_proba',
        ) as predictor:
            detected = list(
                get_pages(
                    text,
                    window_pre=10**7,
                    window_post=0,
                )
            )
        distribution_builder.assert_not_called()
        feature_builder.assert_not_called()
        predictor.assert_not_called()
        self.assertEqual([], detected)

    def test_same_width_custom_windows_do_not_relabel_page_features(self):
        text = 'one\ntwo\nthree\nfour\nfive\nsix\nseven'
        for window_pre, window_post in ((0, 6), (2, 4), (4, 2), (6, 0)):
            with self.subTest(window_pre=window_pre, window_post=window_post):
                with patch.object(
                    pages,
                    'build_page_break_features',
                ) as feature_builder, patch.object(
                    pages.PAGE_SEGMENTER_MODEL,
                    'predict_proba',
                ) as predictor:
                    detected = list(
                        get_pages(
                            text,
                            window_pre=window_pre,
                            window_post=window_post,
                        )
                    )
                feature_builder.assert_not_called()
                predictor.assert_not_called()
                self.assertEqual([], detected)

    def test_incompatible_page_model_metadata_fails_closed(self):
        child_244 = SimpleNamespace(
            tree_=SimpleNamespace(n_features=244)
        )
        models = (
            (
                'wrong width',
                SimpleNamespace(
                    n_features_in_=243,
                    predict_proba=Mock(),
                ),
                243,
            ),
            (
                'missing width',
                SimpleNamespace(predict_proba=Mock()),
                None,
            ),
            (
                'inconsistent children',
                SimpleNamespace(
                    estimators_=[
                        child_244,
                        SimpleNamespace(
                            tree_=SimpleNamespace(n_features=243)
                        ),
                    ],
                    predict_proba=Mock(),
                ),
                None,
            ),
        )
        for name, model, resolved_width in models:
            with self.subTest(name=name):
                self.assertEqual(
                    resolved_width,
                    resolve_model_feature_width(model),
                )
                self.assertFalse(
                    has_compatible_feature_width(model, 244)
                )
                with patch.object(
                    pages,
                    'PAGE_SEGMENTER_MODEL',
                    model,
                ):
                    self.assertEqual(
                        [],
                        list(get_pages('one\ntwo')),
                    )
                model.predict_proba.assert_not_called()

    def test_exact_default_page_schema_calls_model(self):
        text = 'one\ntwo\nthree\nfour'
        with patch.object(
            pages.PAGE_SEGMENTER_MODEL,
            'predict_proba',
            return_value=[[1.0, 0.0]] * 4,
        ) as predictor:
            self.assertEqual([], list(get_pages(text)))
        predictor.assert_called_once()
        self.assertEqual((4, 244), predictor.call_args.args[0].shape)

    def test_page_examples(self):
        file_path = os.path.join(self.TEST_PATH, 'test_page_examples.csv')
        for (_i, text, _input_args, expected) in lexnlp_tests.iter_test_data_text_and_tuple(
                file_name=file_path):
            def remove_blankspace(r):
                return r.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

            # Get list of pages
            page_list = list(lexnlp_tests.benchmark_extraction_func(get_pages, text))
            assert len(page_list) == len(expected)
            clean_result = [remove_blankspace(p) for p in expected]
            for page in page_list:
                assert remove_blankspace(page) in clean_result
