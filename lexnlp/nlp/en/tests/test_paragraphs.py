#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os
import string
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

# Project imports
from lexnlp.extract.common.base_path import lexnlp_test_path
from lexnlp.nlp.en.segments import paragraphs
from lexnlp.nlp.en.segments.paragraphs import get_paragraph_list, get_paragraph_span_list, splitlines_with_spans
from lexnlp.nlp.en.segments.utils import build_document_distribution
from lexnlp.tests import lexnlp_tests


DOCUMENT_EXAMPLE_1 = "this is a test 123!"
DOCUMENT_EXAMPLE_1_RESULT_LC = {'doc_char_s': 0.2727272727272727, 'doc_char_p': 0.0, 'doc_char_a': 0.09090909090909091,
                                'doc_char_f': 0.0, 'doc_char_q': 0.0, 'doc_char_j': 0.0, 'doc_char_z': 0.0,
                                'doc_char_n': 0.0, 'doc_char_r': 0.0, 'doc_char_m': 0.0, 'doc_char_y': 0.0,
                                'doc_char_t': 0.2727272727272727, 'doc_char_b': 0.0, 'doc_char_w': 0.0,
                                'doc_char_v': 0.0, 'doc_char_c': 0.0, 'doc_char_h': 0.09090909090909091,
                                'doc_char_i': 0.18181818181818182, 'doc_char_e': 0.09090909090909091, 'doc_char_k': 0.0,
                                'doc_char_x': 0.0, 'doc_char_g': 0.0, 'doc_char_l': 0.0, 'doc_char_d': 0.0,
                                'doc_char_o': 0.0, 'doc_char_u': 0.0}
DOCUMENT_EXAMPLE_1_RESULT_DI = {'doc_char_1': 0.3333333333333333, 'doc_char_3': 0.3333333333333333,
                                'doc_char_2': 0.3333333333333333, 'doc_char_6': 0.0, 'doc_char_0': 0.0,
                                'doc_char_7': 0.0, 'doc_char_9': 0.0, 'doc_char_5': 0.0, 'doc_char_4': 0.0,
                                'doc_char_8': 0.0}
DOCUMENT_EXAMPLE_1_RESULT_CUSTOM = {'doc_char_1': 0.3333333333333333, 'doc_char_3': 0.3333333333333333,
                                    'doc_char_2': 0.3333333333333333}
DOCUMENT_EXAMPLE_1_RESULT_CUSTOM_NO_NORM = {'doc_char_1': 1, 'doc_char_3': 1, 'doc_char_2': 1}
DOCUMENT_EXAMPLE_1_RESULT_PRINT = {'doc_char_1': 0.05263157894736842, 'doc_char_Q': 0.0, 'doc_char_y': 0.0,
                                   'doc_char_W': 0.0, 'doc_char_6': 0.0, 'doc_char_j': 0.0, 'doc_char_|': 0.0,
                                   'doc_char_n': 0.0, 'doc_char_?': 0.0, 'doc_char_4': 0.0, 'doc_char__': 0.0,
                                   'doc_char_m': 0.0, 'doc_char_Y': 0.0, 'doc_char_E': 0.0, 'doc_char_[': 0.0,
                                   'doc_char_b': 0.0, 'doc_char_O': 0.0, 'doc_char_K': 0.0, 'doc_char_`': 0.0,
                                   'doc_char_h': 0.05263157894736842, 'doc_char_"': 0.0, 'doc_char_U': 0.0,
                                   'doc_char_B': 0.0, 'doc_char_V': 0.0, 'doc_char_%': 0.0, 'doc_char_8': 0.0,
                                   'doc_char_!': 0.05263157894736842, 'doc_char_<': 0.0, 'doc_char_(': 0.0,
                                   'doc_char_,': 0.0, 'doc_char_w': 0.0, 'doc_char_*': 0.0, 'doc_char_g': 0.0,
                                   'doc_char_H': 0.0, 'doc_char_I': 0.0, 'doc_char_/': 0.0,
                                   'doc_char_ ': 0.21052631578947367, 'doc_char_\x0c': 0.0, 'doc_char_L': 0.0,
                                   'doc_char_-': 0.0, 'doc_char_\\': 0.0, 'doc_char_9': 0.0, 'doc_char_5': 0.0,
                                   'doc_char_N': 0.0, 'doc_char_>': 0.0, 'doc_char_7': 0.0, 'doc_char_$': 0.0,
                                   'doc_char_\r': 0.0, 'doc_char_i': 0.10526315789473684, 'doc_char_\n': 0.0,
                                   'doc_char_.': 0.0, 'doc_char_k': 0.0, 'doc_char_C': 0.0, 'doc_char_R': 0.0,
                                   'doc_char_u': 0.0, 'doc_char_p': 0.0, 'doc_char_f': 0.0, 'doc_char_q': 0.0,
                                   'doc_char_3': 0.05263157894736842, 'doc_char_z': 0.0, 'doc_char_F': 0.0,
                                   'doc_char_r': 0.0, 'doc_char_M': 0.0, 'doc_char_;': 0.0,
                                   'doc_char_a': 0.05263157894736842, 'doc_char_D': 0.0,
                                   'doc_char_t': 0.15789473684210525, "doc_char_'": 0.0, 'doc_char_x': 0.0,
                                   'doc_char_X': 0.0, 'doc_char_~': 0.0, 'doc_char_:': 0.0, 'doc_char_&': 0.0,
                                   'doc_char_{': 0.0, 'doc_char_o': 0.0, 'doc_char_l': 0.0, 'doc_char_d': 0.0,
                                   'doc_char_^': 0.0, 'doc_char_s': 0.15789473684210525, 'doc_char_S': 0.0,
                                   'doc_char_}': 0.0, 'doc_char_v': 0.0, 'doc_char_+': 0.0, 'doc_char_P': 0.0,
                                   'doc_char_T': 0.0, 'doc_char_\t': 0.0, 'doc_char_0': 0.0, 'doc_char_G': 0.0,
                                   'doc_char_#': 0.0, 'doc_char_=': 0.0, 'doc_char_\x0b': 0.0, 'doc_char_]': 0.0,
                                   'doc_char_c': 0.0, 'doc_char_A': 0.0, 'doc_char_e': 0.05263157894736842,
                                   'doc_char_2': 0.05263157894736842, 'doc_char_J': 0.0, 'doc_char_)': 0.0,
                                   'doc_char_@': 0.0, 'doc_char_Z': 0.0}


class TestParagraphs(TestCase):
    TEST_PATH = os.path.join(lexnlp_test_path, 'lexnlp/nlp/en/tests/test_paragraphs/')

    def test_splitlines_with_spans(self):
        data = [
            ('single line', ['single line'], [(0, 11)]),
            ('1\n1', ['1', '1'], [(0, 2), (2, 3)]),
            ('2\r2', ['2', '2'], [(0, 2), (2, 3)]),
            ('3\n\r3', ['3', '3'], [(0, 3), (3, 4)]),
            ('4\r\n4', ['4', '4'], [(0, 3), (3, 4)]),
            ('5\r\r\n5', ['5', '', '5'], [(0, 2), (2, 4), (4, 5)]),
            ('\r\n\r\n\n\r', ['', '', ''], [(0, 2), (2, 4), (4, 6)]),
        ]

        for text, expected_lines, expected_spans in data:
            actual_lines, actual_spans = splitlines_with_spans(text)
            assert actual_lines == expected_lines, (
                "Actual lines do not match the expected lines for text:\n{0}".format(text)
            )
            assert actual_spans == expected_spans, (
                "Actual spans do not match the expected spans for text:\n{0}".format(text)
            )

    def test_single_line_paragraph_span_starts_at_zero(self):
        text = '2021-01-20T10:32:31.938706'

        # The one-line matrix is narrower than the fitted estimator schema and
        # must use the deterministic whole-source compatibility fallback.
        with patch.object(
            paragraphs.PARAGRAPH_SEGMENTER_MODEL,
            'predict_proba',
            return_value=[[0.0, 1.0]],
        ) as predictor:
            self.assertEqual([(0, len(text), text)], get_paragraph_span_list(text))
        predictor.assert_not_called()

    def test_document_distribution_1_lc(self):
        """
        Test lowercase letters only.
        """
        # Check all dictionaries
        assert DOCUMENT_EXAMPLE_1_RESULT_LC == lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            DOCUMENT_EXAMPLE_1,
            characters=string.ascii_lowercase,
        )

    def test_document_distribution_1_digits(self):
        """
        Test digits only.
        """
        # Check all dictionaries
        assert DOCUMENT_EXAMPLE_1_RESULT_DI == lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            DOCUMENT_EXAMPLE_1,
            characters=string.digits,
        )

    def test_document_distribution_1_custom(self):
        """
        Test custom set.
        """
        # Check all dictionaries
        assert DOCUMENT_EXAMPLE_1_RESULT_CUSTOM == lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            DOCUMENT_EXAMPLE_1,
            characters=['1', '2', '3'],
        )

    def test_document_distribution_1_custom_nn(self):
        """
        Test custom set.
        """
        # Check all dictionaries
        assert DOCUMENT_EXAMPLE_1_RESULT_CUSTOM_NO_NORM == lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            DOCUMENT_EXAMPLE_1,
            characters=['1', '2', '3'],
            norm=False,
        )

    def test_document_distribution_1_print(self):
        """
        Test all printable.
        """
        # Check all dictionaries
        assert DOCUMENT_EXAMPLE_1_RESULT_PRINT == lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            DOCUMENT_EXAMPLE_1,
            characters=string.printable,
        )

    def test_document_distribution_empty(self):
        """
        Test all printable.
        """
        # Check all dictionaries
        _ = lexnlp_tests.benchmark_extraction_func(
            build_document_distribution,
            '',
            characters=string.printable)

    def test_document_line_distribution_empty(self):
        """
        Test all printable.
        """
        # Check all dictionaries
        assert {} == lexnlp_tests.benchmark_extraction_func(
            func=build_document_distribution,
            text='',
            characters=string.printable,
        )

    def test_get_paragraphs_too_small_text_with_spans(self):
        text = '\nToo small text\n'
        for predicted in (
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ):
            with self.subTest(predicted=predicted):
                with patch.object(
                    paragraphs.PARAGRAPH_SEGMENTER_MODEL,
                    'predict_proba',
                    return_value=predicted,
                ) as predictor:
                    spans = get_paragraph_span_list(text=text)
                predictor.assert_not_called()
                self.assertEqual([(0, len(text), text)], spans)

    def test_custom_underspecified_window_uses_paragraph_fallback(self):
        text = 'one\ntwo\nthree\nfour'
        with patch.object(
            paragraphs.PARAGRAPH_SEGMENTER_MODEL,
            'predict_proba',
            return_value=[[0.0, 1.0]] * 4,
        ) as predictor:
            spans = get_paragraph_span_list(
                text,
                window_pre=0,
                window_post=0,
            )
        predictor.assert_not_called()
        self.assertEqual([(0, len(text), text)], spans)

    def test_same_width_custom_windows_do_not_relabel_model_features(self):
        text = 'one\ntwo\nthree\nfour\nfive\nsix\nseven'
        for window_pre, window_post in ((0, 6), (2, 4), (4, 2), (6, 0)):
            with self.subTest(window_pre=window_pre, window_post=window_post):
                with patch.object(
                    paragraphs.PARAGRAPH_SEGMENTER_MODEL,
                    'predict_proba',
                    return_value=[[0.0, 1.0]] * 7,
                ) as predictor:
                    spans = get_paragraph_span_list(
                        text,
                        window_pre=window_pre,
                        window_post=window_post,
                    )
                predictor.assert_not_called()
                self.assertEqual([(0, len(text), text)], spans)

    def test_feature_width_requires_consistent_forest_metadata(self):
        child_361 = SimpleNamespace(tree_=SimpleNamespace(n_features=361))
        consistent = SimpleNamespace(estimators_=[child_361, child_361])
        inconsistent = SimpleNamespace(
            estimators_=[
                child_361,
                SimpleNamespace(tree_=SimpleNamespace(n_features=360)),
            ]
        )
        incomplete = SimpleNamespace(
            estimators_=[child_361, SimpleNamespace()]
        )
        contradictory_parent = SimpleNamespace(
            n_features_in_=361,
            n_features_=360,
            estimators_=[child_361, child_361],
        )

        self.assertEqual(
            paragraphs.resolve_model_feature_width(consistent),
            361,
        )
        self.assertTrue(
            paragraphs.has_compatible_feature_width(consistent, 361)
        )
        self.assertFalse(
            paragraphs.has_compatible_feature_width(consistent, 360)
        )
        self.assertIsNone(
            paragraphs.resolve_model_feature_width(inconsistent)
        )
        self.assertIsNone(
            paragraphs.resolve_model_feature_width(incomplete)
        )
        self.assertIsNone(
            paragraphs.resolve_model_feature_width(contradictory_parent)
        )

    def test_four_lines_realise_the_complete_paragraph_model_schema(self):
        text = 'one\ntwo\nthree\nfour'
        lines, _spans = splitlines_with_spans(text)
        distribution = paragraphs.build_document_line_distribution(text)
        columns = paragraphs.get_paragraph_break_feature_names(
            lines_count=len(lines),
            line_window_pre=3,
            line_window_post=3,
            include_doc=distribution,
        )
        self.assertTrue(paragraphs.has_compatible_line_window(3, 3))
        self.assertTrue(
            paragraphs.has_compatible_feature_width(
                paragraphs.PARAGRAPH_SEGMENTER_MODEL,
                len(columns),
            )
        )

    def test_date_text(self):
        text = '2021-01-20T10:32:31.938706'
        ps = get_paragraph_list(text=text)
        self.assertEqual(text, ps[0])

    def test_paragraph_examples(self):
        file_path = os.path.join(self.TEST_PATH, 'test_paragraph_examples.csv')
        for (_i, text, _input_args, expected) in lexnlp_tests.iter_test_data_text_and_tuple(
                file_name=file_path):
            self.run_paragraph_test(text, expected)

    @classmethod
    def run_paragraph_test(cls,
                           text: str,
                           expected_paragraphs,
                           window_pre=3,
                           window_post=3):
        """
        Base test method to run against text with given results.
        """
        def remove_blankspace(r: str):
            r = r.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            while '  ' in r:
                r = r.replace('  ', ' ')
            return r.strip()

        # Get list from text
        actual_paragraphs = get_paragraph_list(
            text=text,
            window_pre=window_pre,
            window_post=window_post
        )

        actual_paragraphs = [remove_blankspace(p) for p in actual_paragraphs]
        expected_paragraphs = [remove_blankspace(p) for p in expected_paragraphs]

        assert actual_paragraphs == expected_paragraphs
