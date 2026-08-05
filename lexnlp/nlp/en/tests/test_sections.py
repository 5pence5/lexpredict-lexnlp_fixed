#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Section segmentation unit tests for English.

This module implements unit tests for the section segmentation code in English.

Todo:
    * More pathological and difficult cases
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import os

# Project imports
from unittest import TestCase
from unittest.mock import patch

from lexnlp import get_module_path
from lexnlp.nlp.en.segments.sections import (
    DocumentSection,
    SectionSegmenterModel,
    build_section_break_features,
    find_section_titles,
    get_section_feature_names,
    get_section_spans,
    get_sections,
)
from lexnlp.nlp.en.segments.sentences import get_sentence_span_list
from lexnlp.nlp.en.segments.utils import (
    build_document_line_distribution,
    has_compatible_feature_width,
    has_compatible_line_window,
    resolve_model_feature_width,
)
from lexnlp.tests import lexnlp_tests


class TestSectionSpans(TestCase):

    @staticmethod
    def get_text(path):
        base_path = get_module_path()
        with open(os.path.join(base_path, "../test_data", path), "rb") as f:
            return f.read().decode("utf-8")

    def test_file_1(self):
        text = self.get_text('1582586_2015-08-31')
        sections = list(lexnlp_tests.benchmark('get_sections(text)', get_sections, text))
        num_sections = len(sections)
        assert num_sections == 23

    def test_file_2(self):
        text = self.get_text('1031296_2004-11-04')
        sections = list(lexnlp_tests.benchmark('get_sections(text)', get_sections, text))
        num_sections = len(sections)
        assert num_sections == 11

    def test_file_3(self):
        text = self.get_text('1100644_2016-11-21')
        sections = list(lexnlp_tests.benchmark('get_sections(text)', get_sections, text))
        num_sections = len(sections)
        assert num_sections == 72

    def test_file_4_use_ml(self):
        text = self.get_text('test_get_section_spans_1.txt')

        # test all sections
        sections = list(get_section_spans(text))
        print(f'{len(sections)} sections are found')
        for s in sections:
            print(f'Section #{s.start}, "{s.title}"')
        self.assertEqual(len(sections), 207)

        # test only sections with titles
        sections = list(get_section_spans(text, skip_empty_headers=True))
        self.assertEqual(len([i for i in sections if i.title is None]), 0)

        self.assertEqual(
            sections[1],
            DocumentSection(
                start=2280,
                end=2340,
                title='SECTION 2',
                title_start=2280,
                title_end=2289,
                level=1,
                abs_level=3,
                text='SECTION 2.  Letters of Credit........................... 15\n'))

    def test_file_4_use_regex(self):
        text = self.get_text('test_get_section_spans_1.txt')

        # test all sections
        sections = list(get_section_spans(text, use_ml=False))
        self.assertEqual(len(sections), 554)

        self.assertEqual(
            sections[2],
            DocumentSection(
                start=1378,
                end=1438,
                title='SECTION 1',
                title_start=1378,
                title_end=1387,
                level=2,
                abs_level=3,
                text='SECTION 1.  Amount and Terms of Credit..................  1\n'))

    def test_bad_text(self):
        text = 'text'
        sections = list(get_section_spans(text))
        self.assertEqual(sections, [])

    def test_short_documents_keep_the_full_section_feature_schema(self):
        for text in ('text', 'one\ntwo'):
            with self.subTest(text=text):
                lines = text.splitlines()
                distribution = build_document_line_distribution(text)
                columns = get_section_feature_names(
                    lines_count=len(lines),
                    line_window_pre=3,
                    line_window_post=3,
                    include_doc=distribution,
                )
                self.assertEqual(len(columns), 369)
                self.assertEqual(
                    len(columns),
                    resolve_model_feature_width(
                        SectionSegmenterModel.SECTION_SEGMENTER_MODEL
                    ),
                )

        lines = ["one", "two"]
        first = build_section_break_features(lines, 0, 3, 3)
        second = build_section_break_features(lines, 1, 3, 3)
        self.assertIn("line_len_0", first)
        self.assertIn("line_len_1", first)
        self.assertNotIn("line_len_-1", first)
        self.assertIn("line_len_-1", second)
        self.assertIn("line_len_0", second)
        self.assertNotIn("line_len_1", second)

    def test_empty_section_input_returns_none_without_model_call(self):
        with patch.object(
            SectionSegmenterModel.SECTION_SEGMENTER_MODEL,
            'predict_proba',
        ) as predictor:
            self.assertEqual([], list(get_sections('')))
        predictor.assert_not_called()

    def test_huge_incompatible_section_window_returns_immediately(self):
        text = 'one\ntwo\nthree\nfour'
        with patch.object(
            SectionSegmenterModel.SECTION_SEGMENTER_MODEL,
            'predict_proba',
        ) as predictor:
            detected = list(
                get_sections(
                    text,
                    window_pre=10**7,
                    window_post=0,
                )
            )
        predictor.assert_not_called()
        self.assertEqual(detected, [])

    def test_custom_underspecified_window_yields_no_ml_section(self):
        text = 'one\ntwo\nthree\nfour'
        with patch.object(
            SectionSegmenterModel.SECTION_SEGMENTER_MODEL,
            'predict_proba',
            return_value=[[0.0, 1.0]] * 4,
        ) as predictor:
            detected = list(
                get_sections(text, window_pre=0, window_post=0)
            )
        predictor.assert_not_called()
        self.assertEqual(detected, [])

    def test_same_width_custom_windows_do_not_relabel_section_features(self):
        text = 'one\ntwo\nthree\nfour\nfive\nsix\nseven'
        for window_pre, window_post in ((0, 6), (2, 4), (4, 2), (6, 0)):
            with self.subTest(window_pre=window_pre, window_post=window_post):
                with patch.object(
                    SectionSegmenterModel.SECTION_SEGMENTER_MODEL,
                    'predict_proba',
                    return_value=[[0.0, 1.0]] * 7,
                ) as predictor:
                    detected = list(
                        get_sections(
                            text,
                            window_pre=window_pre,
                            window_post=window_post,
                        )
                    )
                predictor.assert_not_called()
                self.assertEqual(detected, [])

    def test_four_lines_realise_the_complete_section_model_schema(self):
        text = 'one\ntwo\nthree\nfour'
        lines = text.splitlines()
        distribution = build_document_line_distribution(text)
        columns = get_section_feature_names(
            lines_count=len(lines),
            line_window_pre=3,
            line_window_post=3,
            include_doc=distribution,
        )
        self.assertTrue(has_compatible_line_window(3, 3))
        self.assertTrue(
            has_compatible_feature_width(
                SectionSegmenterModel.SECTION_SEGMENTER_MODEL,
                len(columns),
            )
        )

    def test_title_start_end(self):
        text = self.get_text('lexnlp/nlp/en/tests/test_sections/skewed_document.txt')
        sentence_spans = get_sentence_span_list(text)
        sections = list(get_section_spans(
            text, use_ml=False, return_text=False, skip_empty_headers=True))
        self.assertGreater(len(sections), 3)
        # test title coordinates before enhancing titles ...
        for sect in sections:
            title = text[sect.title_start: sect.title_end]
            self.assertEqual(sect.title, title)

        # ... and after enhancing
        find_section_titles(sections, sentence_spans, text)
        for sect in sections:
            title = text[sect.title_start: sect.title_end]
            self.assertEqual(sect.title, title)
