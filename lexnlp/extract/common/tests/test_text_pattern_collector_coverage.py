__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.common.annotations.text_annotation import TextAnnotation
from lexnlp.extract.common.pattern_found import PatternFound
from lexnlp.extract.common.text_pattern_collector import TextPatternCollector
from lexnlp.utils.lines_processing.line_processor import LineOrPhrase, LineSplitParams


class DummyCollector(TextPatternCollector):
    def make_annotation_from_pattern(
        self,
        locale: str,
        ptrn: PatternFound,
        phrase: LineOrPhrase,
    ) -> TextAnnotation:
        return TextAnnotation(name=ptrn.name, locale=locale or "en", coords=(ptrn.start, ptrn.end))


def _make_match(name: str, start: int, end: int, probability: float) -> PatternFound:
    m = PatternFound()
    m.name = name
    m.start = start
    m.end = end
    m.probability = probability
    return m


def _collector() -> DummyCollector:
    return DummyCollector(parsing_functions=[], split_params=LineSplitParams())


class TestTextPatternCollectorCoverage(TestCase):
    def test_base_make_annotation_raises(self):
        collector = _collector()
        phrase = LineOrPhrase("hello", 0)
        match = _make_match("hello", 0, 5, 1.0)
        with self.assertRaises(NotImplementedError):
            TextPatternCollector.make_annotation_from_pattern(collector, "en", match, phrase)

    def test_choose_best_matches_picks_best(self):
        weak = _make_match("Alpha", 0, 5, 0.1)
        strong = _make_match("Alpha", 0, 5, 0.9)
        result = TextPatternCollector.choose_best_matches([weak, strong])
        self.assertEqual(1, len(result))
        self.assertIs(strong, result[0])
        self.assertEqual("Alpha", result[0].name)

    def test_choose_best_matches_strips_quotes(self):
        first = _make_match("  'Alpha'  ", 0, 5, 0.2)
        second = _make_match("Alpha", 0, 5, 0.8)
        result = TextPatternCollector.choose_best_matches([first, second])
        self.assertEqual(1, len(result))
        self.assertIs(second, result[0])

    def test_choose_best_matches_distinct_names_kept(self):
        first = _make_match("Alpha", 0, 5, 0.9)
        second = _make_match("Beta", 6, 10, 0.9)
        result = TextPatternCollector.choose_best_matches([first, second])
        self.assertEqual(2, len(result))

    def test_choose_more_precise_matches_short(self):
        single = [_make_match("Alpha", 0, 5, 0.9)]
        self.assertIs(single, TextPatternCollector.choose_more_precise_matches(single, "Alpha text"))
        empty: list[PatternFound] = []
        self.assertIs(empty, TextPatternCollector.choose_more_precise_matches(empty, "text"))

    def test_choose_more_precise_matches_drops_consumed(self):
        outer = _make_match("outer has inner", 0, 15, 0.5)
        inner = _make_match("inner", 6, 11, 0.5)
        text = "outer has inner"
        result = TextPatternCollector.choose_more_precise_matches([outer, inner], text)
        self.assertEqual(1, len(result))
        self.assertIs(inner, result[0])

    def test_estimate_match_quality(self):
        match = _make_match("Alpha", 0, 100, 0.5)
        self.assertEqual(400, TextPatternCollector.estimate_match_quality(match))
        match2 = _make_match("Beta", 10, 20, 1.0)
        self.assertEqual(990, TextPatternCollector.estimate_match_quality(match2))

    def test_remove_prohibited_words(self):
        collector = _collector()
        collector.prohibited_words = {"banned": True}
        good = _make_match("good", 0, 4, 1.0)
        bad = _make_match("banned", 5, 11, 1.0)
        result = collector.remove_prohibited_words([good, bad])
        self.assertEqual([good], result)
