"""The paragraph feature window must be clamped by position, not by its own size.

``build_paragraph_break_features`` builds one line's feature vector and
``get_paragraph_break_feature_names`` builds the column set those vectors are
reindexed onto. Both used to clamp the forward window by subtracting the window
size rather than the line position, which is unrelated to how much room is left
in the document. The vector builder then silently dropped features for lines
that do exist, and the name builder could go negative and drop even the
zero-offset columns.
"""

from unittest import TestCase

from lexnlp.nlp.en.segments.paragraphs import (
    get_paragraph_break_feature_names,
    build_paragraph_break_features,
)


def _offsets(keys) -> set[int]:
    return {int(key.rsplit("_", 1)[1]) for key in keys if key.startswith("line_len_")}


class TestParagraphFeatureWindow(TestCase):
    def test_forward_window_reaches_every_following_line(self):
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=2, line_window_pre=0, line_window_post=9)

        self.assertEqual(_offsets(features), set(range(0, 8)), "line 2 of 10 has seven lines after it")

    def test_forward_window_stops_at_the_last_line(self):
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=9, line_window_pre=0, line_window_post=3)

        self.assertEqual(_offsets(features), {0}, "nothing follows the last line")

    def test_backward_window_never_wraps_to_the_end_of_the_document(self):
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=0, line_window_pre=5, line_window_post=0)

        self.assertEqual(_offsets(features), {0}, "nothing precedes the first line")

    def test_feature_names_survive_a_window_wider_than_the_document(self):
        names = get_paragraph_break_feature_names(lines_count=10, line_window_pre=10, line_window_post=10)

        self.assertEqual(_offsets(names), set(range(-9, 10)))

    def test_names_cover_every_offset_any_line_actually_produces(self):
        lines_count, pre, post = 10, 4, 4
        lines = [f"line {index}" for index in range(lines_count)]

        names = get_paragraph_break_feature_names(lines_count=lines_count, line_window_pre=pre, line_window_post=post)
        name_offsets = _offsets(names)

        for line_id in range(lines_count):
            produced = _offsets(
                build_paragraph_break_features(lines, line_id=line_id, line_window_pre=pre, line_window_post=post)
            )
            self.assertTrue(
                produced <= name_offsets,
                f"line {line_id} produced offsets {sorted(produced - name_offsets)} that the column set omits",
            )
