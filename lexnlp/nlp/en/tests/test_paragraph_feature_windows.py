"""Characterisation tests for the paragraph feature window.

These pin behaviour that is arithmetically wrong but that the bundled paragraph
segmenter depends on. Read this before "fixing" it.

Both halves of the feature builder clamp the forward window by subtracting the
window size rather than the line position, which is unrelated to how much room
is left in the document:

* ``build_paragraph_break_features`` uses ``len(lines) - line_window_post - 1``.
  For a short document that is too small, so features for lines that do exist
  are dropped. For a line near the end it is too large, and the loop's
  ``except IndexError`` quietly swallows every step past the end.
* ``get_paragraph_break_feature_names`` uses ``lines_count - line_window_post - 1``
  and has no such guard, so once the window reaches the document length the
  value goes negative and the column set loses even its zero-offset entries.

Correcting both to ``min(window, lines_available)`` was tried and reverted. The
bundled segmenter was trained on vectors produced by this arithmetic, so
widening the window at inference time feeds the model features it never saw in
training and it stops splitting on a blank line. ``test_sota_legacy_parity``
catches that: "First operative paragraph." and "Second operative paragraph."
separated by a blank line come back as one paragraph instead of two.

Fixing the arithmetic therefore requires retraining and re-exporting
``paragraph_segmenter.skops`` in the same change. Until then these tests hold
the current behaviour still so the coupling cannot be broken by accident.
"""

from unittest import TestCase

from lexnlp.nlp.en.segments.paragraphs import (
    build_paragraph_break_features,
    get_paragraph_break_feature_names,
)


def _offsets(keys) -> list[int]:
    return sorted({int(key.rsplit("_", 1)[1]) for key in keys if key.startswith("line_len_")})


class TestParagraphFeatureWindowIsCoupledToTheBundledModel(TestCase):
    def test_window_wider_than_the_remaining_lines_collapses_to_one_offset(self):
        """Arithmetically wrong: line 2 of 10 has seven lines after it."""
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=2, line_window_pre=0, line_window_post=9)

        self.assertEqual(_offsets(features), [0])

    def test_window_within_the_document_is_honoured(self):
        """When the window fits, the offsets are what you would expect."""
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=0, line_window_pre=0, line_window_post=5)

        self.assertEqual(_offsets(features), [0, 1, 2, 3, 4, 5])

    def test_backward_window_never_wraps_to_the_end_of_the_document(self):
        """The backward clamp is correct, so nothing reads lines[-1] as a predecessor."""
        lines = [f"line {index}" for index in range(10)]

        features = build_paragraph_break_features(lines, line_id=0, line_window_pre=5, line_window_post=0)

        self.assertEqual(_offsets(features), [0])

    def test_feature_names_lose_the_forward_half_when_the_window_hits_the_length(self):
        """Arithmetically wrong: the column set drops offset 0 and every positive offset."""
        names = get_paragraph_break_feature_names(lines_count=10, line_window_pre=10, line_window_post=10)

        self.assertEqual(_offsets(names), [-9, -8, -7, -6, -5, -4, -3, -2, -1])

    def test_feature_names_are_symmetric_while_the_window_fits(self):
        names = get_paragraph_break_feature_names(lines_count=10, line_window_pre=4, line_window_post=4)

        self.assertEqual(_offsets(names), [-4, -3, -2, -1, 0, 1, 2, 3, 4])

    def test_names_cover_every_offset_any_line_produces_at_the_shipped_window(self):
        """The two halves still agree at the window sizes the segmenter uses."""
        lines_count, pre, post = 10, 4, 4
        lines = [f"line {index}" for index in range(lines_count)]

        name_offsets = set(
            _offsets(get_paragraph_break_feature_names(lines_count=lines_count, line_window_pre=pre, line_window_post=post))
        )

        for line_id in range(lines_count):
            produced = set(
                _offsets(
                    build_paragraph_break_features(lines, line_id=line_id, line_window_pre=pre, line_window_post=post)
                )
            )
            self.assertTrue(
                produced <= name_offsets,
                f"line {line_id} produced offsets {sorted(produced - name_offsets)} that the column set omits",
            )
