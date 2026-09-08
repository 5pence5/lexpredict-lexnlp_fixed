"""Depth must come from the drafter's actual nesting, not from the marker text.

A clause numbered ``1.2.3`` is not inherently deeper than one numbered ``4.``,
and an ``Article`` heading is not inherently shallower than a ``(i)`` item. The
only evidence of nesting in a plain-text document is where the drafter put the
marker, so the level assigned to a clause must follow its indentation. Deriving
it from the marker string instead gets documents that nest against convention
exactly backwards.
"""

from unittest import TestCase

from lexnlp.nlp.en.segments.hierarchy import segment_document


def _labelled_levels(text: str) -> dict[str, int]:
    levels: dict[str, int] = {}
    for segment in segment_document(text).segments():
        label = (segment.label or "").strip()
        if label:
            levels[label] = segment.level
    return levels


class TestDepthComesFromNestingNotMarkers(TestCase):
    def test_multi_part_number_does_not_outrank_indentation(self):
        text = "1.2.3 Shallow clause with a deep looking number.\n    4. Deeper clause with a shallow looking number.\n"

        levels = _labelled_levels(text)

        self.assertIn("1.2.3", levels)
        self.assertIn("4.", levels)
        self.assertLess(
            levels["1.2.3"],
            levels["4."],
            "a three-part number at the margin must stay above a one-part number indented under it",
        )

    def test_same_marker_at_different_indents_gets_different_levels(self):
        text = "1. Outer clause text here.\n        1. Inner clause text here.\n"

        levels = _labelled_levels(text)

        self.assertEqual(len(levels), 1, "identical labels collapse, so compare the segments directly")

        segments = [s for s in segment_document(text).segments() if (s.label or "").strip() == "1."]
        self.assertEqual(len(segments), 2)
        self.assertLess(
            segments[0].level,
            segments[1].level,
            "the same marker text must take its depth from where it sits, not from the text",
        )

    def test_deeper_indentation_never_produces_a_shallower_level(self):
        text = (
            "10.4.7.2 First clause at the margin.\n"
            "    5. Second clause indented once.\n"
            "        99.1 Third clause indented twice.\n"
        )

        levels = _labelled_levels(text)
        ordered = [levels[label] for label in ("10.4.7.2", "5.", "99.1") if label in levels]

        self.assertEqual(len(ordered), 3, f"expected all three markers, got {sorted(levels)}")
        self.assertEqual(ordered, sorted(ordered), "levels must increase monotonically with indentation")
        self.assertEqual(len(set(ordered)), 3, "each indentation step must be its own level")
