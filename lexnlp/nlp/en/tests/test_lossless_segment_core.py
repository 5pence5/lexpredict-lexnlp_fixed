"""Focused invariants for lossless hierarchical segmentation."""

from __future__ import annotations

import statistics
import time
import unittest

from lexnlp.nlp.en.segments import hierarchy as hierarchy_core
from lexnlp.nlp.en.segments.hierarchy import (
    DocumentHierarchy,
    HierarchyManifest,
    Segment,
    SegmentKind,
    StructuralMode,
    StructuralSpan,
    StructureProfile,
    segment_document,
)


def whole_paragraph(text):
    return ((0, len(text), text),) if text else ()


def whole_sentence(text):
    return ((0, len(text), text),) if text else ()


def empty_backend(_text):
    return ()


class LosslessHierarchyTests(unittest.TestCase):
    def segment(self, text, **kwargs):
        kwargs.setdefault("paragraph_segmenter", whole_paragraph)
        kwargs.setdefault("sentence_segmenter", whole_sentence)
        kwargs.setdefault("paragraph_backend_id", "tests.whole-paragraph.v1")
        kwargs.setdefault("sentence_backend_id", "tests.whole-sentence.v1")
        return segment_document(text, **kwargs)

    def test_exact_unicode_crlf_reconstruction_and_single_paragraph(self):
        text = "Café terms apply.\r\nEmoji 📬 remains exact."
        hierarchy = self.segment(text)

        self.assertEqual(hierarchy.reconstruct(), text)
        self.assertEqual(
            "".join(hierarchy.text(leaf) for leaf in hierarchy.leaves()),
            text,
        )
        paragraphs = list(hierarchy.segments(SegmentKind.PARAGRAPH))
        self.assertEqual([(node.start, node.end) for node in paragraphs], [(0, len(text))])
        self.assertEqual(len(list(hierarchy.segments(SegmentKind.SENTENCE))), 1)

    def test_custom_backend_text_must_match_exact_slice(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.segment(
                "operative text",
                sentence_segmenter=lambda text: ((0, len(text), "different"),),
                sentence_backend_id="tests.invalid.v1",
            )

        with self.assertRaisesRegex(ValueError, "does not match"):
            segment_document(
                "operative text",
                paragraph_segmenter=lambda text: ((0, len(text), "different"),),
                sentence_segmenter=whole_sentence,
                paragraph_backend_id="tests.invalid.v1",
                sentence_backend_id="tests.whole-sentence.v1",
            )

    def test_empty_custom_backends_do_not_invent_confident_nodes(self):
        hierarchy = segment_document(
            "operative text",
            paragraph_segmenter=empty_backend,
            sentence_segmenter=empty_backend,
            paragraph_backend_id="tests.empty-paragraph.v1",
            sentence_backend_id="tests.empty-sentence.v1",
        )

        self.assertEqual([leaf.kind for leaf in hierarchy.leaves()], [SegmentKind.TEXT])
        self.assertEqual(list(hierarchy.segments(SegmentKind.PARAGRAPH)), [])
        self.assertEqual(list(hierarchy.segments(SegmentKind.SENTENCE)), [])

    def test_uppercase_numbered_heading_is_section_only(self):
        text = "3. GOVERNING LAW\nThis Agreement is governed by English law."
        hierarchy = self.segment(text)

        sections = list(hierarchy.segments(SegmentKind.SECTION))
        self.assertEqual([(node.label, node.start, node.end) for node in sections], [
            ("3. GOVERNING LAW", 0, len(text)),
        ])
        self.assertEqual(list(hierarchy.segments(SegmentKind.CLAUSE)), [])

    def test_explicit_heading_scope_is_nested_by_compatible_prefix(self):
        text = (
            "PART II\nGeneral\n"
            "ARTICLE IV-A\nCovenants\n"
            "SECTION 4.1 Duties\nAlpha applies.\n"
            "SECTION 4.2 Notices\nBeta applies."
        )
        hierarchy = self.segment(text)
        sections = list(hierarchy.segments(SegmentKind.SECTION))
        labels_and_levels = [(node.label, node.level) for node in sections]

        self.assertEqual(
            labels_and_levels,
            [
                ("PART II", 1),
                ("ARTICLE IV-A", 2),
                ("SECTION 4.1 Duties", 3),
                ("SECTION 4.2 Notices", 3),
            ],
        )
        part, article, first_section, second_section = sections
        self.assertIn(article, tuple(part.walk(SegmentKind.SECTION)))
        self.assertIn(first_section, tuple(article.walk(SegmentKind.SECTION)))
        self.assertIn(second_section, tuple(article.walk(SegmentKind.SECTION)))
        self.assertEqual(hierarchy.reconstruct(), text)

    def test_numeric_and_parenthetical_heading_depth(self):
        text = (
            "SECTION 1 General\nroot\n"
            "SECTION 1(a) First\none\n"
            "SECTION 1(b) Second\ntwo\n"
            "SECTION 2 Other\nthree"
        )
        hierarchy = self.segment(text)
        sections = list(hierarchy.segments(SegmentKind.SECTION))

        self.assertEqual(
            [(node.label, node.level) for node in sections],
            [
                ("SECTION 1 General", 1),
                ("SECTION 1(a) First", 2),
                ("SECTION 1(b) Second", 2),
                ("SECTION 2 Other", 1),
            ],
        )
        self.assertEqual(sections[0].end, sections[3].start)

    def test_statute_profile_promotes_numbered_sequence_but_default_does_not(self):
        text = "1 General duty\nbody\n2 Exceptions\nbody"
        conservative = self.segment(text)
        statute = self.segment(text, structure_profile=StructureProfile.STATUTE)

        self.assertEqual(list(conservative.segments(SegmentKind.SECTION)), [])
        self.assertEqual(
            [node.label for node in statute.segments(SegmentKind.SECTION)],
            ["1 General duty", "2 Exceptions"],
        )
        self.assertEqual(
            statute.manifest.structure_profile,
            StructureProfile.STATUTE,
        )
        self.assertIn(
            ":statute",
            statute.manifest.structural_detector_id,
        )

    def test_statute_sequence_promotes_hierarchical_descendant(self):
        text = (
            "1. Definitions\nbody\n"
            "1.1 Included term\nbody\n"
            "2. Term\nbody"
        )
        hierarchy = self.segment(text, structure_profile=StructureProfile.STATUTE)
        sections = list(hierarchy.segments(SegmentKind.SECTION))

        self.assertEqual(
            [(node.label, node.level) for node in sections],
            [
                ("1. Definitions", 1),
                ("1.1 Included term", 2),
                ("2. Term", 1),
            ],
        )
        self.assertLess(sections[0].start, sections[1].start)
        self.assertEqual(sections[0].end, sections[2].start)

    def test_clause_list_and_indentation_are_source_mapped(self):
        text = (
            "PART B — SERVICE LEVELS\n"
            "B.1 Availability\n"
            "Service applies.\n"
            "B.2 Credits\n"
            "1) First credit\n"
            "  (i) nested condition\n"
            "2) Second credit"
        )
        hierarchy = self.segment(text)
        clauses = list(hierarchy.segments(SegmentKind.CLAUSE))
        items = list(hierarchy.segments(SegmentKind.LIST_ITEM))

        self.assertEqual([node.label for node in clauses], ["B.1", "B.2"])
        self.assertEqual(
            [node.label for node in items],
            ["1)", "(i)", "2)"],
        )
        nested_start = text.index("  (i)")
        self.assertIn(nested_start, [node.start for node in items])
        self.assertEqual(hierarchy.reconstruct(), text)

    def test_injected_table_augments_builtin_schedule(self):
        text = "SCHEDULE 1 — DATA\nIntro text.\nRows here.\n"
        table_start = text.index("Rows")
        table_end = text.index("\n", table_start) + 1
        hierarchy = self.segment(
            text,
            structural_spans=(
                StructuralSpan(
                    SegmentKind.TABLE,
                    table_start,
                    table_end,
                    attributes=(("page", "7"), ("bbox", "1,2,3,4")),
                ),
            ),
            structural_mode=StructuralMode.AUGMENT,
            structural_backend_id="tests.layout.v1",
        )

        self.assertEqual(
            [node.label for node in hierarchy.segments(SegmentKind.SECTION)],
            ["SCHEDULE 1 — DATA"],
        )
        table = next(hierarchy.segments(SegmentKind.TABLE))
        self.assertEqual(
            table.attributes,
            (("page", "7"), ("bbox", "1,2,3,4")),
        )
        self.assertEqual(hierarchy.manifest.structural_mode, StructuralMode.AUGMENT)
        self.assertEqual(
            hierarchy.manifest.structural_detector_id,
            "builtin.legal_structure.v1:conservative+tests.layout.v1",
        )

    def test_replace_tree_digest_distinguishes_realised_structure(self):
        text = "operative text"
        common = dict(
            paragraph_segmenter=whole_paragraph,
            sentence_segmenter=whole_sentence,
            paragraph_backend_id="tests.whole-paragraph.v1",
            sentence_backend_id="tests.whole-sentence.v1",
            structural_backend_id="tests.layout.v1",
        )
        section = segment_document(
            text,
            structural_spans=(StructuralSpan(SegmentKind.SECTION, 0, len(text)),),
            **common,
        )
        table = segment_document(
            text,
            structural_spans=(StructuralSpan(SegmentKind.TABLE, 0, len(text)),),
            **common,
        )

        self.assertNotEqual(
            section.manifest.tree_sha256,
            table.manifest.tree_sha256,
        )

    def test_crossing_and_identical_structural_spans_fail_clearly(self):
        text = "0123456789"
        common = dict(
            paragraph_segmenter=empty_backend,
            sentence_segmenter=empty_backend,
            paragraph_backend_id="tests.empty-paragraph.v1",
            sentence_backend_id="tests.empty-sentence.v1",
            structural_backend_id="tests.layout.v1",
        )
        with self.assertRaisesRegex(ValueError, "crossing"):
            segment_document(
                text,
                structural_spans=(
                    StructuralSpan(SegmentKind.SECTION, 0, 7),
                    StructuralSpan(SegmentKind.TABLE, 5, 10),
                ),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "identical"):
            segment_document(
                text,
                structural_spans=(
                    StructuralSpan(SegmentKind.SECTION, 0, 10),
                    StructuralSpan(SegmentKind.TABLE, 0, 10),
                ),
                **common,
            )

    def test_low_level_hierarchy_rejects_duplicate_ids_and_empty_partition(self):
        inner = Segment(SegmentKind.TEXT, 0, 1)
        outer = Segment(SegmentKind.TEXT, 0, 1, (inner,))
        root = Segment(SegmentKind.DOCUMENT, 0, 1, (outer,), level=0)
        manifest = HierarchyManifest(
            "tests.tree.v1",
            StructuralMode.REPLACE,
            "tests.embedded.v1",
            "tests.embedded.v1",
        )
        with self.assertRaisesRegex(ValueError, "duplicate segment identity"):
            DocumentHierarchy("x", root, manifest)
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            DocumentHierarchy.from_segments("operative", ())

    def test_public_value_types_reject_invalid_labels_and_zero_width(self):
        with self.assertRaises(TypeError):
            Segment(SegmentKind.TEXT, 0, 1, label=123)
        with self.assertRaises(TypeError):
            StructuralSpan(SegmentKind.SECTION, 0, 1, label=123)
        with self.assertRaises(ValueError):
            Segment(SegmentKind.TEXT, 0, 0)
        with self.assertRaises(ValueError):
            StructuralSpan(SegmentKind.TABLE, 0, 0)

    def test_irrelevant_structural_backend_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            self.segment(
                "plain text",
                structural_mode=StructuralMode.AUGMENT,
                structural_backend_id="claimed.but.unused.v1",
            )


class StatuteSequenceScalingTests(unittest.TestCase):
    @staticmethod
    def timed(candidates):
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            result = hierarchy_core._sequence_numbered_heading_indices(candidates)
            samples.append(time.perf_counter() - started)
        return statistics.median(samples), result

    def test_numbered_sequence_detector_scales_linearly(self):
        small = tuple(
            hierarchy_core._NumericCandidate(index, (str(index + 1),), "Heading")
            for index in range(1500)
        )
        large = tuple(
            hierarchy_core._NumericCandidate(index, (str(index + 1),), "Heading")
            for index in range(4500)
        )

        self.timed(small)
        small_time, small_result = self.timed(small)
        large_time, large_result = self.timed(large)

        self.assertEqual(len(small_result), len(small))
        self.assertEqual(len(large_result), len(large))
        denominator = max(small_time, 1e-6)
        self.assertLess(
            large_time / denominator,
            6.0,
            (small_time, large_time),
        )


if __name__ == "__main__":
    unittest.main()
