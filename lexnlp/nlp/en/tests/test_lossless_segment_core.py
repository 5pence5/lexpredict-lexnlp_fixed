"""Focused invariants for lossless hierarchical segmentation."""

from __future__ import annotations

import dis
import hashlib
import statistics
import time
import unittest
from dataclasses import replace

from lexnlp.nlp.en.segments import chunks as chunks_core
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

from lexnlp.nlp.en.segments.chunks import (
    ChunkProvenance,
    ContainerPolicy,
    SegmentReference,
    TokenCounterPolicy,
    TokenSearchLimitExceeded,
    chunk_document,
    compute_chunk_metadata_sha256,
    compute_provenance_sha256,
    iter_chunks,
    reconstruct_chunks,
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
        nested_line_start = text.index("  (i)")
        nested_start = text.index("(i)")
        self.assertNotEqual(nested_line_start, nested_start)
        self.assertIn(nested_start, [node.start for node in items])
        self.assertNotIn(nested_line_start, [node.start for node in items])
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


class LosslessChunkTests(unittest.TestCase):
    def options(self):
        return {
            "paragraph_segmenter": whole_paragraph,
            "sentence_segmenter": whole_sentence,
            "paragraph_backend_id": "tests.whole-paragraph.v1",
            "sentence_backend_id": "tests.whole-sentence.v1",
        }

    def test_default_character_budget_hard_splits_oversized_atomic_text(self):
        chunks = chunk_document(
            "x" * 9001,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual([len(chunk.text) for chunk in chunks], [4000, 4000, 1001])
        self.assertTrue(all(chunk.unit_count == len(chunk.text) for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), "x" * 9001)

    def test_character_overlap_is_exact_and_reconstructs_fresh_content_once(self):
        text = "abcdefghij"
        chunks = chunk_document(
            text,
            max_chars=5,
            overlap_chars=2,
            respect_boundaries=False,
            document_id="doc-1",
            **self.options(),
        )
        self.assertEqual(
            [(chunk.start, chunk.new_content_start, chunk.end) for chunk in chunks],
            [(0, 0, 5), (3, 5, 8), (6, 8, 10)],
        )
        self.assertEqual([chunk.unit_count for chunk in chunks], [5, 5, 4])
        self.assertEqual(reconstruct_chunks(chunks), text)
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))
        self.assertTrue(
            all(chunk.manifest.source_id.startswith("doc-1@sha256:") for chunk in chunks)
        )

    def test_preserve_policy_never_packs_complete_section_siblings(self):
        text = (
            "1. FIRST\nAlpha applies.\n\n"
            "2. SECOND\nBeta applies.\n\n"
            "3. THIRD\nGamma applies."
        )
        expected = [(0, 25), (25, 50), (50, 73)]
        for budget in (30, 60, 100):
            with self.subTest(budget=budget):
                chunks = chunk_document(
                    text,
                    max_chars=budget,
                    overlap_chars=5 if budget == 30 else 0,
                    **self.options(),
                )
                self.assertEqual(
                    [(chunk.start, chunk.end) for chunk in chunks],
                    expected,
                )
                self.assertEqual(
                    [chunk.new_content_start for chunk in chunks],
                    [0, 25, 50],
                )
                self.assertEqual(reconstruct_chunks(chunks), text)
        packed = chunk_document(
            text,
            max_chars=100,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
            **self.options(),
        )
        self.assertEqual([(chunk.start, chunk.end) for chunk in packed], [(0, 73)])

    def test_overlap_provenance_distinguishes_context_from_primary_section(self):
        text = (
            "1. FIRST\nAlpha applies.\n\n"
            "2. SECOND\nBeta applies.\n\n"
            "3. THIRD\nGamma applies."
        )
        chunks = chunk_document(
            text,
            max_chars=30,
            overlap_chars=5,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
            **self.options(),
        )
        second = chunks[1]
        self.assertEqual(second.start, 20)
        self.assertEqual(second.new_content_start, 25)
        self.assertEqual(second.content_provenance.section_labels, ("2. SECOND",))
        self.assertEqual(second.overlap_provenance.section_labels, ("1. FIRST",))
        self.assertEqual(
            second.provenance.section_labels,
            ("1. FIRST", "2. SECOND"),
        )
        self.assertIs(second.context_provenance, second.overlap_provenance)

    def test_token_mode_requires_explicit_counter_and_identity(self):
        with self.assertRaisesRegex(ValueError, "explicit token_counter"):
            chunk_document("abcdef", max_tokens=2, **self.options())
        with self.assertRaisesRegex(ValueError, "require max_tokens"):
            chunk_document(
                "abcdef",
                max_chars=2,
                token_counter=len,
                token_counter_id="tests.len.v1",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
                **self.options(),
            )
        with self.assertRaisesRegex(ValueError, "token_counter_id"):
            chunk_document(
                "abcdef",
                max_tokens=2,
                token_counter=len,
                **self.options(),
            )
        with self.assertRaisesRegex(ValueError, "token_counter_policy"):
            chunk_document(
                "abcdef",
                max_tokens=2,
                token_counter=len,
                token_counter_id="tests.len.v1",
                **self.options(),
            )
        chunks = chunk_document(
            "abcdefghij",
            max_tokens=4,
            token_counter=len,
            token_counter_id="tests.len.v1",
            token_counter_policy=TokenCounterPolicy.MONOTONIC,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual([chunk.unit_count for chunk in chunks], [4, 4, 2])
        self.assertEqual(reconstruct_chunks(chunks), "abcdefghij")

    def test_non_monotonic_black_box_counter_still_obeys_strict_budget(self):
        def non_monotonic(text):
            return 1 if len(text) % 5 == 0 else len(text)

        source = "abcdefghijklmnopq"
        chunks = chunk_document(
            source,
            max_tokens=2,
            token_counter=non_monotonic,
            token_counter_id="tests.non-monotonic.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=2_000,
            token_search_max_input_bytes=20_000,
            token_search_max_steps=10_000,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.unit_count <= 2 for chunk in chunks))
        self.assertEqual(reconstruct_chunks(chunks), source)

    def test_token_counter_work_is_near_linear_and_cache_is_per_chunk(self):
        calls = 0
        counted_characters = 0

        def measured_len(text):
            nonlocal calls, counted_characters
            calls += 1
            counted_characters += len(text)
            return len(text)

        source = "a" * 100000
        chunks = chunk_document(
            source,
            max_tokens=100,
            token_counter=measured_len,
            token_counter_id="tests.measured-len.v1",
            token_counter_policy=TokenCounterPolicy.MONOTONIC,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual(len(chunks), 1000)
        self.assertLess(calls, 25000)
        self.assertLess(counted_characters, len(source) * 20)
        self.assertEqual(reconstruct_chunks(chunks), source)

    def test_layout_attributes_survive_into_chunk_references(self):
        text = "name | value\nA | 1\n"
        table = StructuralSpan(
            SegmentKind.TABLE,
            0,
            len(text),
            attributes=(("page", "7"), ("bbox", "1,2,3,4")),
        )
        chunk = chunk_document(
            text,
            max_chars=100,
            structural_spans=(table,),
            structural_backend_id="tests.layout.v1",
            **self.options(),
        )[0]
        reference = next(
            item
            for item in chunk.provenance.segments
            if item.kind is SegmentKind.TABLE
        )
        self.assertEqual(reference.attributes, table.attributes)
        self.assertIn(reference, chunk.content_provenance.segments)

    def test_manifest_and_chunk_ids_cover_source_tree_and_configuration(self):
        text = "operative text"
        char_four = chunk_document(
            text,
            max_chars=4,
            respect_boundaries=False,
            document_id="contract",
            **self.options(),
        )[0]
        char_five = chunk_document(
            text,
            max_chars=5,
            respect_boundaries=False,
            document_id="contract",
            **self.options(),
        )[0]
        revised = chunk_document(
            "operative text revised",
            max_chars=4,
            respect_boundaries=False,
            document_id="contract",
            **self.options(),
        )[0]
        self.assertNotEqual(char_four.manifest.manifest_id, char_five.manifest.manifest_id)
        self.assertNotEqual(char_four.chunk_id, char_five.chunk_id)
        self.assertNotEqual(char_four.manifest.source_id, revised.manifest.source_id)

        section = chunk_document(
            text,
            max_chars=100,
            structural_spans=(StructuralSpan(SegmentKind.SECTION, 0, len(text)),),
            structural_backend_id="tests.layout.v1",
            **self.options(),
        )[0]
        table = chunk_document(
            text,
            max_chars=100,
            structural_spans=(StructuralSpan(SegmentKind.TABLE, 0, len(text)),),
            structural_backend_id="tests.layout.v1",
            **self.options(),
        )[0]
        self.assertNotEqual(section.manifest.manifest_id, table.manifest.manifest_id)
        self.assertNotEqual(section.chunk_id, table.chunk_id)

    def test_text_digest_authenticates_overlap_only_bytes_and_changes_id(self):
        chunks = chunk_document(
            "abcdefghij",
            max_chars=5,
            overlap_chars=2,
            respect_boundaries=False,
            **self.options(),
        )
        original = chunks[1]
        mutated_text = "X" + original.text[1:]
        mutated_text_sha = hashlib.sha256(
            mutated_text.encode("utf-8", "surrogatepass")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "chunk metadata"):
            replace(
                original,
                text=mutated_text,
                text_sha256=mutated_text_sha,
            )
        new_metadata = compute_chunk_metadata_sha256(
            manifest=original.manifest,
            index=original.index,
            start=original.start,
            end=original.end,
            new_content_start=original.new_content_start,
            unit_count=original.unit_count,
            text_sha256=mutated_text_sha,
            provenance_sha256=original.provenance_sha256,
        )
        authenticated = replace(
            original,
            text=mutated_text,
            text_sha256=mutated_text_sha,
            chunk_metadata_sha256=new_metadata,
        )
        self.assertNotEqual(authenticated.chunk_id, original.chunk_id)
        self.assertEqual(authenticated.content, original.content)

    def test_provenance_digest_rejects_stale_fabrication_and_changes_id(self):
        original = chunk_document(
            "abcdefghij",
            max_chars=5,
            overlap_chars=2,
            respect_boundaries=False,
            **self.options(),
        )[1]
        fake = SegmentReference("text:0:1", SegmentKind.TEXT, 0, 1, label="FAKE")
        fabricated_full = ChunkProvenance((fake, *original.provenance.segments))
        with self.assertRaisesRegex(ValueError, "provenance partitions"):
            replace(original, provenance=fabricated_full)

        provenance_sha = compute_provenance_sha256(
            fabricated_full,
            original.content_provenance,
            original.overlap_provenance,
        )
        metadata_sha = compute_chunk_metadata_sha256(
            manifest=original.manifest,
            index=original.index,
            start=original.start,
            end=original.end,
            new_content_start=original.new_content_start,
            unit_count=original.unit_count,
            text_sha256=original.text_sha256,
            provenance_sha256=provenance_sha,
        )
        authenticated = replace(
            original,
            provenance=fabricated_full,
            provenance_sha256=provenance_sha,
            chunk_metadata_sha256=metadata_sha,
        )
        self.assertNotEqual(authenticated.chunk_id, original.chunk_id)

    def test_public_chunk_invariants_and_authenticated_reconstruction(self):
        chunks = chunk_document(
            "abcdefghij",
            max_chars=4,
            respect_boundaries=False,
            **self.options(),
        )
        with self.assertRaisesRegex(ValueError, "unit_count"):
            replace(chunks[0], unit_count=999)
        with self.assertRaisesRegex(ValueError, "overlap"):
            replace(chunks[0], new_content_start=1)
        with self.assertRaisesRegex(ValueError, "source SHA-256"):
            reconstruct_chunks(chunks[:1])
        with self.assertRaisesRegex(ValueError, "ordered"):
            reconstruct_chunks(tuple(reversed(chunks)))


class CorrectiveCoreRegressionTests(unittest.TestCase):
    def options(self):
        return {
            "paragraph_segmenter": whole_paragraph,
            "sentence_segmenter": whole_sentence,
            "paragraph_backend_id": "tests.whole-paragraph.v1",
            "sentence_backend_id": "tests.whole-sentence.v1",
        }

    def test_table_rows_take_precedence_over_outline_markers(self):
        text = "1. introduction\na | b | c\n2. | d | e\n"
        hierarchy = segment_document(text, **self.options())

        self.assertEqual(hierarchy.reconstruct(), text)
        clauses = list(hierarchy.segments(SegmentKind.CLAUSE))
        tables = list(hierarchy.segments(SegmentKind.TABLE))
        self.assertEqual([node.label for node in clauses], ["1."])
        self.assertEqual(len(tables), 1)
        self.assertLessEqual(clauses[0].start, tables[0].start)
        self.assertLessEqual(tables[0].end, clauses[0].end)

    def test_typed_digest_framing_distinguishes_none_and_literal_sentinel(self):
        source_text = "x"
        none_tree = DocumentHierarchy.from_segments(
            source_text,
            (Segment(SegmentKind.TEXT, 0, 1, label=None),),
        )
        literal_tree = DocumentHierarchy.from_segments(
            source_text,
            (Segment(SegmentKind.TEXT, 0, 1, label="<none>"),),
        )
        self.assertNotEqual(
            none_tree.manifest.tree_sha256,
            literal_tree.manifest.tree_sha256,
        )

        none_reference = SegmentReference(
            "text:0:1", SegmentKind.TEXT, 0, 1, label=None
        )
        literal_reference = SegmentReference(
            "text:0:1", SegmentKind.TEXT, 0, 1, label="<none>"
        )
        empty = ChunkProvenance()
        none_digest = compute_provenance_sha256(
            ChunkProvenance((none_reference,)), empty, empty
        )
        literal_digest = compute_provenance_sha256(
            ChunkProvenance((literal_reference,)), empty, empty
        )
        self.assertNotEqual(none_digest, literal_digest)

        none_chunk = chunk_document(
            source_text,
            max_chars=10,
            structural_spans=(
                StructuralSpan(
                    SegmentKind.SECTION,
                    0,
                    1,
                    label=None,
                ),
            ),
            structural_backend_id="tests.typed-tree.v1",
            **self.options(),
        )[0]
        literal_chunk = chunk_document(
            source_text,
            max_chars=10,
            structural_spans=(
                StructuralSpan(
                    SegmentKind.SECTION,
                    0,
                    1,
                    label="<none>",
                ),
            ),
            structural_backend_id="tests.typed-tree.v1",
            **self.options(),
        )[0]
        self.assertNotEqual(
            none_chunk.manifest.manifest_id,
            literal_chunk.manifest.manifest_id,
        )
        self.assertNotEqual(none_chunk.chunk_id, literal_chunk.chunk_id)

    def test_backend_identity_requires_execution_and_falsey_callables_are_used(self):
        with self.assertRaisesRegex(ValueError, "corresponding backend"):
            segment_document(
                "text",
                sentence_backend_id="claimed.sentence.v1",
                paragraph_segmenter=whole_paragraph,
                paragraph_backend_id="tests.whole-paragraph.v1",
            )
        with self.assertRaisesRegex(ValueError, "corresponding backend"):
            segment_document(
                "text",
                paragraph_backend_id="claimed.paragraph.v1",
                sentence_segmenter=whole_sentence,
                sentence_backend_id="tests.whole-sentence.v1",
            )

        defaults = segment_document(
            "",
            sentence_backend_id="lexnlp.sentences.get_sentence_span:v1",
            paragraph_backend_id="builtin.blank_lines.v1",
        )
        self.assertEqual(
            defaults.manifest.sentence_backend_id,
            "lexnlp.sentences.get_sentence_span:v1",
        )
        self.assertEqual(
            defaults.manifest.paragraph_backend_id,
            "builtin.blank_lines.v1",
        )

        class FalseyBackend:
            def __init__(self, backend_id, result_factory):
                self.backend_id = backend_id
                self.result_factory = result_factory
                self.calls = 0

            def __bool__(self):
                return False

            def __call__(self, text):
                self.calls += 1
                return self.result_factory(text)

        paragraph = FalseyBackend(
            "tests.falsey-paragraph.v1",
            lambda text: ((0, len(text), text),) if text else (),
        )
        sentence = FalseyBackend(
            "tests.falsey-sentence.v1",
            lambda _text: (),
        )
        hierarchy = segment_document(
            "operative text",
            paragraph_segmenter=paragraph,
            sentence_segmenter=sentence,
        )
        self.assertEqual(paragraph.calls, 1)
        self.assertEqual(sentence.calls, 1)
        self.assertEqual(
            hierarchy.manifest.paragraph_backend_id,
            paragraph.backend_id,
        )
        self.assertEqual(
            hierarchy.manifest.sentence_backend_id,
            sentence.backend_id,
        )
        self.assertEqual(
            [leaf.kind for leaf in hierarchy.leaves()],
            [SegmentKind.TEXT],
        )

    def test_preserve_policy_finds_table_beneath_unprotected_list_item(self):
        text = "lead\nrow | value | note\ntail"
        table_start = text.index("row")
        table_end = text.index("\n", table_start) + 1
        hierarchy = segment_document(
            text,
            structural_spans=(
                StructuralSpan(SegmentKind.LIST_ITEM, 0, len(text), label="(a)"),
                StructuralSpan(
                    SegmentKind.TABLE,
                    table_start,
                    table_end,
                    attributes=(("page", "1"),),
                ),
            ),
            structural_backend_id="tests.nested-layout.v1",
            **self.options(),
        )
        preserved = chunk_document(
            hierarchy,
            max_chars=100,
            container_policy=ContainerPolicy.PRESERVE,
        )
        packed = chunk_document(
            hierarchy,
            max_chars=100,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )

        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in preserved],
            [(0, table_start), (table_start, table_end), (table_end, len(text))],
        )
        self.assertEqual([(chunk.start, chunk.end) for chunk in packed], [(0, len(text))])
        table_reference = next(
            reference
            for reference in preserved[1].provenance.segments
            if reference.kind is SegmentKind.TABLE
        )
        self.assertEqual(table_reference.attributes, (("page", "1"),))
        self.assertEqual(reconstruct_chunks(preserved), text)

    @staticmethod
    def timed(operation):
        samples = []
        result = None
        for _ in range(3):
            started = time.perf_counter()
            result = operation()
            samples.append(time.perf_counter() - started)
        return statistics.median(samples), result

    def test_heading_index_keeps_thousands_of_preserved_chunks_subquadratic(self):
        def hierarchy_for(size):
            text = "".join(
                f"{index}. H{index}\nbody\n"
                for index in range(1, size + 1)
            )
            return segment_document(
                text,
                paragraph_segmenter=empty_backend,
                sentence_segmenter=empty_backend,
                paragraph_backend_id="tests.empty-paragraph.v1",
                sentence_backend_id="tests.empty-sentence.v1",
            )

        small = hierarchy_for(600)
        large = hierarchy_for(1800)
        self.timed(lambda: chunk_document(small, max_chars=32))
        small_time, small_chunks = self.timed(
            lambda: chunk_document(small, max_chars=32)
        )
        large_time, large_chunks = self.timed(
            lambda: chunk_document(large, max_chars=32)
        )

        self.assertEqual(len(small_chunks), 600)
        self.assertEqual(len(large_chunks), 1800)
        self.assertLess(
            large_time / max(small_time, 1e-6),
            6.0,
            (small_time, large_time),
        )

    def test_table_parent_sweep_is_subquadratic(self):
        def detect(size):
            text = "".join(
                f"{index}. introduction\nA | B | C\n\n"
                for index in range(1, size + 1)
            )
            return segment_document(
                text,
                paragraph_segmenter=empty_backend,
                sentence_segmenter=empty_backend,
                paragraph_backend_id="tests.empty-paragraph.v1",
                sentence_backend_id="tests.empty-sentence.v1",
            )

        self.timed(lambda: detect(300))
        small_time, small = self.timed(lambda: detect(300))
        large_time, large = self.timed(lambda: detect(900))

        self.assertEqual(len(list(small.segments(SegmentKind.TABLE))), 300)
        self.assertEqual(len(list(large.segments(SegmentKind.TABLE))), 900)
        self.assertEqual(small.reconstruct(), small.source)
        self.assertEqual(large.reconstruct(), large.source)
        self.assertLess(
            large_time / max(small_time, 1e-6),
            6.0,
            (small_time, large_time),
        )


class SecondCorrectiveCoreRegressionTests(unittest.TestCase):
    def options(self):
        return {
            "paragraph_segmenter": whole_paragraph,
            "sentence_segmenter": whole_sentence,
            "paragraph_backend_id": "tests.whole-paragraph.v1",
            "sentence_backend_id": "tests.whole-sentence.v1",
        }

    def test_uppercase_table_row_cannot_become_crossing_heading(self):
        text = "1. INTRODUCTION\na | b | c\n2. | D | E\n"
        hierarchy = segment_document(text, **self.options())

        self.assertEqual(hierarchy.reconstruct(), text)
        self.assertEqual(
            [node.label for node in hierarchy.segments(SegmentKind.SECTION)],
            ["1. INTRODUCTION"],
        )
        tables = list(hierarchy.segments(SegmentKind.TABLE))
        self.assertEqual(len(tables), 1)
        self.assertEqual(hierarchy.text(tables[0]), "a | b | c\n2. | D | E\n")

    def test_zero_width_overlap_has_no_overlap_provenance(self):
        chunks = chunk_document(
            "abcdef",
            max_chars=3,
            overlap_chars=0,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual(len(chunks), 2)
        second = chunks[1]
        self.assertEqual(second.overlap_span, (3, 3))
        self.assertEqual(second.overlap_char_count, 0)
        self.assertEqual(second.overlap_provenance.segments, ())
        self.assertEqual(second.context_provenance.segments, ())
        self.assertTrue(second.provenance.segments)
        self.assertTrue(second.content_provenance.segments)
        self.assertEqual(reconstruct_chunks(chunks), "abcdef")

    def test_provenance_walker_does_not_slice_remaining_siblings(self):
        opnames = {
            instruction.opname
            for instruction in dis.get_instructions(
                chunks_core._HierarchyIndex.references
            )
        }
        self.assertTrue(
            {"BUILD_SLICE", "BINARY_SLICE"}.isdisjoint(opnames),
            opnames,
        )


class FinalCoreCorrectionTests(unittest.TestCase):
    def options(self):
        return {
            "paragraph_segmenter": whole_paragraph,
            "sentence_segmenter": whole_sentence,
            "paragraph_backend_id": "tests.whole-paragraph.v1",
            "sentence_backend_id": "tests.whole-sentence.v1",
        }

    def test_initial_all_caps_document_title_scopes_to_next_heading(self):
        text = (
            "MASTER SERVICES AGREEMENT\n"
            "Preamble terms apply.\n"
            "1. SCOPE\n"
            "Operative body.\n"
        )
        hierarchy = segment_document(text, **self.options())
        sections = list(hierarchy.segments(SegmentKind.SECTION))

        self.assertEqual(hierarchy.reconstruct(), text)
        self.assertEqual(
            [section.label for section in sections],
            ["MASTER SERVICES AGREEMENT", "1. SCOPE"],
        )
        self.assertEqual(sections[0].start, 0)
        self.assertEqual(sections[0].end, sections[1].start)
        self.assertEqual(sections[1].end, len(text))
        self.assertIn(
            ("heading_type", "DOCUMENT_TITLE"),
            sections[0].attributes,
        )

    def test_parenthetical_decimal_clauses_are_detected(self):
        text = (
            "SECTION 4 TERMS\n"
            "4.1(a) First obligation\n"
            "4.2(a) Second obligation\n"
        )
        hierarchy = segment_document(text, **self.options())
        clauses = list(hierarchy.segments(SegmentKind.CLAUSE))

        self.assertEqual(hierarchy.reconstruct(), text)
        self.assertEqual(
            [clause.label for clause in clauses],
            ["4.1(a)", "4.2(a)"],
        )
        self.assertEqual(clauses[0].start, text.index("4.1(a)"))
        self.assertEqual(clauses[0].end, text.index("4.2(a)"))
        self.assertEqual(clauses[1].end, len(text))

    def test_indented_roman_and_bullet_lists_preserve_marker_offsets_and_nesting(self):
        text = (
            "SECTION 1 LISTS\n"
            "(b) Parent\n"
            "    (i) Child one\n"
            "    (ii) Child two\n"
            "(c) Next\n"
            "• Bullet\n"
        )
        hierarchy = segment_document(text, **self.options())
        items = list(hierarchy.segments(SegmentKind.LIST_ITEM))
        by_label = {item.label: item for item in items}

        self.assertEqual(hierarchy.reconstruct(), text)
        self.assertEqual(
            [item.label for item in items],
            ["(b)", "(i)", "(ii)", "(c)", "•"],
        )
        self.assertEqual(by_label["(i)"].start, text.index("(i)"))
        self.assertEqual(by_label["(ii)"].start, text.index("(ii)"))
        self.assertEqual(by_label["(b)"].end, text.index("(c)"))
        next_nested_line_start = text.index("    (ii)")
        self.assertEqual(by_label["(i)"].end, next_nested_line_start)
        self.assertLess(by_label["(i)"].end, by_label["(ii)"].start)
        self.assertEqual(by_label["(ii)"].end, text.index("(c)"))
        self.assertGreater(by_label["(i)"].level, by_label["(b)"].level)
        self.assertGreater(by_label["(ii)"].level, by_label["(b)"].level)

    def test_boundary_opt_out_uses_raw_endpoints_in_both_budget_modes(self):
        source = "xxHEADzz"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (
                Segment(
                    SegmentKind.SECTION,
                    0,
                    len(source),
                    attributes=(("heading_end", "6"),),
                ),
            ),
        )
        characters = chunk_document(
            hierarchy,
            max_chars=4,
            respect_boundaries=False,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )
        tokens = chunk_document(
            hierarchy,
            max_tokens=4,
            token_counter=len,
            token_counter_id="tests.len.v1",
            token_counter_policy=TokenCounterPolicy.MONOTONIC,
            respect_boundaries=False,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )
        expected = [(0, 4), (4, 8)]
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in characters],
            expected,
        )
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in tokens],
            expected,
        )

    def test_arbitrary_counter_finds_a_later_feasible_endpoint(self):
        def counter(text):
            return 1 if text in {"a", "abc"} else 2

        chunks = chunk_document(
            "abc",
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.later-feasible.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=50,
            token_search_max_input_bytes=100,
            token_search_max_steps=100,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual(
            [(chunk.start, chunk.end, chunk.unit_count) for chunk in chunks],
            [(0, 3, 1)],
        )

    def test_arbitrary_counter_backtracks_around_a_greedy_dead_end(self):
        def counter(text):
            return 1 if text in {"ab", "abc", "cd"} else 2

        chunks = chunk_document(
            "abcd",
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.global-path.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=100,
            token_search_max_input_bytes=500,
            token_search_max_steps=500,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks],
            [(0, 2), (2, 4)],
        )

    def test_arbitrary_counter_checks_every_feasible_overlap_context(self):
        def counter(text):
            return 1 if text in {"ab", "b", "bcd"} else 2

        chunks = chunk_document(
            "abcd",
            max_tokens=1,
            overlap_tokens=1,
            token_counter=counter,
            token_counter_id="tests.overlap-context.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=200,
            token_search_max_input_bytes=1_000,
            token_search_max_steps=1_000,
            respect_boundaries=False,
            **self.options(),
        )
        self.assertEqual(
            [
                (chunk.start, chunk.new_content_start, chunk.end)
                for chunk in chunks
            ],
            [(0, 0, 2), (1, 2, 4)],
        )

    def test_arbitrary_search_keeps_registered_heading_interior_endpoints(self):
        source = "abc"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (
                Segment(
                    SegmentKind.SECTION,
                    0,
                    len(source),
                    children=(
                        Segment(SegmentKind.TEXT, 0, 1),
                        Segment(SegmentKind.TEXT, 1, len(source)),
                    ),
                    attributes=(("heading_end", "3"),),
                ),
            ),
        )

        def counter(text):
            return 1 if text in {"a", "bc"} else 2

        chunks = chunk_document(
            hierarchy,
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.registered-heading-interior.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=100,
            token_search_max_input_bytes=500,
            token_search_max_steps=500,
        )
        self.assertEqual(
            [(chunk.start, chunk.end) for chunk in chunks],
            [(0, 1), (1, 3)],
        )

    def test_arbitrary_counter_prioritises_coverage_before_overlap(self):
        source = "abcde"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (
                Segment(SegmentKind.TEXT, 0, 2),
                Segment(SegmentKind.TEXT, 2, len(source)),
            ),
        )

        def counter(text):
            return 1 if text in {"ab", "abc", "cde", "de", "e"} else 2

        chunks = chunk_document(
            hierarchy,
            max_tokens=1,
            overlap_tokens=1,
            token_counter=counter,
            token_counter_id="tests.coverage-before-overlap.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=500,
            token_search_max_input_bytes=5_000,
            token_search_max_steps=5_000,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )
        self.assertEqual(
            [
                (chunk.start, chunk.new_content_start, chunk.end)
                for chunk in chunks
            ],
            [(0, 0, 2), (2, 2, 5)],
        )

    def test_arbitrary_counter_prefers_heading_safe_endpoints(self):
        source = "xxHEADzz"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (
                Segment(SegmentKind.TEXT, 0, 2),
                Segment(
                    SegmentKind.SECTION,
                    2,
                    len(source),
                    attributes=(("heading_end", "6"),),
                ),
            ),
        )

        def counter(text):
            return 1 if len(text) == 1 or text == "xxHEA" else 2

        chunks = chunk_document(
            hierarchy,
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.heading-order.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=500,
            token_search_max_input_bytes=5_000,
            token_search_max_steps=5_000,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )
        self.assertEqual(chunks[0].end, 1)
        self.assertFalse(2 < chunks[0].end < 6)

    def test_arbitrary_search_envelopes_fail_before_partial_output(self):
        cases = (
            ("calls", "ab", 1, 100, 100, 2, 1),
            ("input_bytes", "é", 10, 1, 100, 2, 1),
            ("steps", "ab", 10, 100, 1, 2, 1),
        )
        for (
            envelope,
            source,
            max_calls,
            max_input_bytes,
            max_steps,
            observed,
            maximum,
        ) in cases:
            calls = 0

            def counter(_text):
                nonlocal calls
                calls += 1
                return 2

            stream = iter_chunks(
                source,
                max_tokens=1,
                token_counter=counter,
                token_counter_id=f"tests.limit-{envelope}.v1",
                token_counter_policy=TokenCounterPolicy.ARBITRARY,
                token_search_max_calls=max_calls,
                token_search_max_input_bytes=max_input_bytes,
                token_search_max_steps=max_steps,
                respect_boundaries=False,
                **self.options(),
            )
            with self.subTest(envelope=envelope):
                with self.assertRaises(TokenSearchLimitExceeded) as caught:
                    next(stream)
                self.assertEqual(caught.exception.envelope, envelope)
                self.assertEqual(caught.exception.observed, observed)
                self.assertEqual(caught.exception.maximum, maximum)
                self.assertLessEqual(calls, max_calls)

    def test_large_source_tiny_byte_envelope_calls_no_counter(self):
        source = "a" * 1_000_000 + "é"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (Segment(SegmentKind.TEXT, 0, len(source)),),
        )
        calls = 0

        def counter(_text):
            nonlocal calls
            calls += 1
            return 1

        stream = iter_chunks(
            hierarchy,
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.large-byte-preflight.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=10,
            token_search_max_input_bytes=1,
            token_search_max_steps=10,
            respect_boundaries=False,
        )
        with self.assertRaises(TokenSearchLimitExceeded) as caught:
            next(stream)
        self.assertEqual(caught.exception.envelope, "input_bytes")
        self.assertEqual(calls, 0)

    def test_counter_policy_and_envelopes_change_authenticated_identity(self):
        monotonic = chunk_document(
            "abcd",
            max_tokens=4,
            token_counter=len,
            token_counter_id="tests.len.v1",
            token_counter_policy=TokenCounterPolicy.MONOTONIC,
            respect_boundaries=False,
            **self.options(),
        )[0]
        arbitrary = chunk_document(
            "abcd",
            max_tokens=4,
            token_counter=len,
            token_counter_id="tests.len.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=100,
            token_search_max_input_bytes=1_000,
            token_search_max_steps=1_000,
            respect_boundaries=False,
            **self.options(),
        )[0]
        self.assertEqual(monotonic.text, arbitrary.text)
        self.assertNotEqual(monotonic.manifest.manifest_id, arbitrary.manifest.manifest_id)
        self.assertNotEqual(monotonic.chunk_id, arbitrary.chunk_id)
        self.assertEqual(monotonic.manifest.schema_version, 2)
        self.assertEqual(
            arbitrary.manifest.token_counter_policy,
            TokenCounterPolicy.ARBITRARY,
        )
        self.assertEqual(arbitrary.manifest.token_search_max_calls, 100)
        self.assertEqual(arbitrary.manifest.token_search_max_input_bytes, 1_000)
        self.assertEqual(arbitrary.manifest.token_search_max_steps, 1_000)

    def test_nonmonotonic_token_counter_keeps_known_feasible_hard_endpoint(self):
        source = "aHxyz"
        hierarchy = DocumentHierarchy.from_segments(
            source,
            (
                Segment(SegmentKind.TEXT, 0, 1),
                Segment(
                    SegmentKind.SECTION,
                    1,
                    5,
                    attributes=(("heading_end", "5"),),
                ),
            ),
        )

        def counter(text):
            return 1 if text in {"aHx", "yz"} else 2

        chunks = chunk_document(
            hierarchy,
            max_tokens=1,
            token_counter=counter,
            token_counter_id="tests.nonmonotonic.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=500,
            token_search_max_input_bytes=5_000,
            token_search_max_steps=5_000,
            respect_boundaries=False,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )

        self.assertEqual(
            [(chunk.start, chunk.end, chunk.unit_count) for chunk in chunks],
            [(0, 3, 1), (3, 5, 1)],
        )
        self.assertEqual(reconstruct_chunks(chunks), source)


if __name__ == "__main__":
    unittest.main()
