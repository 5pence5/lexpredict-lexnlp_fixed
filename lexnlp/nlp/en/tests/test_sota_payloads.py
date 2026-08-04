"""Contracts for exact, tokenizer-budgeted embedding payloads."""

from unittest import TestCase

from lexnlp.nlp.en.segments.chunks import ContainerPolicy, chunk_document
from lexnlp.nlp.en.segments.hierarchy import SegmentKind, StructuralSpan, segment_document
from lexnlp.nlp.en.segments.payloads import (
    EMBEDDING_PAYLOAD_SERIALIZATION_VERSION,
    ContextFragment,
    PayloadBudgetExceeded,
    render_embedding_payload,
)


def _whole_span(text: str):
    if text:
        yield 0, len(text), text


class TestEmbeddingPayloads(TestCase):
    def test_no_context_is_exact_source_and_identity_covers_budget(self):
        text = "Exact\r\nsource £ and 雪."
        chunk = chunk_document(
            text,
            max_chars=100,
            sentence_segmenter=_whole_span,
        )[0]
        first = render_embedding_payload(
            chunk,
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=len(text),
            include_ancestry_labels=False,
        )
        wider = render_embedding_payload(
            chunk,
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=len(text) + 1,
            include_ancestry_labels=False,
        )
        self.assertEqual(EMBEDDING_PAYLOAD_SERIALIZATION_VERSION, first.serialization_version)
        self.assertEqual(text, first.source_text)
        self.assertEqual(text, first.text)
        self.assertEqual((), first.context_fragments)
        self.assertEqual(64, len(first.payload_metadata_sha256))
        self.assertNotEqual(first.payload_id, wider.payload_id)

    def test_only_true_carried_ancestry_is_automatic(self):
        text = "SECTION 1\n" + "A" * 30
        hierarchy = segment_document(
            text,
            sentence_segmenter=_whole_span,
            structural_spans=(
                StructuralSpan(SegmentKind.SECTION, 0, len(text), label="GENERAL"),
                StructuralSpan(SegmentKind.CLAUSE, 10, len(text), label="GENERAL"),
            ),
        )
        chunks = chunk_document(hierarchy, max_chars=15, respect_boundaries=False)
        first = render_embedding_payload(
            chunks[0],
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=40,
        )
        continuation = render_embedding_payload(
            chunks[-1],
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=40,
        )
        self.assertEqual((), first.context_fragments)
        self.assertEqual(("GENERAL",), tuple(f.text for f in continuation.context_fragments))
        self.assertEqual(1, continuation.text.count("GENERAL"))
        self.assertTrue(continuation.text.endswith(chunks[-1].text))

    def test_adjacent_sections_and_overlap_do_not_claim_false_ancestry(self):
        text = "A" * 10 + "B" * 10
        hierarchy = segment_document(
            text,
            sentence_segmenter=_whole_span,
            structural_spans=(
                StructuralSpan(SegmentKind.SECTION, 0, 10, label="FIRST"),
                StructuralSpan(SegmentKind.SECTION, 10, 20, label="SECOND"),
            ),
        )
        second = chunk_document(
            hierarchy,
            max_chars=12,
            overlap_chars=2,
            container_policy=ContainerPolicy.PACK_SIBLINGS,
        )[1]
        payload = render_embedding_payload(
            second,
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=30,
        )
        self.assertEqual((), payload.context_fragments)
        self.assertNotIn("FIRST", payload.text)
        overlap = next(
            ref for ref in second.overlap_provenance.segments if ref.label == "FIRST"
        )
        with self.assertRaisesRegex(ValueError, "overlap-only"):
            render_embedding_payload(
                second,
                token_counter=len,
                tokenizer_id="characters:v1",
                max_tokens=30,
                context_fragments=(
                    ContextFragment("heading", "FIRST", overlap.segment_id),
                ),
            )

    def test_explicit_table_context_is_traced_deduplicated_and_strict(self):
        text = "prefix Body row suffix"
        table_start = text.index("Body")
        table_end = table_start + len("Body row")
        hierarchy = segment_document(
            text,
            sentence_segmenter=_whole_span,
            structural_spans=(
                StructuralSpan(SegmentKind.SECTION, 0, len(text), label="ARTICLE I"),
                StructuralSpan(
                    SegmentKind.TABLE,
                    table_start,
                    table_end,
                    label="FEE TABLE",
                ),
            ),
        )
        chunks = chunk_document(hierarchy, max_chars=100)
        chunk = next(
            candidate
            for candidate in chunks
            if any(
                ref.kind is SegmentKind.TABLE
                for ref in candidate.content_provenance.segments
            )
        )
        table = next(
            ref for ref in chunk.content_provenance.segments
            if ref.kind is SegmentKind.TABLE
        )
        fragment = ContextFragment("table_header", "Name | Amount", table.segment_id)
        payload = render_embedding_payload(
            chunk,
            token_counter=len,
            tokenizer_id="characters:v1",
            max_tokens=100,
            context_fragments=(fragment, fragment),
        )
        self.assertEqual(("Name | Amount",), tuple(f.text for f in payload.context_fragments))
        self.assertEqual("Name | Amount\n\n" + chunk.text, payload.text)

        with self.assertRaises(PayloadBudgetExceeded):
            render_embedding_payload(
                chunk,
                token_counter=len,
                tokenizer_id="characters:v1",
                max_tokens=len(chunk.text),
                context_fragments=(fragment,),
            )

    def test_context_role_owner_contract_fails_at_boundary(self):
        with self.assertRaisesRegex(ValueError, "role"):
            ContextFragment(1, "Header", "table:0:1")

        chunk = chunk_document(
            "Body",
            max_chars=100,
            sentence_segmenter=_whole_span,
        )[0]
        paragraph = next(
            ref for ref in chunk.content_provenance.segments
            if ref.kind is SegmentKind.PARAGRAPH
        )
        with self.assertRaisesRegex(ValueError, "TABLE"):
            render_embedding_payload(
                chunk,
                token_counter=len,
                tokenizer_id="characters:v1",
                max_tokens=20,
                context_fragments=(
                    ContextFragment("table_header", "Header", paragraph.segment_id),
                ),
                include_ancestry_labels=False,
            )
