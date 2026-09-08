"""Coverage tests for lexnlp/nlp/en/segments/chunks.py validation branches.

Covers the constructor/manifest/provenance guards, token-search helpers and
``iter_chunks``/``reconstruct_chunks`` error paths. Happy-path chunking runs
for real; only tiny hand-built hierarchies are used where needed.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from lexnlp.nlp.en.segments.chunks import (
    ChunkingManifest,
    ChunkProvenance,
    ContainerPolicy,
    DocumentChunk,
    SegmentReference,
    TokenCounterPolicy,
    TokenSearchLimitExceeded,
    _attributes,
    _digest_value,
    _HeadingIndex,
    _HierarchyIndex,
    _preserved_units,
    _raw_token_end,
    _token_context_start,
    _token_end,
    chunk_document,
    compute_chunk_metadata_sha256,
    compute_provenance_sha256,
    count_tokens,
    iter_chunks,
    reconstruct_chunks,
)
from lexnlp.nlp.en.segments.hierarchy import (
    DocumentHierarchy,
    HierarchyManifest,
    Segment,
    SegmentKind,
    StructuralSpan,
    segment_document,
)


def whole_paragraph(text: str):
    yield 0, len(text)


def whole_sentence(text: str):
    yield 0, len(text)


def options() -> dict:
    return {
        "paragraph_segmenter": whole_paragraph,
        "sentence_segmenter": whole_sentence,
        "paragraph_backend_id": "tests.whole-paragraph.v1",
        "sentence_backend_id": "tests.whole-sentence.v1",
    }


def _hierarchy(text: str = "abcdefghij") -> DocumentHierarchy:
    return segment_document(text, **options())


def _manifest(hierarchy: DocumentHierarchy, **kwargs) -> ChunkingManifest:
    args: dict = {
        "source_sha256": hashlib.sha256(b"abcdefghij").hexdigest(),
        "document_id": None,
        "unit_kind": "characters",
        "budget": 5,
        "overlap": 0,
        "respect_boundaries": True,
        "container_policy": ContainerPolicy.PRESERVE,
        "hierarchy_manifest": hierarchy.manifest,
    }
    args.update(kwargs)
    return ChunkingManifest(**args)


def _text_ref(start: int = 0, end: int = 1, **kwargs) -> SegmentReference:
    return SegmentReference(f"text:{start}:{end}", SegmentKind.TEXT, start, end, **kwargs)


class TestEnumGuard:
    def test_invalid_container_policy_choice_lists_options(self) -> None:
        with pytest.raises(ValueError, match="preserve"):
            _manifest(_hierarchy(), container_policy="bogus")

    def test_invalid_iter_chunks_policy(self) -> None:
        with pytest.raises(ValueError, match="container_policy"):
            list(iter_chunks("hi", container_policy="bogus", **options()))


class TestDigestValue:
    def test_bool_framing_differs_from_int(self) -> None:
        def framed(value) -> str:
            digest = hashlib.sha256()
            _digest_value(digest, value)
            return digest.hexdigest()

        assert framed(True) != framed(1)
        assert framed(False) != framed(0)
        assert framed(True) != framed(False)
        assert framed(None) != framed("None")
        assert framed("a") == framed("a")

    def test_unsupported_type_raises(self) -> None:
        digest = hashlib.sha256()
        with pytest.raises(TypeError, match="unsupported digest value type"):
            _digest_value(digest, 3.14)


class TestAttributesGuard:
    def test_none_gives_empty(self) -> None:
        assert _attributes(None) == ()

    def test_non_iterable_raises(self) -> None:
        with pytest.raises(TypeError, match="iterable of string pairs"):
            _attributes(42)

    def test_non_tuple_item_raises(self) -> None:
        with pytest.raises(TypeError, match="\\(name, value\\) tuple"):
            _attributes(["page"])  # type: ignore[arg-type]

    def test_wrong_arity_raises(self) -> None:
        with pytest.raises(TypeError, match="\\(name, value\\) tuple"):
            _attributes([("a", "b", "c")])  # type: ignore[arg-type]

    def test_non_string_name_or_value_raises(self) -> None:
        with pytest.raises(TypeError, match="must be strings"):
            _attributes([(1, "x")])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be strings"):
            _attributes([("x", 2)])  # type: ignore[arg-type]

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate attribute name"):
            _attributes([("page", "1"), ("page", "2")])

    def test_reference_rejects_bad_attributes(self) -> None:
        with pytest.raises(TypeError, match="iterable of string pairs"):
            _text_ref(attributes=42)  # type: ignore[arg-type]


class TestChunkingManifestGuards:
    def test_bad_source_sha256(self) -> None:
        with pytest.raises(ValueError, match="source_sha256"):
            _manifest(_hierarchy(), source_sha256="xyz")

    def test_empty_document_id(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            _manifest(_hierarchy(), document_id="")

    def test_bad_unit_kind(self) -> None:
        with pytest.raises(ValueError, match="unit_kind"):
            _manifest(_hierarchy(), unit_kind="words")

    def test_overlap_at_least_budget(self) -> None:
        with pytest.raises(ValueError, match="overlap must be smaller than budget"):
            _manifest(_hierarchy(), budget=5, overlap=5)

    def test_respect_boundaries_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="respect_boundaries must be a boolean"):
            _manifest(_hierarchy(), respect_boundaries="yes")  # type: ignore[arg-type]

    def test_hierarchy_manifest_wrong_type(self) -> None:
        with pytest.raises(TypeError, match="must be a HierarchyManifest"):
            _manifest(_hierarchy(), hierarchy_manifest="nope")  # type: ignore[arg-type]

    def test_hierarchy_manifest_without_tree(self) -> None:
        bare = HierarchyManifest(
            structural_detector_id="tests.detector.v1",
            structural_mode="replace",
            paragraph_backend_id="tests.para.v1",
            sentence_backend_id="tests.sent.v1",
        )
        assert bare.tree_sha256 is None
        with pytest.raises(ValueError, match="realised tree"):
            _manifest(_hierarchy(), hierarchy_manifest=bare)

    def test_token_mode_requires_counter_id(self) -> None:
        with pytest.raises(ValueError, match="token_counter_id"):
            _manifest(_hierarchy(), unit_kind="tokens")

    def test_token_mode_requires_policy(self) -> None:
        with pytest.raises(ValueError, match="token_counter_policy"):
            _manifest(
                _hierarchy(),
                unit_kind="tokens",
                token_counter_id="tests.len.v1",
            )

    def test_monotonic_rejects_search_envelopes(self) -> None:
        with pytest.raises(ValueError, match="only valid for arbitrary counters"):
            _manifest(
                _hierarchy(),
                unit_kind="tokens",
                token_counter_id="tests.len.v1",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
                token_search_max_calls=10,
            )

    def test_char_mode_rejects_token_options(self) -> None:
        with pytest.raises(ValueError, match="only valid in token mode"):
            _manifest(_hierarchy(), token_counter_id="tests.len.v1")

    def test_token_counter_id_must_be_non_blank(self) -> None:
        with pytest.raises(ValueError, match="token_counter_id"):
            _manifest(
                _hierarchy(),
                unit_kind="tokens",
                token_counter_id="   ",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
            )

    def test_source_id_with_document_id(self) -> None:
        manifest = _manifest(_hierarchy(), document_id="doc-9")
        assert manifest.source_id.startswith("doc-9@sha256:")

    def test_arbitrary_manifest_accepts_envelopes(self) -> None:
        manifest = _manifest(
            _hierarchy(),
            unit_kind="tokens",
            token_counter_id="tests.len.v1",
            token_counter_policy=TokenCounterPolicy.ARBITRARY,
            token_search_max_calls=10,
            token_search_max_input_bytes=100,
            token_search_max_steps=100,
        )
        assert manifest.token_counter_policy is TokenCounterPolicy.ARBITRARY


class TestSegmentReferenceGuards:
    def test_zero_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive length"):
            SegmentReference("text:1:1", SegmentKind.TEXT, 1, 1)

    def test_segment_id_mismatch(self) -> None:
        with pytest.raises(ValueError, match="segment_id must equal"):
            SegmentReference("bogus", SegmentKind.TEXT, 0, 1)

    def test_non_string_label(self) -> None:
        with pytest.raises(TypeError, match="label must be None or a string"):
            _text_ref(label=42)  # type: ignore[arg-type]

    def test_from_segment_rejects_non_segment(self) -> None:
        with pytest.raises(TypeError, match="must be a Segment"):
            SegmentReference.from_segment("nope")  # type: ignore[arg-type]

    def test_from_segment_round_trip(self) -> None:
        hierarchy = _hierarchy()
        node = next(item for item in hierarchy.root.walk() if item.kind is not SegmentKind.DOCUMENT)
        reference = SegmentReference.from_segment(node)
        assert reference.segment_id == node.segment_id
        assert (reference.start, reference.end) == (node.start, node.end)


class TestChunkProvenanceGuards:
    def test_non_iterable_segments(self) -> None:
        with pytest.raises(TypeError, match="iterable of SegmentReference"):
            ChunkProvenance(segments=42)  # type: ignore[arg-type]

    def test_non_reference_items(self) -> None:
        with pytest.raises(TypeError, match="only SegmentReference objects"):
            ChunkProvenance(segments=("x",))  # type: ignore[arg-type]

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            ChunkProvenance(segments=(_text_ref(), _text_ref()))

    def test_derived_labels_must_match(self) -> None:
        reference = SegmentReference("section:0:5", SegmentKind.SECTION, 0, 5, label="A")
        with pytest.raises(ValueError, match="section_labels must be derived"):
            ChunkProvenance(segments=(reference,), section_labels=("WRONG",))

    def test_derived_labels_accepted(self) -> None:
        reference = SegmentReference("section:0:5", SegmentKind.SECTION, 0, 5, label="A")
        provenance = ChunkProvenance(segments=(reference,), section_labels=("A",))
        assert provenance.section_labels == ("A",)
        assert provenance.segment_ids == ("section:0:5",)


class TestProvenanceDigestGuards:
    def test_non_provenance_partition_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be ChunkProvenance objects"):
            compute_provenance_sha256("x", ChunkProvenance(), ChunkProvenance())  # type: ignore[arg-type]

    def test_metadata_requires_manifest(self) -> None:
        with pytest.raises(TypeError, match="must be a ChunkingManifest"):
            compute_chunk_metadata_sha256(
                manifest="x",  # type: ignore[arg-type]
                index=0,
                start=0,
                end=1,
                new_content_start=0,
                unit_count=1,
                text_sha256="a" * 64,
                provenance_sha256="b" * 64,
            )


def _first_chunk(text: str = "abcdefghij", **kwargs) -> DocumentChunk:
    args: dict = {"max_chars": 5, "respect_boundaries": False}
    args.update(kwargs)
    return chunk_document(text, **options(), **args)[0]


class TestDocumentChunkGuards:
    def test_zero_length_chunk(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(ValueError, match="positive length"):
            replace(chunk, start=chunk.end, end=chunk.end, text="")

    def test_new_content_start_outside(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(ValueError, match="inside the chunk"):
            replace(chunk, new_content_start=chunk.end)

    def test_non_string_text(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(TypeError, match="text must be a string"):
            replace(chunk, text=123)  # type: ignore[arg-type]

    def test_text_length_mismatch(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(ValueError, match="source span length"):
            replace(chunk, text="xyz")

    def test_text_digest_mismatch(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(ValueError, match="text_sha256"):
            replace(chunk, text="XXXXX")

    def test_provenance_wrong_type(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(TypeError, match="provenance must be ChunkProvenance"):
            replace(chunk, provenance="x")  # type: ignore[arg-type]

    def test_manifest_wrong_type(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(TypeError, match="must be a ChunkingManifest"):
            replace(chunk, manifest="x")  # type: ignore[arg-type]

    def test_char_unit_count_must_equal_text_length(self) -> None:
        chunk = _first_chunk()
        with pytest.raises(ValueError, match="must equal len\\(text\\)"):
            replace(chunk, unit_count=3)


class TestCountTokens:
    def test_non_string_rejected(self) -> None:
        with pytest.raises(TypeError, match="text must be a string"):
            count_tokens(123)  # type: ignore[arg-type]

    def test_counts_words_and_punctuation(self) -> None:
        assert count_tokens("hi there") == 2
        assert count_tokens("") == 0
        assert count_tokens("a,b") == 3


class TestHierarchyIndexInternals:
    def test_bogus_heading_end_is_ignored(self) -> None:
        text = "Section Title\nbody text here"
        span = StructuralSpan(
            SegmentKind.SECTION,
            0,
            len(text),
            attributes=(("heading_end", "bogus"),),
        )
        hierarchy = segment_document(
            text,
            structural_spans=(span,),
            structural_backend_id="tests.headings.v1",
            **options(),
        )
        index = _HierarchyIndex(hierarchy)
        assert index.boundaries[0] == 0
        assert index.boundaries[-1] == len(text)

    def test_empty_range_references(self) -> None:
        index = _HierarchyIndex(_hierarchy())
        assert index.references(5, 5) == ()
        assert index.references(7, 3) == ()

    def test_references_skip_outside_nodes(self) -> None:
        index = _HierarchyIndex(_hierarchy())
        full = index.references(0, 10)
        assert full
        narrow = index.references(0, 2)
        assert narrow
        assert all(ref.start < 2 for ref in narrow)
        assert len(narrow) <= len(full)

    def test_references_outside_source_are_empty(self) -> None:
        index = _HierarchyIndex(_hierarchy())
        assert index.references(100, 200) == ()
        assert index.references(10, 12) == ()

    def test_references_skip_sibling_sections(self) -> None:
        text = "aaaaabbbbb"
        hierarchy = segment_document(
            text,
            structural_spans=(
                StructuralSpan(SegmentKind.SECTION, 0, 5, label="first"),
                StructuralSpan(SegmentKind.SECTION, 5, 10, label="second"),
            ),
            structural_backend_id="tests.siblings.v1",
            **options(),
        )
        index = _HierarchyIndex(hierarchy)
        first = index.references(0, 5)
        labels = [ref.label for ref in first if ref.kind is SegmentKind.SECTION]
        assert labels == ["first"]
        second = index.references(5, 10)
        assert [ref.label for ref in second if ref.kind is SegmentKind.SECTION] == ["second"]


class TestPreservedUnits:
    def test_partition_around_protected_child(self) -> None:
        root = Segment(
            SegmentKind.DOCUMENT,
            0,
            10,
            children=(Segment(SegmentKind.SECTION, 2, 8),),
        )
        assert _preserved_units(root) == ((0, 2), (2, 8), (8, 10))

    def test_unprotected_tree_is_single_unit(self) -> None:
        root = Segment(
            SegmentKind.DOCUMENT,
            0,
            10,
            children=(Segment(SegmentKind.TEXT, 0, 10),),
        )
        assert _preserved_units(root) == ((0, 10),)


class TestTokenHelpers:
    def test_cache_evicts_when_full(self) -> None:
        from lexnlp.nlp.en.segments.chunks import _TokenCountCache

        cache = _TokenCountCache("abcdef", len, max_entries=1)
        assert cache.count(0, 2) == 2
        assert cache.count(0, 2) == 2
        assert cache.count(2, 4) == 2
        assert len(cache.values) == 1
        assert cache.count(2, 4) == 2

    def test_context_start_reaches_unit_start(self) -> None:
        assert _token_context_start("abcdefghij", len, 0, 5, 10) == 0

    def test_context_start_zero_overlap(self) -> None:
        assert _token_context_start("abcdefghij", len, 0, 5, 0) == 5

    def test_raw_token_end_past_limit_is_none(self) -> None:
        from lexnlp.nlp.en.segments.chunks import _TokenCountCache

        cache = _TokenCountCache("abcde", len)
        assert _raw_token_end(cache, 0, 5, 5, 10) is None

    def test_token_end_falls_back_to_raw_limit(self) -> None:
        # No registered boundary sits strictly inside (0, 1], so the planner
        # falls back to the feasible raw endpoint instead of failing.
        result = _token_end(
            "x" * 50,
            len,
            (0, 50),
            _HeadingIndex(()),
            context_start=0,
            fresh_start=0,
            limit=50,
            budget=1,
            respect_boundaries=True,
        )
        assert result == (1, 1)

    def test_limit_envelope_error_carries_counts(self) -> None:
        error = TokenSearchLimitExceeded("calls", 11, 10)
        assert error.envelope == "calls"
        assert error.observed == 11
        assert error.maximum == 10
        assert "11 > 10" in str(error)


class TestIterChunksGuards:
    def test_respect_boundaries_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="respect_boundaries must be a boolean"):
            list(iter_chunks("hi", respect_boundaries="yes", **options()))  # type: ignore[arg-type]

    def test_char_options_rejected_in_token_mode(self) -> None:
        with pytest.raises(ValueError, match="character budget options"):
            chunk_document(
                "abcdef",
                max_tokens=2,
                token_counter=len,
                token_counter_id="tests.len.v1",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
                overlap_chars=1,
                **options(),
            )

    def test_monotonic_rejects_search_envelopes(self) -> None:
        with pytest.raises(ValueError, match="only valid for arbitrary counters"):
            chunk_document(
                "abcdef",
                max_tokens=2,
                token_counter=len,
                token_counter_id="tests.len.v1",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
                token_search_max_calls=5,
                **options(),
            )

    def test_hierarchy_with_segmenters_rejected(self) -> None:
        hierarchy = _hierarchy("hello")
        with pytest.raises(ValueError, match="cannot be supplied with DocumentHierarchy"):
            list(iter_chunks(hierarchy, **options()))

    def test_non_string_document_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a string or DocumentHierarchy"):
            list(iter_chunks(123, **options()))  # type: ignore[arg-type]

    def test_monotonic_counter_that_cannot_fit(self) -> None:
        with pytest.raises(ValueError, match="cannot fit advancing source content"):
            chunk_document(
                "ab",
                max_tokens=2,
                token_counter=lambda text: 100,
                token_counter_id="tests.huge.v1",
                token_counter_policy=TokenCounterPolicy.MONOTONIC,
                respect_boundaries=False,
                **options(),
            )

    def test_arbitrary_counter_that_cannot_fit(self) -> None:
        with pytest.raises(ValueError, match="cannot fit advancing source content"):
            chunk_document(
                "ab",
                max_tokens=2,
                token_counter=lambda text: 100,
                token_counter_id="tests.huge.v1",
                token_counter_policy=TokenCounterPolicy.ARBITRARY,
                respect_boundaries=False,
                **options(),
            )

    def test_hierarchy_document_chunks(self) -> None:
        hierarchy = _hierarchy("hello world")
        chunks = chunk_document(hierarchy, max_chars=4, respect_boundaries=False)
        assert reconstruct_chunks(chunks) == "hello world"


def _reindex(chunk: DocumentChunk, index: int) -> DocumentChunk:
    metadata = compute_chunk_metadata_sha256(
        manifest=chunk.manifest,
        index=index,
        start=chunk.start,
        end=chunk.end,
        new_content_start=chunk.new_content_start,
        unit_count=chunk.unit_count,
        text_sha256=chunk.text_sha256,
        provenance_sha256=chunk.provenance_sha256,
    )
    return replace(chunk, index=index, chunk_metadata_sha256=metadata)


class TestReconstructGuards:
    def test_empty_reconstructs_to_empty(self) -> None:
        assert reconstruct_chunks([]) == ""

    def test_non_chunk_items_rejected(self) -> None:
        with pytest.raises(TypeError, match="only DocumentChunk objects"):
            reconstruct_chunks(["x"])  # type: ignore[arg-type]

    def test_manifest_mismatch_rejected(self) -> None:
        first = chunk_document("aaa", max_chars=10, respect_boundaries=False, **options())[0]
        second = chunk_document("bbb", max_chars=10, respect_boundaries=False, **options())[0]
        aligned = _reindex(second, 1)
        with pytest.raises(ValueError, match="one manifest"):
            reconstruct_chunks([first, aligned])

    def test_non_contiguous_fresh_coverage_rejected(self) -> None:
        chunks = chunk_document("abcdefghij", max_chars=5, overlap_chars=2, respect_boundaries=False, **options())
        gapped = [chunks[0], _reindex(chunks[2], 1)]
        with pytest.raises(ValueError, match="contiguous fresh source coverage"):
            reconstruct_chunks(gapped)
