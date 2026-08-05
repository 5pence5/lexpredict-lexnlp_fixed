# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import pytest

from lexnlp.nlp.en.segments.chunks import (
    DEFAULT_MAX_CHARS,
    ContainerPolicy,
    TokenCounterPolicy,
    chunk_document,
    count_tokens,
)
from lexnlp.nlp.en.segments.hierarchy import (
    SegmentKind,
    StructuralSpan,
    StructureProfile,
    segment_document,
)
from lexnlp.nlp.en.tests.segmentation_quality import (
    assert_lossless_hierarchy,
    deterministic_sentence_spans,
)


BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"
WHOLE_SPAN_BACKEND_ID = "lexnlp-hermetic-whole-span-v1"


def whole_span(text):
    if not text:
        return ()
    return ((0, len(text), text),)


def chunks(text, **kwargs):
    return chunk_document(
        text,
        sentence_segmenter=deterministic_sentence_spans,
        sentence_backend_id=BACKEND_ID,
        **kwargs,
    )


def assert_chunk_invariants(text, result, *, max_chars=None, max_tokens=None, counter=None):
    assert [chunk.index for chunk in result] == list(range(len(result)))
    rebuilt = []
    previous_end = 0
    for index, chunk in enumerate(result):
        assert chunk.text == text[chunk.start:chunk.end]
        assert chunk.start <= chunk.new_content_start <= chunk.end
        assert chunk.text_sha256 and len(chunk.text_sha256) == 64
        assert ":chunk:sha256:" in chunk.chunk_id
        assert chunk.new_content_start == previous_end
        rebuilt.append(text[chunk.new_content_start:chunk.end])
        previous_end = chunk.end
        if max_chars is not None:
            assert len(chunk.text) <= max_chars
            assert chunk.unit_count == len(chunk.text)
        if max_tokens is not None:
            assert counter is not None
            assert counter(chunk.text) <= max_tokens
            assert chunk.unit_count == counter(chunk.text)
    assert "".join(rebuilt) == text


@pytest.mark.parametrize("max_chars", [1, 7, 30, 61, 4000])
@pytest.mark.parametrize("overlap_chars", [0, 1])
def test_character_budget_is_strict_lossless_and_deterministic(max_chars, overlap_chars):
    if overlap_chars >= max_chars:
        pytest.skip("invalid overlap is covered separately")
    text = "§ 1 — Café\r\nThe Supplier shall pay £2.25.\n\n(a) First;\n(b) Second. 📄"
    first = chunks(text, max_chars=max_chars, overlap_chars=overlap_chars)
    second = chunks(text, max_chars=max_chars, overlap_chars=overlap_chars)
    assert_chunk_invariants(text, first, max_chars=max_chars)
    assert [
        (item.start, item.new_content_start, item.end, item.chunk_id)
        for item in first
    ] == [
        (item.start, item.new_content_start, item.end, item.chunk_id)
        for item in second
    ]


def test_default_budget_hard_splits_an_oversized_atomic_unit():
    text = "x" * 9001
    result = chunks(text)
    assert DEFAULT_MAX_CHARS == 4000
    assert [len(chunk.text) for chunk in result] == [4000, 4000, 1001]
    assert_chunk_invariants(text, result, max_chars=4000)


def test_token_mode_requires_explicit_counter_and_versioned_identity():
    with pytest.raises(ValueError):
        chunks("alpha beta", max_tokens=4)
    with pytest.raises(ValueError):
        chunks("alpha beta", max_tokens=4, token_counter=count_tokens)
    with pytest.raises(ValueError):
        chunks("alpha beta", max_chars=10, token_counter=count_tokens, token_counter_id="bad")


def test_token_budget_and_overlap_are_strict_and_reproducible():
    text = "Alpha beta, gamma. Delta epsilon zeta. Eta theta iota."
    kwargs = {
        "max_tokens": 7,
        "overlap_tokens": 2,
        "token_counter": count_tokens,
        "token_counter_id": "lexnlp-count-tokens-v1",
        "token_counter_policy": TokenCounterPolicy.MONOTONIC,
    }
    first = chunks(text, **kwargs)
    second = chunks(text, **kwargs)
    assert_chunk_invariants(text, first, max_tokens=7, counter=count_tokens)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]


def test_non_monotonic_token_counter_cannot_break_the_hard_cap():
    text = "abcdefghijklmno"

    def deliberately_non_monotonic(value):
        if len(value) in {5, 11}:
            return 50
        return len(value)

    result = chunks(
        text,
        max_tokens=6,
        token_counter=deliberately_non_monotonic,
        token_counter_id="test-non-monotonic-v1",
        token_counter_policy=TokenCounterPolicy.ARBITRARY,
        token_search_max_calls=5_000,
        token_search_max_input_bytes=100_000,
        token_search_max_steps=20_000,
    )
    assert result
    assert_chunk_invariants(
        text, result, max_tokens=6, counter=deliberately_non_monotonic
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 0},
        {"max_chars": True},
        {"max_chars": 10, "max_tokens": 5, "token_counter": count_tokens, "token_counter_id": "x"},
        {"max_chars": 10, "overlap_chars": -1},
        {"max_chars": 10, "overlap_chars": 10},
        {"max_chars": 10, "overlap_tokens": 1},
        {"max_tokens": 5, "overlap_chars": 1, "token_counter": count_tokens, "token_counter_id": "x"},
        {"max_tokens": 5, "overlap_tokens": 5, "token_counter": count_tokens, "token_counter_id": "x"},
    ],
)
def test_invalid_budget_combinations_fail_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        chunks("text", **kwargs)


def test_empty_source_has_no_chunks():
    assert chunks("", max_chars=10) == []


SECTION_TEXT = (
    "1. FIRST\nAlpha section.\n\n"
    "2. SECOND\nBeta section.\n\n"
    "3. THIRD\nGamma section."
)



def test_preserve_policy_fences_table_gaps_to_their_enclosing_section():
    text = "1. FIRST\nA | B | C\npost\n2. SECOND\nD | E | F\npost2"
    hierarchy = segment_document(
        text,
        paragraph_segmenter=whole_span,
        sentence_segmenter=whole_span,
        paragraph_backend_id=WHOLE_SPAN_BACKEND_ID,
        sentence_backend_id=WHOLE_SPAN_BACKEND_ID,
    )
    result = chunk_document(hierarchy, max_chars=100)
    section_boundary = text.index("2. SECOND")
    assert [(chunk.start, chunk.end) for chunk in result] == [
        (0, 9),
        (9, 19),
        (19, section_boundary),
        (section_boundary, 34),
        (34, 44),
        (44, len(text)),
    ]
    assert not any(
        chunk.start < section_boundary < chunk.end for chunk in result
    )
    assert_chunk_invariants(text, result, max_chars=100)


def test_preserve_policy_keeps_a_table_nested_under_a_list_item_atomic():
    text = "• Charges\nA | B | C\n1 | 2 | 3\ntail"
    table_start = text.index("A | B | C")
    table_end = text.index("tail")
    hierarchy = segment_document(
        text,
        structural_spans=(
            StructuralSpan(SegmentKind.LIST_ITEM, 0, len(text)),
            StructuralSpan(SegmentKind.TABLE, table_start, table_end),
        ),
        paragraph_segmenter=whole_span,
        sentence_segmenter=whole_span,
        paragraph_backend_id=WHOLE_SPAN_BACKEND_ID,
        sentence_backend_id=WHOLE_SPAN_BACKEND_ID,
    )
    result = chunk_document(hierarchy, max_chars=100)
    assert [(chunk.start, chunk.end) for chunk in result] == [
        (0, table_start),
        (table_start, table_end),
        (table_end, len(text)),
    ]
    assert_chunk_invariants(text, result, max_chars=100)


@pytest.mark.parametrize("max_chars", [30, 60, 100])
def test_default_preserve_policy_never_crosses_complete_top_level_sections(max_chars):
    result = chunks(SECTION_TEXT, max_chars=max_chars)
    assert [(chunk.start, chunk.end) for chunk in result] == [
        (0, 25),
        (25, 50),
        (50, 73),
    ]


def test_pack_siblings_is_explicit_and_never_bisects_a_heading():
    packed = chunks(
        SECTION_TEXT,
        max_chars=60,
        container_policy=ContainerPolicy.PACK_SIBLINGS,
    )
    assert [(chunk.start, chunk.end) for chunk in packed] == [(0, 50), (50, 73)]
    heading_starts = [0, SECTION_TEXT.index("2. SECOND"), SECTION_TEXT.index("3. THIRD")]
    assert all(
        chunk.end in heading_starts[1:] + [len(SECTION_TEXT)]
        for chunk in packed
    )



def test_boundary_opt_out_makes_container_policies_plan_identical_chunks():
    kwargs = {
        "max_chars": 17,
        "overlap_chars": 3,
        "respect_boundaries": False,
    }
    preserved = chunks(
        SECTION_TEXT,
        container_policy=ContainerPolicy.PRESERVE,
        **kwargs,
    )
    packed = chunks(
        SECTION_TEXT,
        container_policy=ContainerPolicy.PACK_SIBLINGS,
        **kwargs,
    )

    def signature(result):
        return [
            (
                chunk.start,
                chunk.new_content_start,
                chunk.end,
                chunk.text,
                chunk.unit_count,
            )
            for chunk in result
        ]

    assert signature(preserved) == signature(packed)
    assert_chunk_invariants(SECTION_TEXT, preserved, max_chars=17)
    assert_chunk_invariants(SECTION_TEXT, packed, max_chars=17)


def test_preserve_policy_drops_overlap_at_complete_protected_sections():
    result = chunks(SECTION_TEXT, max_chars=30, overlap_chars=5)
    assert [
        (chunk.start, chunk.new_content_start, chunk.end) for chunk in result
    ] == [(0, 0, 25), (25, 25, 50), (50, 50, 73)]
    assert_chunk_invariants(SECTION_TEXT, result, max_chars=30)


def test_overlap_is_dropped_when_it_would_split_the_next_atomic_section():
    text = "1. FIRST\n" + "a" * 20 + "\n2. SECOND\n" + "b" * 19
    boundary = text.index("2. SECOND")
    result = chunks(text, max_chars=30, overlap_chars=5)
    second = next(chunk for chunk in result if chunk.new_content_start == boundary)
    assert second.start == boundary


def test_chunk_identity_authenticates_configuration_text_and_provenance():
    text = "1. SCOPE\n1.1 A duty applies.\n\n2. TERM\nThis survives."
    first = chunks(text, max_chars=40, document_id="contract-A")
    repeated = chunks(text, max_chars=40, document_id="contract-A")
    changed_config = chunks(text, max_chars=41, document_id="contract-A")
    changed_document = chunks(text + "!", max_chars=40, document_id="contract-A")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in changed_config]
    assert [chunk.chunk_id for chunk in first] != [chunk.chunk_id for chunk in changed_document]
    for chunk in first:
        assert len(chunk.text_sha256) == 64
        assert len(chunk.chunk_metadata_sha256) == 64
        assert len(chunk.provenance_sha256) == 64
        assert chunk.provenance.segment_ids


def test_chunking_a_prebuilt_hierarchy_preserves_profile_manifest():
    text = "1 General duty\nbody\n2 Exceptions\nbody"
    hierarchy = segment_document(
        text,
        sentence_segmenter=deterministic_sentence_spans,
        sentence_backend_id=BACKEND_ID,
        structure_profile=StructureProfile.STATUTE,
    )
    assert_lossless_hierarchy(hierarchy)
    result = chunk_document(hierarchy, max_chars=50)
    assert result
    assert all(chunk.provenance.segment_ids for chunk in result)
