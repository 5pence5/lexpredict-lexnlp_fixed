"""Final-coverage tests for lexnlp.nlp.en.segments.chunks.

The last uncovered lines in this module are defensive guards whose
entry conditions cannot be constructed through the public API (each
carries a `# pragma: no cover` with its exact reason). The tests below
exercise the live code paths immediately around those guards so the
surrounding behavior stays pinned by real assertions.
"""

from __future__ import annotations

from itertools import pairwise

from lexnlp.nlp.en.segments.chunks import (
    _HeadingIndex,
    _HierarchyIndex,
    _raw_token_end,
    _token_end,
    _TokenCountCache,
    chunk_document,
    reconstruct_chunks,
)
from lexnlp.nlp.en.segments.hierarchy import segment_document


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


class TestRawTokenEndSearch:
    def test_doubling_reaches_limit(self) -> None:
        cache = _TokenCountCache("x" * 50, len)
        assert _raw_token_end(cache, 0, 0, 50, 100) == 50

    def test_doubling_returns_limit_on_exact_hit(self) -> None:
        cache = _TokenCountCache("x" * 50, len)
        assert _raw_token_end(cache, 0, 0, 32, 32) == 32

    def test_binary_search_finds_longest_feasible_prefix(self) -> None:
        cache = _TokenCountCache("x" * 50, len)
        assert _raw_token_end(cache, 0, 0, 50, 7) == 7

    def test_binary_search_narrows_after_overshoot(self) -> None:
        cache = _TokenCountCache("x" * 50, len)
        assert _raw_token_end(cache, 0, 0, 50, 20) == 20

    def test_search_respects_nonzero_context_start(self) -> None:
        cache = _TokenCountCache("x" * 50, len)
        assert _raw_token_end(cache, 2, 5, 50, 7) == 9

    def test_infeasible_first_slice_is_none(self) -> None:
        cache = _TokenCountCache("abcde", len)
        assert _raw_token_end(cache, 0, 0, 5, 0) is None

    def test_capped_counter_still_advances(self) -> None:
        cache = _TokenCountCache("x" * 50, lambda text: min(len(text), 10))
        assert _raw_token_end(cache, 0, 0, 50, 10) == 50


class TestTokenEndSelection:
    def test_snaps_back_to_largest_feasible_boundary(self) -> None:
        boundaries = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
        result = _token_end(
            "x" * 50,
            len,
            boundaries,
            _HeadingIndex(()),
            context_start=0,
            fresh_start=0,
            limit=50,
            budget=12,
            respect_boundaries=True,
        )
        assert result == (10, 10)

    def test_ignoring_boundaries_returns_raw_end(self) -> None:
        result = _token_end(
            "x" * 50,
            len,
            (0, 5, 10, 50),
            _HeadingIndex(()),
            context_start=0,
            fresh_start=0,
            limit=50,
            budget=7,
            respect_boundaries=False,
        )
        assert result == (7, 7)

    def test_infeasible_head_returns_none(self) -> None:
        result = _token_end(
            "x" * 50,
            len,
            (0, 50),
            _HeadingIndex(()),
            context_start=0,
            fresh_start=0,
            limit=50,
            budget=0,
            respect_boundaries=True,
        )
        assert result is None


class TestCharacterOverlapClamp:
    def test_overlap_windows_slide_by_budget_minus_overlap(self) -> None:
        chunks = chunk_document("abcdefghij", max_chars=4, overlap_chars=2, **options())
        assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 4), (2, 6), (4, 8), (6, 10)]
        assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh", "ghij"]
        assert [chunk.unit_count for chunk in chunks] == [4, 4, 4, 4]

    def test_unit_budget_chunks_every_character(self) -> None:
        chunks = chunk_document("abcd", max_chars=1, **options())
        assert [chunk.text for chunk in chunks] == ["a", "b", "c", "d"]
        assert reconstruct_chunks(chunks) == "abcd"

    def test_fresh_coverage_is_contiguous(self) -> None:
        text = "abcdefghij" * 3
        chunks = chunk_document(text, max_chars=7, overlap_chars=3, **options())
        assert chunks[0].new_content_start == 0
        assert chunks[-1].end == len(text)
        assert all(second.new_content_start == first.end for first, second in pairwise(chunks))
        assert reconstruct_chunks(chunks) == text


class TestHeadingFences:
    def _two_sections(self) -> str:
        body = "Body sentence one. Body sentence two. Body sentence three.\n"
        return f"Section Zero Title\n{body}Section One Title\n{body}"

    def test_hard_end_caps_at_heading_start(self) -> None:
        text = self._two_sections()
        index = _HierarchyIndex(segment_document(text, **options()))
        second_start = index.headings.intervals[1][0]
        headed = chunk_document(text, max_chars=12, **options())
        spans = [(chunk.start, chunk.end) for chunk in headed]
        # The window approaching the second heading stops exactly at its start
        # instead of running 12 chars deep into the heading body.
        assert (72, second_start) in spans
        assert second_start == 78
        flat = chunk_document(
            text,
            max_chars=12,
            structural_spans=(),
            structural_mode="replace",
            structural_backend_id="tests.flat.v1",
            **options(),
        )
        flat_spans = [(chunk.start, chunk.end) for chunk in flat]
        assert (72, 84) in flat_spans
        assert reconstruct_chunks(headed) == text
        assert reconstruct_chunks(flat) == text

    def test_roomy_budget_keeps_edges_out_of_headings(self) -> None:
        text = self._two_sections()
        headings = _HierarchyIndex(segment_document(text, **options())).headings.intervals
        assert len(headings) == 2
        longest = max(end - start for start, end in headings)
        for budget in (longest + 1, longest + 12):
            chunks = chunk_document(text, max_chars=budget, **options())
            assert chunks
            for chunk in chunks:
                assert chunk.unit_count <= budget
                for start, end in headings:
                    assert not start < chunk.end < end
                    assert not start < chunk.new_content_start < end
            assert reconstruct_chunks(chunks) == text


class TestStrictProgressInvariant:
    def test_fresh_starts_advance_and_cover_source(self) -> None:
        texts = [
            "abcdefghij",
            "Hello world. This is a test. Another sentence here!",
            "x" * 100,
            "line one\nline two\nline three\nline four\n",
        ]
        for text in texts:
            for budget in (1, 3, 7):
                chunks = chunk_document(text, max_chars=budget, **options())
                assert chunks
                fresh = [chunk.new_content_start for chunk in chunks]
                assert fresh[0] == 0
                assert all(second > first for first, second in pairwise(fresh))
                assert all(chunk.unit_count <= budget for chunk in chunks)
                assert reconstruct_chunks(chunks) == text

    def test_token_chunks_advance_within_budget(self) -> None:
        text = "aa bb cc dd ee ff gg"
        chunks = chunk_document(
            text,
            max_tokens=5,
            token_counter=len,
            token_counter_id="tests.len.v1",
            token_counter_policy="monotonic",
            respect_boundaries=False,
            **options(),
        )
        assert [chunk.text for chunk in chunks] == ["aa bb", " cc d", "d ee ", "ff gg"]
        assert [chunk.unit_count for chunk in chunks] == [5, 5, 5, 5]
        assert reconstruct_chunks(chunks) == text

    def test_hierarchy_entry_point_agrees_with_text(self) -> None:
        hierarchy = segment_document("abcdefghij", **options())
        from_text = chunk_document("abcdefghij", max_chars=4, **options())
        from_hierarchy = chunk_document(hierarchy, max_chars=4)
        assert [chunk.text for chunk in from_hierarchy] == [chunk.text for chunk in from_text]
