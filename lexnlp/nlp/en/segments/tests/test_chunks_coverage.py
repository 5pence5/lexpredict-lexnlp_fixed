"""Coverage tests for preserved-unit partitioning in lexnlp.nlp.en.segments.chunks."""

from __future__ import annotations

from itertools import pairwise

from lexnlp.nlp.en.segments.chunks import _preserved_units
from lexnlp.nlp.en.segments.hierarchy import Segment, SegmentKind


def _nested_wrapper_tree() -> Segment:
    clause_a = Segment(SegmentKind.CLAUSE, 20, 30)
    clause_b = Segment(SegmentKind.CLAUSE, 35, 45)
    paragraph = Segment(SegmentKind.PARAGRAPH, 10, 50, children=(clause_a, clause_b))
    return Segment(SegmentKind.DOCUMENT, 0, 100, children=(paragraph,))


class TestPreservedUnitsMerge:
    def test_adjacent_unprotected_gaps_merge_across_wrapper(self) -> None:
        units = _preserved_units(_nested_wrapper_tree())

        assert units == ((0, 20), (20, 30), (30, 35), (35, 45), (45, 100))

    def test_merged_units_cover_source_contiguously(self) -> None:
        units = _preserved_units(_nested_wrapper_tree())

        assert units[0][0] == 0
        assert units[-1][1] == 100
        assert all(first[1] == second[0] for first, second in pairwise(units))

    def test_flat_protected_child_does_not_merge(self) -> None:
        section = Segment(SegmentKind.SECTION, 10, 50)
        root = Segment(SegmentKind.DOCUMENT, 0, 100, children=(section,))

        assert _preserved_units(root) == ((0, 10), (10, 50), (50, 100))
