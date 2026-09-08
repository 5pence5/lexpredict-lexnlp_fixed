"""Final-coverage regression tests pinning the shadowing behavior behind hierarchy pragmas."""

from __future__ import annotations

import pytest

from lexnlp.nlp.en.segments.hierarchy import (
    DocumentHierarchy,
    Segment,
    SegmentKind,
    StructuralSpan,
    StructureProfile,
    _split_lines,
    _table_spans,
    segment_document,
)


def test_oversized_child_shadowed_by_parent_check() -> None:
    source = "abcdef"
    root = Segment(SegmentKind.DOCUMENT, 0, 6, (Segment(SegmentKind.TEXT, 0, 99),))
    with pytest.raises(ValueError, match="exceeds its parent"):
        DocumentHierarchy(source, root)


def test_gapped_children_shadowed_by_partition_check() -> None:
    source = "abcdef"
    root = Segment(
        SegmentKind.DOCUMENT,
        0,
        6,
        (Segment(SegmentKind.TEXT, 0, 3), Segment(SegmentKind.TEXT, 4, 6)),
    )
    with pytest.raises(ValueError, match="do not exactly partition"):
        DocumentHierarchy(source, root)


@pytest.mark.parametrize(
    "text",
    ["", "a", "a\n", "a\rb\nc\r\nd", "x\x85y\rz", "\u2028\u2029", "lone\ud800surrogate\n"],
)
def test_split_lines_always_consumes_full_text(text: str) -> None:
    lines = _split_lines(text)
    consumed = "".join(text[line.start : line.end] for line in sorted(lines, key=lambda line: line.start))
    assert consumed == text
    assert sum(line.end - line.start for line in lines) == len(text)


def test_table_spans_parent_assignment_without_inner_pop() -> None:
    text = "SECTION 1 Intro\n\n| a | b |\n| c | d |\n"
    hierarchy = segment_document(text, structure_profile=StructureProfile.CONSERVATIVE)
    tables = [segment for segment in hierarchy.segments() if segment.kind == SegmentKind.TABLE]
    assert len(tables) == 1
    assert hierarchy.text(tables[0]) == "| a | b |\n| c | d |\n"


def test_table_spans_direct_crossing_containers_stay_stable() -> None:
    containers = [
        StructuralSpan(SegmentKind.SECTION, 0, 100, "outer", 1),
        StructuralSpan(SegmentKind.SECTION, 50, 60, "inner", 2),
    ]
    assert _table_spans([(90, 95), (5, 8)], containers)[0].start == 90
    nested = _table_spans([(55, 58)], containers)
    assert [(span.start, span.end, span.level) for span in nested] == [(55, 58, 3)]
    assert _table_spans([], containers) == []
