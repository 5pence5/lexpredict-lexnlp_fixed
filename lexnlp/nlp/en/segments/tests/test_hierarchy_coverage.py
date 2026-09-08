"""Coverage tests for lexnlp.nlp.en.segments.hierarchy validation and edge paths."""

from __future__ import annotations

import hashlib

import pytest

from lexnlp.nlp.en.segments.hierarchy import (
    MAX_HIERARCHY_DEPTH,
    DocumentHierarchy,
    HierarchyManifest,
    Segment,
    SegmentKind,
    StructuralMode,
    StructuralSpan,
    StructureProfile,
    _attributes,
    _callable_id,
    _delimited_blocks,
    _digest_value,
    _enum,
    _integer,
    _merge_augmented_span,
    _NumericCandidate,
    _ordinal,
    _outline_spans,
    _plain_segments,
    _prediction_spans,
    _section_spans,
    _separator_intervals,
    _sequence_numbered_heading_indices,
    _span_result,
    _split_lines,
    _table_spans,
    _tree_sha256,
    _validate_hierarchy,
    _validate_structural_spans,
    iter_document_segments,
    segment_document,
)


def whole_paragraph(text: str):
    if text:
        yield 0, len(text), text


def whole_sentence(text: str):
    if text:
        yield 0, len(text), text


def backends() -> dict:
    return {
        "paragraph_segmenter": whole_paragraph,
        "sentence_segmenter": whole_sentence,
        "paragraph_backend_id": "tests.whole-paragraph.v1",
        "sentence_backend_id": "tests.whole-sentence.v1",
    }


def _manifest(**kwargs) -> HierarchyManifest:
    args = dict(
        structural_detector_id="tests.tree.v1",
        structural_mode=StructuralMode.REPLACE,
        paragraph_backend_id="tests.embedded.v1",
        sentence_backend_id="tests.embedded.v1",
    )
    args.update(kwargs)
    return HierarchyManifest(**args)


class TestEnumIntegerAttributes:
    def test_enum_lists_choices_on_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of") as caught:
            _enum("not-a-kind", SegmentKind, "kind")
        assert "'document'" in str(caught.value)
        assert "'separator'" in str(caught.value)

    def test_integer_rejects_bool_and_non_int(self) -> None:
        with pytest.raises(TypeError, match="start must be an integer"):
            _integer(True, "start")
        with pytest.raises(TypeError, match="end must be an integer"):
            _integer("3", "end")

    def test_integer_enforces_minimum(self) -> None:
        with pytest.raises(ValueError, match="level must be at least 0"):
            _integer(-1, "level", minimum=0)
        assert _integer(0, "level", minimum=0) == 0

    def test_attributes_none_is_empty(self) -> None:
        assert _attributes(None) == ()

    def test_attributes_non_iterable_raises(self) -> None:
        with pytest.raises(TypeError, match="iterable of string pairs"):
            _attributes(42)

    def test_attributes_item_must_be_name_value_tuple(self) -> None:
        with pytest.raises(TypeError, match="\\(name, value\\) tuple"):
            _attributes(["page"])
        with pytest.raises(TypeError, match="\\(name, value\\) tuple"):
            _attributes([("a", "b", "c")])

    def test_attributes_names_and_values_must_be_strings(self) -> None:
        with pytest.raises(TypeError, match="must be strings"):
            _attributes([(1, "x")])
        with pytest.raises(TypeError, match="must be strings"):
            _attributes([("x", 2)])

    def test_duplicate_attribute_name_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate attribute name 'page'"):
            _attributes([("page", "1"), ("page", "2")])

    def test_segment_accepts_none_attributes(self) -> None:
        segment = Segment(SegmentKind.TEXT, 0, 4, attributes=None)
        assert segment.attributes == ()
        assert segment.length == 4


class TestManifestAndSegmentGuards:
    def test_manifest_rejects_blank_detector_id(self) -> None:
        with pytest.raises(ValueError, match="structural_detector_id must be a non-empty string"):
            _manifest(structural_detector_id="  ")

    def test_manifest_rejects_invalid_tree_digest(self) -> None:
        with pytest.raises(ValueError, match="tree_sha256 must be a lowercase SHA-256 digest"):
            _manifest(tree_sha256="not-a-digest")
        with pytest.raises(ValueError, match="tree_sha256 must be a lowercase SHA-256 digest"):
            _manifest(tree_sha256="A" * 64)

    def test_segment_end_must_not_precede_start(self) -> None:
        with pytest.raises(ValueError, match="segment end must not precede start"):
            Segment(SegmentKind.TEXT, 4, 1)

    def test_children_must_be_iterable_of_segments(self) -> None:
        with pytest.raises(TypeError, match="iterable of Segment objects"):
            Segment(SegmentKind.PARAGRAPH, 0, 1, children=1)
        with pytest.raises(TypeError, match="children must contain only Segment objects"):
            Segment(SegmentKind.PARAGRAPH, 0, 1, children=["x"])

    def test_text_requires_string_source_inside_bounds(self) -> None:
        segment = Segment(SegmentKind.TEXT, 0, 4)
        with pytest.raises(TypeError, match="source must be a string"):
            segment.text(b"abcd")
        with pytest.raises(ValueError, match="segment lies outside source"):
            segment.text("ab")
        assert segment.text("abcd") == "abcd"

    def test_structural_span_rejects_paragraph_kind(self) -> None:
        with pytest.raises(ValueError, match="section, clause, list_item, or table"):
            StructuralSpan(SegmentKind.PARAGRAPH, 0, 4)


class TestDigestAndValidateHierarchy:
    def test_bool_digest_framing_differs_from_int(self) -> None:
        def framed(value: object) -> str:
            digest = hashlib.sha256()
            _digest_value(digest, value)
            return digest.hexdigest()

        assert framed(True) != framed(1)
        assert framed(False) != framed(0)
        assert framed(True) != framed(False)

    def test_unsupported_digest_type_raises(self) -> None:
        digest = hashlib.sha256()
        with pytest.raises(TypeError, match="unsupported digest value type: float"):
            _digest_value(digest, 3.14)

    def test_root_must_be_document_covering_source(self) -> None:
        source = "ab"
        with pytest.raises(ValueError, match="root segment must have kind DOCUMENT"):
            _validate_hierarchy(source, Segment(SegmentKind.TEXT, 0, 2))
        with pytest.raises(ValueError, match="root must cover the exact source span"):
            _validate_hierarchy(source, Segment(SegmentKind.DOCUMENT, 0, 1, level=0))

    def test_non_empty_document_requires_children(self) -> None:
        with pytest.raises(ValueError, match="partitioning children"):
            _validate_hierarchy("ab", Segment(SegmentKind.DOCUMENT, 0, 2, (), level=0))

    def test_children_must_partition_parent(self) -> None:
        source = "abc"
        gap_at_start = Segment(
            SegmentKind.DOCUMENT,
            0,
            3,
            (Segment(SegmentKind.TEXT, 1, 3),),
            level=0,
        )
        with pytest.raises(ValueError, match="do not exactly partition"):
            _validate_hierarchy(source, gap_at_start)

        exceeds = Segment(
            SegmentKind.DOCUMENT,
            0,
            3,
            (Segment(SegmentKind.TEXT, 0, 4),),
            level=0,
        )
        with pytest.raises(ValueError, match="exceeds its parent"):
            _validate_hierarchy(source, exceeds)

        gap_at_end = Segment(
            SegmentKind.DOCUMENT,
            0,
            3,
            (Segment(SegmentKind.TEXT, 0, 2),),
            level=0,
        )
        with pytest.raises(ValueError, match="do not exactly partition"):
            _validate_hierarchy(source, gap_at_end)

    def test_depth_limit_is_enforced(self) -> None:
        # Identical (kind, start, end) identities collide before the depth
        # check, so each wrapping level must occupy a distinct span.
        length = MAX_HIERARCHY_DEPTH + 1
        source = "x" * length
        node = Segment(SegmentKind.TEXT, length - 1, length)
        for start in range(length - 2, -1, -1):
            head = Segment(SegmentKind.TEXT, start, start + 1)
            node = Segment(SegmentKind.PARAGRAPH, start, length, (head, node))
        root = Segment(SegmentKind.DOCUMENT, 0, length, (node,), level=0)
        with pytest.raises(ValueError, match="MAX_HIERARCHY_DEPTH"):
            _validate_hierarchy(source, root)

    def test_document_hierarchy_type_guards(self) -> None:
        child = Segment(SegmentKind.TEXT, 0, 1)
        root = Segment(SegmentKind.DOCUMENT, 0, 1, (child,), level=0)
        with pytest.raises(TypeError, match="source must be a string"):
            DocumentHierarchy(b"x", root, _manifest())
        with pytest.raises(TypeError, match="root must be a Segment"):
            DocumentHierarchy("x", "root", _manifest())
        with pytest.raises(TypeError, match="manifest must be a HierarchyManifest"):
            DocumentHierarchy("x", root, "manifest")

    def test_tree_sha256_mismatch_is_rejected(self) -> None:
        child = Segment(SegmentKind.TEXT, 0, 1)
        root = Segment(SegmentKind.DOCUMENT, 0, 1, (child,), level=0)
        digest = _tree_sha256(root)
        other = "0" * 64 if digest != "0" * 64 else "1" * 64
        with pytest.raises(ValueError, match="does not match manifest tree_sha256"):
            DocumentHierarchy("x", root, _manifest(tree_sha256=other))

    def test_from_segments_requires_string_source(self) -> None:
        with pytest.raises(TypeError, match="source must be a string"):
            DocumentHierarchy.from_segments(b"abc", (Segment(SegmentKind.TEXT, 0, 3),))


class TestSplitLinesOrdinalAndSequence:
    def test_split_lines_keeps_crlf_offsets(self) -> None:
        text = "alpha\r\nbeta"
        lines = _split_lines(text)
        assert [line.content for line in lines] == ["alpha", "beta"]
        assert lines[0].start == 0
        assert lines[0].end == 7
        assert lines[1].start == 7
        assert lines[1].end == len(text)
        assert text[lines[0].start : lines[0].content_end] == "alpha"

    def test_ordinal_number_letter_and_none(self) -> None:
        assert _ordinal("12") == ("number", 12)
        assert _ordinal("B") == ("letter", ord("b"))
        assert _ordinal("a") == ("letter", ord("a"))
        assert _ordinal("aa") is None
        assert _ordinal("iv") is None

    def test_sequence_skips_empty_parts_and_non_ordinals(self) -> None:
        candidates = (
            _NumericCandidate(0, (), "Empty"),
            _NumericCandidate(1, ("1", "aa"), "Wide"),
            _NumericCandidate(2, ("1", "a"), "Alpha"),
            _NumericCandidate(3, ("1", "b"), "Beta"),
        )
        promoted = _sequence_numbered_heading_indices(candidates)
        assert 0 not in promoted
        assert 1 not in promoted
        assert promoted == {2, 3}

    def test_statute_letter_sequence_promotes_parenthetical_headings(self) -> None:
        text = "1(a) Alpha heading\nbody of alpha\n1(b) Beta heading\nbody of beta\n"
        hierarchy = segment_document(text, structure_profile=StructureProfile.STATUTE, **backends())
        labels = [node.label for node in hierarchy.segments(SegmentKind.SECTION)]
        assert labels == ["1(a) Alpha heading", "1(b) Beta heading"]

    def test_statute_non_ordinal_parenthetical_is_not_a_heading(self) -> None:
        text = "1(aa) Wide heading\nbody of wide\n1(ab) Other heading\nbody of other\n"
        hierarchy = segment_document(text, structure_profile=StructureProfile.STATUTE, **backends())
        assert list(hierarchy.segments(SegmentKind.SECTION)) == []


class TestOutlineTablesAndStructuralSpans:
    def test_outline_tracks_active_section_while_collecting_list_items(self) -> None:
        text = "SECTION 1 First\nintro\nSECTION 2 Second\n(a) nested item\n1.1 Follow-on clause\n"
        lines = _split_lines(text)
        _blocks, table_lines = _delimited_blocks(lines)
        sections, heading_lines = _section_spans(
            text,
            lines,
            StructureProfile.CONSERVATIVE,
            table_lines,
        )
        assert [span.label for span in sections] == ["SECTION 1 First", "SECTION 2 Second"]
        outlines = _outline_spans(text, lines, sections, heading_lines, table_lines)
        labels = [span.label for span in outlines]
        assert "(a)" in labels
        item = next(span for span in outlines if span.label == "(a)")
        second = sections[1]
        assert second.start <= item.start < item.end <= second.end

    def test_section_then_list_items_are_nested(self) -> None:
        text = "SECTION 1 General\nThe following items apply.\n(a) first condition\n(b) second condition\n"
        hierarchy = segment_document(text, **backends())
        sections = list(hierarchy.segments(SegmentKind.SECTION))
        items = list(hierarchy.segments(SegmentKind.LIST_ITEM))
        assert [node.label for node in sections] == ["SECTION 1 General"]
        assert [node.label for node in items] == ["(a)", "(b)"]
        assert items[0].start > sections[0].start
        assert items[0].end <= sections[0].end
        assert hierarchy.reconstruct() == text

    def test_table_in_second_section_skips_closed_container(self) -> None:
        text = "SECTION 1 First\nintro only\nSECTION 2 Second\nItem | Price | Qty\nA | 1 | 2\nB | 3 | 4\n"
        hierarchy = segment_document(text, **backends())
        sections = list(hierarchy.segments(SegmentKind.SECTION))
        tables = list(hierarchy.segments(SegmentKind.TABLE))
        assert [node.label for node in sections] == ["SECTION 1 First", "SECTION 2 Second"]
        assert len(tables) == 1
        assert tables[0].start >= sections[1].start
        assert tables[0].end <= sections[1].end
        assert ("detector", "delimited_lines") in tables[0].attributes

    def test_table_spans_skips_container_that_ended_before_table(self) -> None:
        first = StructuralSpan(SegmentKind.SECTION, 0, 4, label="early", level=1)
        second = StructuralSpan(SegmentKind.SECTION, 4, 12, label="late", level=1)
        tables = _table_spans(((8, 12),), (first, second))
        assert len(tables) == 1
        assert tables[0].start == 8
        assert tables[0].end == 12
        assert tables[0].level == 2

    def test_table_spans_drops_container_that_does_not_cover_table_end(self) -> None:
        short = StructuralSpan(SegmentKind.SECTION, 0, 10, label="short", level=1)
        tables = _table_spans(((8, 12),), (short,))
        assert len(tables) == 1
        assert tables[0].level == 1
        assert tables[0].start == 8
        assert tables[0].end == 12

    def test_validate_structural_spans_type_and_bounds(self) -> None:
        with pytest.raises(TypeError, match="only StructuralSpan objects"):
            _validate_structural_spans((object(),), 10)
        with pytest.raises(ValueError, match="structural span lies outside source"):
            _validate_structural_spans((StructuralSpan(SegmentKind.SECTION, 0, 11),), 10)

    def test_structural_span_depth_limit(self) -> None:
        spans = tuple(StructuralSpan(SegmentKind.SECTION, 0, 200 - index) for index in range(MAX_HIERARCHY_DEPTH + 1))
        with pytest.raises(ValueError, match="structural spans exceed MAX_HIERARCHY_DEPTH"):
            _validate_structural_spans(spans, 200)

    def test_merge_augmented_span_requires_same_range_and_kind(self) -> None:
        builtin = StructuralSpan(SegmentKind.TABLE, 0, 4, label="built")
        with pytest.raises(ValueError, match="augmented spans must have the same range and kind"):
            _merge_augmented_span(builtin, StructuralSpan(SegmentKind.TABLE, 1, 4))
        with pytest.raises(ValueError, match="augmented spans must have the same range and kind"):
            _merge_augmented_span(builtin, StructuralSpan(SegmentKind.SECTION, 0, 4))


class TestSpanResultPlainSegmentsAndCallableId:
    def test_span_result_rejects_wrong_shape_and_bounds(self) -> None:
        with pytest.raises(TypeError, match="must yield 2- or 3-tuples"):
            _span_result("abcd", "nope", backend_name="paragraph_segmenter")
        with pytest.raises(ValueError, match="invalid span"):
            _span_result("abcd", (0, 0), backend_name="paragraph_segmenter")
        with pytest.raises(ValueError, match="invalid span"):
            _span_result("abcd", (0, 9), backend_name="paragraph_segmenter")
        with pytest.raises(TypeError, match="third tuple item must be a string"):
            _span_result("abcd", (0, 4, 12), backend_name="sentence_segmenter")

    def test_prediction_spans_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="ordered and non-overlapping"):
            _prediction_spans(
                "abcdef",
                ((0, 3), (2, 5)),
                backend_name="paragraph_segmenter",
            )

    def test_separator_intervals_merge_overlapping_blank_and_page_markers(self) -> None:
        text = "Hello.\n\n<PAGE>\nWorld."
        intervals = _separator_intervals(text)
        assert intervals
        blank_start = text.index("\n")
        page_end = text.index("World")
        covering = [span for span in intervals if span[0] <= blank_start and span[1] >= page_end]
        assert covering == [(blank_start, page_end)]

    def test_plain_segments_empty_range_is_empty(self) -> None:
        assert _plain_segments("abc", 2, 2, whole_paragraph, whole_sentence) == ()
        assert _plain_segments("abc", 3, 1, whole_paragraph, whole_sentence) == ()

    def test_callable_id_rejects_blank_explicit_id(self) -> None:
        with pytest.raises(ValueError, match="paragraph_backend_id must be a non-empty string"):
            _callable_id(whole_paragraph, "  ", "builtin.blank_lines.v1", name="paragraph_backend_id")
        assert (
            _callable_id(whole_paragraph, "custom.v1", "builtin.blank_lines.v1", name="paragraph_backend_id")
            == "custom.v1"
        )

    def test_segment_document_requires_string_text(self) -> None:
        with pytest.raises(TypeError, match="text must be a string"):
            segment_document(b"hello", **backends())

    def test_iter_document_segments_yields_requested_kind(self) -> None:
        text = "SECTION 1 General\nThe parties agree to the terms."
        sentences = list(
            iter_document_segments(
                text,
                kind=SegmentKind.SENTENCE,
                **backends(),
            )
        )
        sections = list(
            iter_document_segments(
                text,
                kind="section",
                **backends(),
            )
        )
        assert sentences
        assert all(node.kind is SegmentKind.SENTENCE for node in sentences)
        assert [node.label for node in sections] == ["SECTION 1 General"]
        assert "".join(node.text(text) for node in sentences)
        reconstructed = list(iter_document_segments(text, **backends()))
        document = reconstructed[0]
        assert document.kind is SegmentKind.DOCUMENT
        assert document.reconstruct(text) == text
