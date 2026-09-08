# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import random
import string

import pytest

from lexnlp.nlp.en.segments.hierarchy import (
    SegmentKind,
    StructuralSpan,
    StructureProfile,
    segment_document,
)
from lexnlp.nlp.en.tests.segmentation_quality import (
    COMPLETE_KINDS,
    STRUCTURAL_KINDS,
    actual_spans,
    all_segments,
    assert_lossless_hierarchy,
    count_exact_spans,
    deterministic_sentence_spans,
    kind_value,
    load_fixture,
    normalise_label,
    per_kind_exact_counts,
    resolved_gold,
)

BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"


def make_document(text: str, *, profile: str = "conservative", **kwargs):
    return segment_document(
        text,
        sentence_segmenter=deterministic_sentence_spans,
        sentence_backend_id=BACKEND_ID,
        structure_profile=StructureProfile(profile),
        **kwargs,
    )


@pytest.mark.parametrize(
    "case",
    load_fixture("legal_edge_cases.json")["cases"],
    ids=lambda case: case["id"],
)
def test_edge_corpus_is_lossless_deterministic_and_profile_aware(case):
    first = make_document(case["text"], profile=case["structure_profile"])
    second = make_document(case["text"], profile=case["structure_profile"])
    assert_lossless_hierarchy(first)
    assert_lossless_hierarchy(second)

    first_identity = [
        (kind_value(s.kind), s.start, s.end, s.label, getattr(s, "segment_id", None)) for s in all_segments(first)
    ]
    second_identity = [
        (kind_value(s.kind), s.start, s.end, s.label, getattr(s, "segment_id", None)) for s in all_segments(second)
    ]
    assert first_identity == second_identity

    kinds = {kind_value(segment.kind) for segment in all_segments(first)}
    assert set(case["required_kinds"]) <= kinds
    assert not (set(case["forbidden_kinds"]) & kinds)

    expected = resolved_gold(case["text"], case["gold_structure"])
    actual = actual_spans(first, STRUCTURAL_KINDS)
    assert count_exact_spans(expected, actual) == (len(expected), 0, 0)


@pytest.mark.parametrize(
    "case",
    load_fixture("boundary_gold.json")["cases"],
    ids=lambda case: case["id"],
)
def test_complete_exact_boundary_gold_penalises_false_positives(case):
    document = make_document(case["text"])
    expected = resolved_gold(case["text"], case["gold_spans"])
    actual = actual_spans(document, COMPLETE_KINDS)
    counts = per_kind_exact_counts(expected, actual)
    assert all(value[1:] == (0, 0) for value in counts.values())
    assert sum(value[0] for value in counts.values()) == len(expected)


def test_complete_boundary_metric_rejects_an_oversegmenting_backend():
    case = load_fixture("boundary_gold.json")["cases"][0]

    def one_character_sentences(text):
        return [(index, index + 1) for index, value in enumerate(text) if not value.isspace()]

    document = segment_document(
        case["text"],
        sentence_segmenter=one_character_sentences,
        sentence_backend_id="deliberately-oversegmenting-v1",
    )
    expected = resolved_gold(case["text"], case["gold_spans"])
    actual = actual_spans(document, COMPLETE_KINDS)
    sentence_counts = per_kind_exact_counts(expected, actual)["sentence"]
    assert sentence_counts[1] > 0
    assert sentence_counts[2] > 0


def _find_path(node, wanted_kind, wanted_label, path=()):
    current = path + (node,)
    if kind_value(node.kind) == wanted_kind and normalise_label(node.label) == normalise_label(wanted_label):
        return current
    for child in node.children:
        result = _find_path(child, wanted_kind, wanted_label, current)
        if result is not None:
            return result
    return None


def test_nested_part_article_section_clause_ancestry_is_not_flattened():
    case = next(
        case for case in load_fixture("legal_edge_cases.json")["cases"] if case["id"] == "nested_heading_scopes"
    )
    document = make_document(case["text"])
    for expectation in case["expected_ancestry"]:
        path = _find_path(document.root, expectation["kind"], expectation["label"])
        assert path is not None
        ancestors = {(kind_value(node.kind), normalise_label(node.label)) for node in path[:-1]}
        for kind, label in expectation["ancestors"]:
            assert (kind, normalise_label(label)) in ancestors


def test_schedule_remains_parent_of_part_b():
    case = next(
        case for case in load_fixture("legal_edge_cases.json")["cases"] if case["id"] == "schedule_contains_part"
    )
    document = make_document(case["text"])
    path = _find_path(document.root, "section", "PART B — SERVICE LEVELS")
    assert path is not None
    assert ("section", normalise_label("SCHEDULE 1")) in {
        (kind_value(node.kind), normalise_label(node.label)) for node in path[:-1]
    }


def test_default_profile_does_not_promote_ordinary_numbered_obligations():
    text = "Items required:\n1 Deliver widgets\nat the site\n2 Pay fees\nwithin 30 days"
    conservative = make_document(text)
    statute = make_document(text, profile="statute")
    assert not all_segments(conservative, "section")
    assert [segment.label for segment in all_segments(statute, "section")] == [
        "1 Deliver widgets",
        "2 Pay fees",
    ]


def test_statute_profile_promotes_adjacent_no_blank_numbered_headings():
    text = "1 General duty\nbody\n2 Exceptions\nbody"
    document = make_document(text, profile="statute")
    assert [segment.label for segment in all_segments(document, "section")] == [
        "1 General duty",
        "2 Exceptions",
    ]
    assert [(segment.start, segment.end) for segment in all_segments(document, "section")] == [
        (0, text.index("2 Exceptions")),
        (text.index("2 Exceptions"), len(text)),
    ]


def test_injected_layout_table_span_is_exact_and_lossless():
    text = "SCHEDULE\n\nA | B\n1 | 2\n\nEnd."
    start = text.index("A | B")
    end = text.index("\n\nEnd")
    document = make_document(
        text,
        structural_spans=(
            StructuralSpan(
                kind=SegmentKind.TABLE,
                start=start,
                end=end,
                label="fee table",
                attributes=(("source", "layout-adapter"),),
            ),
        ),
    )
    assert_lossless_hierarchy(document)
    table = all_segments(document, "table")
    assert len(table) == 1
    assert (table[0].start, table[0].end, table[0].text(text)) == (
        start,
        end,
        text[start:end],
    )


def test_seeded_unicode_newline_metamorphic_corpus_is_lossless():
    randomizer = random.Random(20260804)
    alphabet = string.ascii_letters + string.digits + " §£€éΩ中📄.,;:()[]"
    separators = ["\n", "\r\n", "\r", "\n\n"]
    for _ in range(200):
        lines = []
        for _ in range(randomizer.randint(1, 12)):
            lines.append("".join(randomizer.choice(alphabet) for _ in range(randomizer.randint(0, 60))))
        text = "".join(
            line + (randomizer.choice(separators) if index < len(lines) - 1 else "") for index, line in enumerate(lines)
        )
        document = make_document(text)
        assert_lossless_hierarchy(document)
        assert make_document(text).reconstruct() == text
