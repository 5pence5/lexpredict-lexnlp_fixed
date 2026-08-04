# QUALITY_BLOB_RECONSTRUCTION_V3
"""Hermetic quality helpers for the structure-first segmentation tests.

The bundled corpus is intentionally small and manually auditable. Its scores are
regression/plumbing evidence, never population-level state-of-the-art evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FIXTURE_ROOT = Path(__file__).resolve().parents[4] / "test_data" / "lexnlp" / "nlp" / "en" / "sota_segmentation"
STRUCTURAL_KINDS = ("section", "clause", "list_item")
COMPLETE_KINDS = ("section", "clause", "list_item", "paragraph", "sentence")
QUALITY_MARKER = "QUALITY_BLOB_RECONSTRUCTION_V3"


def load_fixture(name: str) -> dict[str, Any]:
    payload = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("marker") != QUALITY_MARKER:
        raise ValueError(f"unsupported or unauthenticated fixture: {name}")
    return payload


def canonical_json_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def kind_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def normalise_label(value: Any) -> str | None:
    return None if value is None else " ".join(str(value).split()).casefold()


def all_segments(document: Any, kind: str | None = None) -> list[Any]:
    segments = list(document.segments())
    return segments if kind is None else [s for s in segments if kind_value(s.kind) == kind]


def assert_lossless_hierarchy(document: Any) -> None:
    source = document.source
    assert document.reconstruct() == source
    assert (document.root.start, document.root.end) == (0, len(source))
    seen: set[str] = set()

    def visit(node: Any) -> None:
        segment_id = getattr(node, "segment_id", None)
        if segment_id:
            assert segment_id not in seen
            seen.add(segment_id)
        assert 0 <= node.start <= node.end <= len(source)
        assert node.text(source) == source[node.start:node.end]
        children = tuple(node.children)
        if not children:
            return
        cursor = node.start
        for child in children:
            assert child.start == cursor
            assert child.end >= child.start
            cursor = child.end
            visit(child)
        assert cursor == node.end
        assert "".join(child.text(source) for child in children) == node.text(source)

    visit(document.root)
    assert "".join(leaf.text(source) for leaf in document.root.leaves()) == source


def resolve_gold_span(text: str, item: Mapping[str, Any]) -> tuple[int, int]:
    if "anchor" in item:
        anchor = str(item["anchor"])
        start = text.find(anchor)
        if start < 0:
            raise AssertionError(f"missing gold anchor {anchor!r}")
        return start, start + len(anchor)
    start_anchor = str(item["start_anchor"])
    start = text.find(start_anchor)
    if start < 0:
        raise AssertionError(f"missing gold start anchor {start_anchor!r}")
    end_anchor = str(item["end_anchor"])
    end = len(text) if end_anchor == "$END" else text.find(end_anchor, start + 1)
    if end <= start:
        raise AssertionError(f"invalid gold span {item!r}")
    return start, end


def resolved_gold(text: str, items: Iterable[Mapping[str, Any]]) -> list[tuple[str, int, int, str | None]]:
    result = []
    for item in items:
        start, end = resolve_gold_span(text, item)
        result.append((str(item["kind"]), start, end, normalise_label(item.get("label"))))
    return result


def actual_spans(document: Any, kinds: Iterable[str]) -> list[tuple[str, int, int, str | None]]:
    accepted = frozenset(kinds)
    return [
        (kind_value(s.kind), s.start, s.end, normalise_label(s.label))
        for s in all_segments(document)
        if kind_value(s.kind) in accepted
    ]


def count_exact_spans(
    expected: Sequence[tuple[str, int, int, str | None]],
    actual: Sequence[tuple[str, int, int, str | None]],
) -> tuple[int, int, int]:
    expected_counter, actual_counter = Counter(expected), Counter(actual)
    return (
        sum((expected_counter & actual_counter).values()),
        sum((actual_counter - expected_counter).values()),
        sum((expected_counter - actual_counter).values()),
    )


def per_kind_exact_counts(
    expected: Sequence[tuple[str, int, int, str | None]],
    actual: Sequence[tuple[str, int, int, str | None]],
    kinds: Iterable[str] = COMPLETE_KINDS,
) -> dict[str, tuple[int, int, int]]:
    return {
        kind: count_exact_spans(
            [span for span in expected if span[0] == kind],
            [span for span in actual if span[0] == kind],
        )
        for kind in kinds
    }


def prf(counts: tuple[int, int, int]) -> dict[str, float | int]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


def anchor_recall(text: str, gold_items: Sequence[Mapping[str, Any]], document: Any) -> tuple[int, int]:
    found = 0
    segments = all_segments(document)
    for item in gold_items:
        start, end = resolve_gold_span(text, item)
        label = normalise_label(item.get("label"))
        found += any(
            kind_value(segment.kind) == str(item["kind"])
            and segment.start <= start and segment.end >= end
            and (label is None or normalise_label(segment.label) == label)
            for segment in segments
        )
    return found, len(gold_items)


_SENTENCE_END = re.compile(r"[.!?](?:[\"'”’)\]]+)?(?=\s+|$)")
_PROTECTED_PERIOD = re.compile(r"(?i)\b(?:u\.s|u\.k|no|s|p|v|vs|dr|mr|mrs|ms|ltd|inc|corp|p\.a)\.$")


def deterministic_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return local half-open spans without a model, corpus or network."""

    result: list[tuple[int, int]] = []
    separators = list(re.finditer(r"(?:\r\n|\r|\n){2,}", text))
    blocks = []
    block_start = 0
    for match in separators:
        blocks.append((block_start, match.start()))
        block_start = match.end()
    blocks.append((block_start, len(text)))
    for begin, finish in blocks:
        local = text[begin:finish]
        cursor = 0
        for match in _SENTENCE_END.finditer(local):
            candidate = local[cursor:match.end()]
            if _PROTECTED_PERIOD.search(candidate.rstrip()):
                continue
            start = begin + cursor
            while start < begin + match.end() and text[start].isspace():
                start += 1
            end = begin + match.end()
            if start < end:
                result.append((start, end))
            cursor = match.end()
        start = begin + cursor
        while start < finish and text[start].isspace():
            start += 1
        if start < finish:
            result.append((start, finish))
    return result


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted((s, e) for s, e in spans if e > s):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def span_length(spans: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge_spans(spans))


def intersection_length(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    a, b = merge_spans(left), merge_spans(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def retrieval_metrics(
    ranked_spans: Sequence[tuple[int, int]],
    gold_spans: Sequence[tuple[int, int]],
    *,
    k: int = 3,
) -> dict[str, float]:
    """Union-safe character metrics with a defensible full-answer nDCG ideal."""

    if k <= 0:
        raise ValueError("k must be positive")
    gold = merge_spans(gold_spans)
    gold_chars = span_length(gold)
    if not gold_chars:
        raise ValueError("gold_spans must contain characters")
    top = list(ranked_spans[:k])
    covered: list[tuple[int, int]] = []
    gains: list[float] = []
    reciprocal_rank = 0.0
    for rank, candidate in enumerate(top, 1):
        relevant = intersection_length([candidate], gold)
        if relevant and not reciprocal_rank:
            reciprocal_rank = 1.0 / rank
        before = intersection_length(covered, gold)
        covered = merge_spans(covered + [candidate])
        gains.append(float(intersection_length(covered, gold) - before))
    retrieved_chars = span_length(covered)
    relevant_chars = intersection_length(covered, gold)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    return {
        "character_recall_at_k": relevant_chars / gold_chars,
        "context_precision_at_k": relevant_chars / retrieved_chars if retrieved_chars else 0.0,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": dcg / gold_chars,
        "hit_at_k": float(relevant_chars > 0),
    }


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def lexical_rank(query: str, chunks: Sequence[Any]) -> list[Any]:
    query_terms = Counter(token.casefold() for token in _TOKEN_RE.findall(query))
    scored = []
    for chunk in chunks:
        terms = Counter(token.casefold() for token in _TOKEN_RE.findall(chunk.text))
        overlap = sum(min(count, terms[token]) for token, count in query_terms.items())
        scored.append((overlap / math.sqrt(max(1, sum(terms.values()))), -chunk.index, chunk))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [chunk for _, _, chunk in scored]
