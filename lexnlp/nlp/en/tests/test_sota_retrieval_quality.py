# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import math

import pytest

from lexnlp.nlp.en.segments.chunks import chunk_document
from lexnlp.nlp.en.tests.segmentation_quality import (
    deterministic_sentence_spans,
    lexical_rank,
    load_fixture,
    retrieval_metrics,
)

BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"


def _gold_spans(text, anchors):
    return [(text.index(anchor), text.index(anchor) + len(anchor)) for anchor in anchors]


def test_character_metrics_use_unions_for_overlapping_ranked_spans():
    metrics = retrieval_metrics([(0, 10), (5, 15), (0, 10)], [(8, 12)], k=3)
    assert metrics["character_recall_at_k"] == 1.0
    assert metrics["context_precision_at_k"] == pytest.approx(4 / 15)
    assert metrics["reciprocal_rank"] == 1.0
    assert 0.0 < metrics["ndcg_at_k"] <= 1.0


def test_ndcg_does_not_normalise_ten_percent_answer_coverage_to_one():
    metrics = retrieval_metrics([(0, 10)], [(0, 100)], k=1)
    assert metrics["character_recall_at_k"] == 0.1
    assert metrics["context_precision_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == pytest.approx(0.1)


def test_late_first_hit_has_reciprocal_rank_penalty():
    metrics = retrieval_metrics([(100, 110), (20, 30), (0, 10)], [(0, 10)], k=3)
    assert metrics["reciprocal_rank"] == pytest.approx(1 / 3)
    assert metrics["hit_at_k"] == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_cutoff_is_rejected(k):
    with pytest.raises(ValueError):
        retrieval_metrics([(0, 1)], [(0, 1)], k=k)


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        retrieval_metrics([(0, 1)], [], k=1)


def test_synthetic_lexical_retrieval_lane_uses_only_generated_chunks():
    fixture = load_fixture("retrieval_gold.json")
    observed = []
    for document in fixture["documents"]:
        chunks = chunk_document(
            document["text"],
            max_chars=120,
            overlap_chars=20,
            sentence_segmenter=deterministic_sentence_spans,
            sentence_backend_id=BACKEND_ID,
            document_id=document["id"],
        )
        assert all(chunk.text == document["text"][chunk.start : chunk.end] for chunk in chunks)
        candidate_ids = {chunk.chunk_id for chunk in chunks}
        for query in document["queries"]:
            ranking = lexical_rank(query["query"], chunks)
            assert {chunk.chunk_id for chunk in ranking} <= candidate_ids
            metrics = retrieval_metrics(
                [(chunk.start, chunk.end) for chunk in ranking],
                _gold_spans(document["text"], query["answer_anchors"]),
                k=3,
            )
            observed.append(metrics)
    assert len(observed) == 6
    assert sum(metric["character_recall_at_k"] for metric in observed) / len(observed) >= 0.8
    assert all(math.isfinite(value) for metric in observed for value in metric.values())


def test_whole_document_candidate_exposes_low_context_precision():
    text = "Relevant answer." + (" irrelevant filler" * 200)
    answer = (0, len("Relevant answer."))
    metrics = retrieval_metrics([(0, len(text))], [answer], k=1)
    assert metrics["character_recall_at_k"] == 1.0
    assert metrics["context_precision_at_k"] < 0.1
