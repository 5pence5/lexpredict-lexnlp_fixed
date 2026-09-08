# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import pytest

from lexnlp.nlp.en.segments.chunks import chunk_document
from lexnlp.nlp.en.segments.paragraphs import get_paragraph_list
from lexnlp.nlp.en.segments.sentences import (
    SENTENCE_SEGMENTER_MODEL,
    get_sentence_list,
)
from lexnlp.nlp.en.tests.segmentation_quality import (
    deterministic_sentence_spans,
    load_fixture,
)

BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"
CASES = load_fixture("legacy_parity.json")["cases"]


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "sentence_texts" in case],
    ids=lambda case: case["id"],
)
def test_frozen_legacy_sentence_outputs(case):
    assert get_sentence_list(case["text"]) == case["sentence_texts"]


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "paragraph_texts" in case],
    ids=lambda case: case["id"],
)
def test_frozen_legacy_paragraph_outputs(case):
    assert get_paragraph_list(case["text"]) == case["paragraph_texts"]


def test_additive_hierarchy_and_chunk_calls_do_not_mutate_legacy_segmenter():
    text = "The U.S. Borrower is Acme. It shall pay £2.25 annually. No. 4 survives."
    before = get_sentence_list(text)
    model_identity = id(SENTENCE_SEGMENTER_MODEL)
    result = chunk_document(
        text,
        max_chars=40,
        overlap_chars=5,
        sentence_segmenter=deterministic_sentence_spans,
        sentence_backend_id=BACKEND_ID,
        document_id="legacy-nonmutation",
    )
    assert result
    assert id(SENTENCE_SEGMENTER_MODEL) == model_identity
    assert get_sentence_list(text) == before
