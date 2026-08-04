# QUALITY_BLOB_RECONSTRUCTION_V3
from __future__ import annotations

import copy
import json

import pytest

from scripts import segmentation_quality_gate as gate


def _external_payload():
    text = (
        "1. TERM\nThe agreement lasts for three years.\n\n"
        "2. PRICE\nThe annual price is £12,500 plus VAT.\n"
    )
    answer = "The annual price is £12,500 plus VAT."
    start = text.index(answer)
    return {
        "schema_version": 1,
        "candidate_chunking": gate.candidate_configuration(
            max_chars=48,
            overlap_chars=8,
            container_policy="preserve",
        ),
        "documents": [
            {
                "id": "synthetic-contract",
                "text": text,
                "queries": [
                    {
                        "id": "annual-price",
                        "query": "What is the annual price?",
                        "gold_spans": [[start, start + len(answer)]],
                    }
                ],
            }
        ],
    }


def _bound_external_payload():
    payload = _external_payload()
    prepared = gate.prepare_external_manifest(payload)
    candidates = prepared["candidates"][0]["chunks"]
    gold_start, gold_end = payload["documents"][0]["queries"][0]["gold_spans"][0]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -max(0, min(item["end"], gold_end) - max(item["start"], gold_start)),
            item["index"],
        ),
    )
    payload["documents"][0]["queries"][0]["ranked_chunk_ids"] = [
        item["chunk_id"] for item in ranked
    ]
    payload["ranking_provenance"] = {
        "schema_version": 1,
        "retriever_id": "synthetic-lexical-regression",
        "index_revision": "fixture-v1",
        "embedding_id": "none",
        "tokenizer_id": "none",
        "retrieval_manifest_sha256": prepared["retrieval_manifest_sha256"],
        "candidate_configuration_sha256": prepared["candidate_configuration_sha256"],
        "candidate_set_sha256": prepared["candidate_set_sha256"],
    }
    return payload


def test_default_gate_is_deterministic_and_reports_bounded_scope():
    first = gate.run_quality_gate()
    second = gate.run_quality_gate()
    assert first["passed"], first["failures"]
    assert second["passed"], second["failures"]
    assert first["population_sota_evidence"] is False
    assert first["evidence"] == second["evidence"]
    assert first["segmentation"] == second["segmentation"]
    assert first["retrieval"] == second["retrieval"]
    assert first["configuration"]["max_chars"] == 180
    assert first["configuration"]["overlap_chars"] == 24
    assert first["segmentation"]["anchors_total"] == 34
    assert first["segmentation"]["anchors_found"] == 34
    aggregate = first["retrieval"]["aggregate"]
    assert aggregate["mrr"] == pytest.approx(0.8333333333333334)
    assert aggregate["character_recall_at_3"] == 1.0
    assert aggregate["context_precision_at_1"] == pytest.approx(0.48098916194431673)
    assert first["packing"]["mean_chunk_characters"] == pytest.approx(90.54545454545455)
    assert first["packing"]["index_character_amplification"] == pytest.approx(
        1.014256619144603
    )
    digest_fields = [key for key in first["evidence"] if key.endswith("_sha256")]
    assert len(digest_fields) == 7
    assert all(len(first["evidence"][key]) == 64 for key in digest_fields)


def test_complete_boundary_gate_reports_exact_per_kind_precision_and_recall():
    report = gate.run_quality_gate()
    matching = report["complete_boundaries"]["matching"]
    assert matching["unit"] == "half_open_character_span"
    assert matching["tolerance_characters"] == 0
    for kind in gate.COMPLETE_KINDS:
        metrics = report["complete_boundaries"]["per_kind"][kind]
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


def test_profile_is_part_of_each_edge_case_result():
    report = gate.run_quality_gate()
    by_id = {case["id"]: case for case in report["segmentation"]["cases"]}
    assert by_id["ordinary_numbered_obligations_are_not_sections"]["structure_profile"] == "conservative"
    assert by_id["statute_profile_promotes_no_blank_numbered_headings"]["structure_profile"] == "statute"


def test_external_ranking_accepts_only_digest_bound_generated_chunk_ids():
    payload = _bound_external_payload()
    result = gate.evaluate_external_manifest(payload, k=3)
    assert result["query_count"] == 1
    assert result["aggregate"]["character_recall_at_k"] == 1.0
    assert result["ranking_provenance"]["embedding_id"] == "none"
    assert result["candidate_chunking"]["payload_kind"] == "exact_source_text"
    assert len(result["evidence"]["rankings_sha256"]) == 64


def test_external_ranking_rejects_arbitrary_non_candidate_span_disguised_as_id():
    payload = _bound_external_payload()
    query = payload["documents"][0]["queries"][0]
    query["ranked_chunk_ids"][0] = "chunk:0:synthetic-arbitrary-answer-span"
    with pytest.raises(ValueError, match="non-candidate"):
        gate.evaluate_external_manifest(payload)


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_manifest_sha256",
        "candidate_configuration_sha256",
        "candidate_set_sha256",
    ],
)
def test_external_provenance_must_bind_active_gold_config_and_candidates(field):
    payload = _bound_external_payload()
    payload["ranking_provenance"][field] = "0" * 64
    with pytest.raises(ValueError, match=field):
        gate.evaluate_external_manifest(payload)


def test_external_manifest_rejects_zero_queries_before_metric_key_access():
    payload = _external_payload()
    payload["documents"][0]["queries"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        gate.prepare_external_manifest(payload)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload["documents"][0].update({"text": ["not", "text"]}), "text must be a string"),
        (lambda payload: payload["documents"][0]["queries"][0].update({"gold_spans": [1, 2]}), "invalid gold"),
        (lambda payload: payload["documents"][0]["queries"][0].update({"gold_spans": [[True, 2]]}), "invalid gold"),
        (lambda payload: payload["documents"][0]["queries"][0].update({"query": ""}), "query must be non-empty"),
    ],
)
def test_external_manifest_validates_text_query_and_span_types(mutation, message):
    payload = _external_payload()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        gate.prepare_external_manifest(payload)


def test_trivial_whole_document_chunking_fails_context_or_size_guardrail():
    report = gate.run_quality_gate(
        max_chars=100_000,
        overlap_chars=0,
        container_policy="pack_siblings",
    )
    assert not report["passed"]
    failed = {failure["check"] for failure in report["failures"]}
    assert {"context_precision_at_1", "mean_chunk_characters"} & failed
    assert report["retrieval"]["aggregate"]["character_recall_at_3"] == 1.0


def test_overlap_amplification_is_an_explicit_gate_not_hidden_in_recall():
    report = gate.run_quality_gate(
        thresholds={"max_index_character_amplification": 1.0}
    )
    assert not report["passed"]
    assert any(
        failure["check"] == "index_character_amplification"
        for failure in report["failures"]
    )


@pytest.mark.parametrize("output_flag", ["--output", "--json-output"])
def test_prepare_external_cli_emits_candidates_without_accepting_rankings(tmp_path, output_flag):
    manifest = tmp_path / "external.json"
    output = tmp_path / "nested" / "prepared.json"
    manifest.write_text(json.dumps(_external_payload()), encoding="utf-8")
    assert gate.main(
        [
            "--external-manifest",
            str(manifest),
            "--prepare-external",
            output_flag,
            str(output),
        ]
    ) == 0
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared["candidates"][0]["chunks"]
    assert "candidate_set_sha256" in prepared
