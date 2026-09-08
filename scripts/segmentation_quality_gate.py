#!/usr/bin/env python3
# QUALITY_BLOB_RECONSTRUCTION_V3
"""Deterministic, dependency-free segmentation regression and retrieval gate.

The default fixture is a small hermetic plumbing/regression corpus. Passing this
script is not population-level SOTA evidence and does not replace evaluation on
licensed legal boundary and retrieval datasets.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from lexnlp.nlp.en.segments.chunks import ContainerPolicy, chunk_document
from lexnlp.nlp.en.segments.hierarchy import StructureProfile, segment_document
from lexnlp.nlp.en.tests.segmentation_quality import (
    COMPLETE_KINDS,
    STRUCTURAL_KINDS,
    actual_spans,
    anchor_recall,
    assert_lossless_hierarchy,
    canonical_json_sha256,
    count_exact_spans,
    deterministic_sentence_spans,
    lexical_rank,
    load_fixture,
    prf,
    resolve_gold_span,
    resolved_gold,
    retrieval_metrics,
)

REPORT_SCHEMA_VERSION = 1
EXTERNAL_SCHEMA_VERSION = 1
SENTENCE_BACKEND_ID = "lexnlp-hermetic-regression-sentences-v1"
CANONICAL_DIGEST_FORMAT = "sha256(canonical-json:utf8,sort-keys,no-whitespace)"
DEFAULT_MAX_CHARS = 180
DEFAULT_OVERLAP_CHARS = 24
DEFAULT_THRESHOLDS = {
    "min_anchor_recall": 1.0,
    "min_structural_precision": 1.0,
    "min_structural_recall": 1.0,
    "min_complete_per_kind_precision": 1.0,
    "min_complete_per_kind_recall": 1.0,
    "min_mrr": 0.75,
    "min_character_recall_at_3": 1.0,
    "min_context_precision_at_1": 0.25,
    "max_mean_chunk_characters": 180.0,
    "max_index_character_amplification": 1.50,
}


def _profile(value: str) -> StructureProfile:
    try:
        return StructureProfile(value)
    except ValueError as error:
        raise ValueError(f"unknown structure profile: {value!r}") from error


def _policy(value: str) -> ContainerPolicy:
    try:
        return ContainerPolicy(value)
    except ValueError as error:
        raise ValueError(f"unknown container policy: {value!r}") from error


def _segment(text: str, *, structure_profile: str = "conservative"):
    return segment_document(
        text,
        sentence_segmenter=deterministic_sentence_spans,
        sentence_backend_id=SENTENCE_BACKEND_ID,
        structure_profile=_profile(structure_profile),
    )


def candidate_configuration(
    *,
    max_chars: int,
    overlap_chars: int,
    container_policy: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "payload_kind": "exact_source_text",
        "unit_kind": "characters",
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "respect_boundaries": True,
        "container_policy": container_policy,
        "structure_profile": "conservative",
        "sentence_backend_id": SENTENCE_BACKEND_ID,
    }


def _generate_candidates(
    documents: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    generated: dict[str, list[Any]] = {}
    evidence: list[dict[str, Any]] = []
    for document in documents:
        document_id = str(document["id"])
        text = str(document["text"])
        chunks = chunk_document(
            text,
            max_chars=int(configuration["max_chars"]),
            overlap_chars=int(configuration["overlap_chars"]),
            respect_boundaries=True,
            container_policy=_policy(str(configuration["container_policy"])),
            structure_profile=StructureProfile.CONSERVATIVE,
            sentence_segmenter=deterministic_sentence_spans,
            sentence_backend_id=SENTENCE_BACKEND_ID,
            document_id=document_id,
        )
        generated[document_id] = list(chunks)
        evidence.append(
            {
                "document_id": document_id,
                "source_sha256": canonical_json_sha256({"text": text}),
                "chunks": [
                    {
                        "index": chunk.index,
                        "chunk_id": chunk.chunk_id,
                        "start": chunk.start,
                        "new_content_start": chunk.new_content_start,
                        "end": chunk.end,
                        "text_sha256": chunk.text_sha256,
                        "chunk_metadata_sha256": chunk.chunk_metadata_sha256,
                        "provenance_sha256": chunk.provenance_sha256,
                    }
                    for chunk in chunks
                ],
            }
        )
    return generated, evidence


def _validate_external_shape(payload: Mapping[str, Any], *, require_rankings: bool) -> None:
    if payload.get("schema_version") != EXTERNAL_SCHEMA_VERSION:
        raise ValueError("external manifest schema_version must be 1")
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("external manifest documents must be a non-empty list")
    seen_documents: set[str] = set()
    seen_queries: set[tuple[str, str]] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("every external document must be an object")
        document_id = document.get("id")
        text = document.get("text")
        queries = document.get("queries")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("every external document requires a non-empty string id")
        if document_id in seen_documents:
            raise ValueError(f"duplicate external document id: {document_id}")
        seen_documents.add(document_id)
        if not isinstance(text, str):
            raise ValueError(f"{document_id}: text must be a string")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"{document_id}: queries must be a non-empty list")
        for query in queries:
            if not isinstance(query, Mapping):
                raise ValueError(f"{document_id}: every query must be an object")
            query_id = query.get("id")
            query_text = query.get("query")
            gold_spans = query.get("gold_spans")
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{document_id}: query id must be a non-empty string")
            if (document_id, query_id) in seen_queries:
                raise ValueError(f"duplicate query id: {document_id}/{query_id}")
            seen_queries.add((document_id, query_id))
            if not isinstance(query_text, str) or not query_text:
                raise ValueError(f"{document_id}/{query_id}: query must be non-empty")
            if not isinstance(gold_spans, list) or not gold_spans:
                raise ValueError(f"{document_id}/{query_id}: gold_spans must be non-empty")
            for span in gold_spans:
                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
                    or not 0 <= span[0] < span[1] <= len(text)
                ):
                    raise ValueError(f"{document_id}/{query_id}: invalid gold character span {span!r}")
            if require_rankings:
                ranking = query.get("ranked_chunk_ids")
                if not isinstance(ranking, list) or not ranking:
                    raise ValueError(f"{document_id}/{query_id}: ranked_chunk_ids must be non-empty")
                if not all(isinstance(value, str) and value for value in ranking):
                    raise ValueError(f"{document_id}/{query_id}: ranked_chunk_ids must contain strings")

    configuration = payload.get("candidate_chunking")
    if not isinstance(configuration, Mapping):
        raise ValueError("candidate_chunking must be an object")
    required_config = {
        "schema_version": 1,
        "payload_kind": "exact_source_text",
        "unit_kind": "characters",
        "respect_boundaries": True,
        "structure_profile": "conservative",
        "sentence_backend_id": SENTENCE_BACKEND_ID,
    }
    for key, expected in required_config.items():
        if configuration.get(key) != expected:
            raise ValueError(f"candidate_chunking.{key} must equal {expected!r}")
    max_chars = configuration.get("max_chars")
    overlap_chars = configuration.get("overlap_chars")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("candidate_chunking.max_chars must be a positive integer")
    if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int) or not 0 <= overlap_chars < max_chars:
        raise ValueError("candidate_chunking.overlap_chars must be in [0,max_chars)")
    _policy(str(configuration.get("container_policy")))


def _external_retrieval_evidence(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload["schema_version"],
        "documents": [
            {
                "id": document["id"],
                "text": document["text"],
                "queries": [
                    {
                        "id": query["id"],
                        "query": query["query"],
                        "gold_spans": query["gold_spans"],
                    }
                    for query in document["queries"]
                ],
            }
            for document in payload["documents"]
        ],
    }


def prepare_external_manifest(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Generate candidate IDs/digests before an external retriever is run."""

    payload = _load_payload(payload_or_path)
    _validate_external_shape(payload, require_rankings=False)
    configuration = dict(payload["candidate_chunking"])
    _, candidate_evidence = _generate_candidates(payload["documents"], configuration)
    return {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "limitations": [
            "character-budget candidates only",
            "exact source-text payload only",
            "conservative structure profile only",
            "no embedding payload or token-mode evaluation",
        ],
        "retrieval_manifest_sha256": canonical_json_sha256(_external_retrieval_evidence(payload)),
        "candidate_configuration_sha256": canonical_json_sha256(configuration),
        "candidate_set_sha256": canonical_json_sha256(candidate_evidence),
        "candidate_chunking": configuration,
        "candidates": candidate_evidence,
    }


def _load_payload(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, Mapping):
        return dict(payload_or_path)
    return json.loads(Path(payload_or_path).read_text(encoding="utf-8"))


def _validate_ranking_provenance(payload: Mapping[str, Any], prepared: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = payload.get("ranking_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("ranking_provenance must be an object")
    for field in ("retriever_id", "index_revision", "embedding_id", "tokenizer_id"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"ranking_provenance.{field} must be a non-empty string")
    if provenance.get("schema_version") != 1:
        raise ValueError("ranking_provenance.schema_version must be 1")
    for field in (
        "retrieval_manifest_sha256",
        "candidate_configuration_sha256",
        "candidate_set_sha256",
    ):
        if provenance.get(field) != prepared[field]:
            raise ValueError(f"ranking_provenance.{field} does not bind the active evidence")
    return provenance


def evaluate_external_manifest(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    k: int = 3,
) -> dict[str, Any]:
    """Evaluate rankings only when every ID is an exact generated LexNLP chunk."""

    payload = _load_payload(payload_or_path)
    _validate_external_shape(payload, require_rankings=True)
    prepared = prepare_external_manifest(payload)
    provenance = _validate_ranking_provenance(payload, prepared)
    generated, _ = _generate_candidates(payload["documents"], payload["candidate_chunking"])
    per_query = []
    rankings_for_digest = []
    for document in payload["documents"]:
        chunks = generated[document["id"]]
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for query in document["queries"]:
            ranking_ids = list(query["ranked_chunk_ids"])
            if len(ranking_ids) != len(set(ranking_ids)):
                raise ValueError(f"{document['id']}/{query['id']}: duplicate ranked chunk id")
            unknown = [chunk_id for chunk_id in ranking_ids if chunk_id not in by_id]
            if unknown:
                raise ValueError(f"{document['id']}/{query['id']}: ranking contains non-candidate chunk IDs")
            ranked = [by_id[chunk_id] for chunk_id in ranking_ids]
            metrics = retrieval_metrics(
                [(chunk.start, chunk.end) for chunk in ranked],
                [tuple(span) for span in query["gold_spans"]],
                k=k,
            )
            per_query.append(
                {
                    "document_id": document["id"],
                    "query_id": query["id"],
                    **metrics,
                }
            )
            rankings_for_digest.append(
                {
                    "document_id": document["id"],
                    "query_id": query["id"],
                    "ranked_chunk_ids": ranking_ids,
                }
            )
    aggregate = {
        key: statistics.fmean(item[key] for item in per_query)
        for key in (
            "character_recall_at_k",
            "context_precision_at_k",
            "reciprocal_rank",
            "ndcg_at_k",
            "hit_at_k",
        )
    }
    return {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "scope": "external-generated-character-chunk-ranking",
        "limitations": prepared["limitations"],
        "query_count": len(per_query),
        "per_query": per_query,
        "aggregate": aggregate,
        "candidate_chunking": prepared["candidate_chunking"],
        "ranking_provenance": dict(provenance),
        "evidence": {
            "canonical_digest_format": CANONICAL_DIGEST_FORMAT,
            "retrieval_manifest_sha256": prepared["retrieval_manifest_sha256"],
            "candidate_configuration_sha256": prepared["candidate_configuration_sha256"],
            "candidate_set_sha256": prepared["candidate_set_sha256"],
            "rankings_sha256": canonical_json_sha256(rankings_for_digest),
        },
    }


def _evaluate_edge_cases(edge_fixture: Mapping[str, Any]) -> dict[str, Any]:
    found = total = 0
    expected_structural = []
    actual_structural = []
    per_case = []
    for case in edge_fixture["cases"]:
        document = _segment(
            case["text"],
            structure_profile=case.get("structure_profile", "conservative"),
        )
        assert_lossless_hierarchy(document)
        case_found, case_total = anchor_recall(case["text"], case["gold_segments"], document)
        found += case_found
        total += case_total
        expected = resolved_gold(case["text"], case["gold_structure"])
        actual = actual_spans(document, STRUCTURAL_KINDS)
        expected_structural.extend((case["id"],) + item for item in expected)
        actual_structural.extend((case["id"],) + item for item in actual)
        per_case.append(
            {
                "id": case["id"],
                "structure_profile": case.get("structure_profile", "conservative"),
                "exact_reconstruction": document.reconstruct() == case["text"],
                "anchors_found": case_found,
                "anchors_total": case_total,
            }
        )
    structure_counts = count_exact_spans(expected_structural, actual_structural)
    return {
        "exact_reconstruction_rate": statistics.fmean(float(item["exact_reconstruction"]) for item in per_case),
        "anchors_found": found,
        "anchors_total": total,
        "anchor_recall": found / total if total else 1.0,
        "structural": prf(structure_counts),
        "cases": per_case,
    }


def _evaluate_complete_boundaries(boundary_fixture: Mapping[str, Any]) -> dict[str, Any]:
    expected_all = []
    actual_all = []
    for case in boundary_fixture["cases"]:
        document = _segment(case["text"])
        assert_lossless_hierarchy(document)
        expected_all.extend((case["id"],) + item for item in resolved_gold(case["text"], case["gold_spans"]))
        actual_all.extend((case["id"],) + item for item in actual_spans(document, COMPLETE_KINDS))
    per_kind = {}
    for kind in COMPLETE_KINDS:
        expected = [item[1:] for item in expected_all if item[1] == kind]
        actual = [item[1:] for item in actual_all if item[1] == kind]
        per_kind[kind] = prf(count_exact_spans(expected, actual))
    return {
        "matching": boundary_fixture["matching"],
        "per_kind": per_kind,
    }


def _evaluate_retrieval(
    fixture: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    generated, candidate_evidence = _generate_candidates(fixture["documents"], configuration)
    query_results = []
    ranking_evidence = []
    source_characters = 0
    indexed_characters = 0
    chunk_lengths = []
    for document in fixture["documents"]:
        chunks = generated[document["id"]]
        source_characters += len(document["text"])
        indexed_characters += sum(len(chunk.text) for chunk in chunks)
        chunk_lengths.extend(len(chunk.text) for chunk in chunks)
        for query in document["queries"]:
            ranked = lexical_rank(query["query"], chunks)
            gold = [resolve_gold_span(document["text"], {"anchor": anchor}) for anchor in query["answer_anchors"]]
            at_one = retrieval_metrics([(chunk.start, chunk.end) for chunk in ranked], gold, k=1)
            at_three = retrieval_metrics([(chunk.start, chunk.end) for chunk in ranked], gold, k=3)
            query_results.append(
                {
                    "document_id": document["id"],
                    "query_id": query["id"],
                    "reciprocal_rank": at_three["reciprocal_rank"],
                    "character_recall_at_3": at_three["character_recall_at_k"],
                    "context_precision_at_1": at_one["context_precision_at_k"],
                    "context_precision_at_3": at_three["context_precision_at_k"],
                    "ndcg_at_3": at_three["ndcg_at_k"],
                }
            )
            ranking_evidence.append(
                {
                    "document_id": document["id"],
                    "query_id": query["id"],
                    "ranked_chunk_ids": [chunk.chunk_id for chunk in ranked],
                }
            )
    aggregate = {
        key: statistics.fmean(item[key] for item in query_results)
        for key in (
            "reciprocal_rank",
            "character_recall_at_3",
            "context_precision_at_1",
            "context_precision_at_3",
            "ndcg_at_3",
        )
    }
    aggregate["mrr"] = aggregate.pop("reciprocal_rank")
    packing = {
        "chunk_count": len(chunk_lengths),
        "mean_chunk_characters": statistics.fmean(chunk_lengths) if chunk_lengths else 0.0,
        "max_chunk_characters": max(chunk_lengths, default=0),
        "source_characters": source_characters,
        "indexed_characters": indexed_characters,
        "index_character_amplification": (indexed_characters / source_characters if source_characters else 1.0),
    }
    return (
        {
            "query_count": len(query_results),
            "aggregate": aggregate,
            "per_query": query_results,
        },
        packing,
        [candidate_evidence, ranking_evidence],
    )


def run_quality_gate(
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    container_policy: str = "preserve",
    thresholds: Mapping[str, float] | None = None,
    external_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    active_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    configuration = candidate_configuration(
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        container_policy=container_policy,
    )
    edge_fixture = load_fixture("legal_edge_cases.json")
    boundary_fixture = load_fixture("boundary_gold.json")
    retrieval_fixture = load_fixture("retrieval_gold.json")

    segmentation = _evaluate_edge_cases(edge_fixture)
    boundaries = _evaluate_complete_boundaries(boundary_fixture)
    retrieval, packing, generated_evidence = _evaluate_retrieval(retrieval_fixture, configuration)

    failures = []
    checks = [
        ("anchor_recall", segmentation["anchor_recall"], ">=", active_thresholds["min_anchor_recall"]),
        (
            "structural_precision",
            segmentation["structural"]["precision"],
            ">=",
            active_thresholds["min_structural_precision"],
        ),
        ("structural_recall", segmentation["structural"]["recall"], ">=", active_thresholds["min_structural_recall"]),
        ("mrr", retrieval["aggregate"]["mrr"], ">=", active_thresholds["min_mrr"]),
        (
            "character_recall_at_3",
            retrieval["aggregate"]["character_recall_at_3"],
            ">=",
            active_thresholds["min_character_recall_at_3"],
        ),
        (
            "context_precision_at_1",
            retrieval["aggregate"]["context_precision_at_1"],
            ">=",
            active_thresholds["min_context_precision_at_1"],
        ),
        (
            "mean_chunk_characters",
            packing["mean_chunk_characters"],
            "<=",
            active_thresholds["max_mean_chunk_characters"],
        ),
        (
            "index_character_amplification",
            packing["index_character_amplification"],
            "<=",
            active_thresholds["max_index_character_amplification"],
        ),
    ]
    for kind, metrics in boundaries["per_kind"].items():
        checks.extend(
            [
                (f"{kind}_precision", metrics["precision"], ">=", active_thresholds["min_complete_per_kind_precision"]),
                (f"{kind}_recall", metrics["recall"], ">=", active_thresholds["min_complete_per_kind_recall"]),
            ]
        )
    for name, observed, operator, threshold in checks:
        failed = observed < threshold if operator == ">=" else observed > threshold
        if failed:
            failures.append({"check": name, "observed": observed, "operator": operator, "threshold": threshold})

    candidate_evidence, ranking_evidence = generated_evidence
    evidence = {
        "canonical_digest_format": CANONICAL_DIGEST_FORMAT,
        "edge_fixture_sha256": canonical_json_sha256(edge_fixture),
        "boundary_fixture_sha256": canonical_json_sha256(boundary_fixture),
        "retrieval_fixture_sha256": canonical_json_sha256(retrieval_fixture),
        "candidate_configuration_sha256": canonical_json_sha256(configuration),
        "candidate_set_sha256": canonical_json_sha256(candidate_evidence),
        "rankings_sha256": canonical_json_sha256(ranking_evidence),
        "sentence_backend_sha256": canonical_json_sha256(
            {"id": SENTENCE_BACKEND_ID, "contract": "local ordered half-open spans"}
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "hermetic-regression-and-plumbing-only",
        "population_sota_evidence": False,
        "limitations": [
            "small synthetic legal corpus",
            "character/exact-source payload retrieval only",
            "no external boundary population",
            "no embedding or token-mode retrieval evaluation",
        ],
        "passed": not failures,
        "failures": failures,
        "configuration": configuration,
        "thresholds": active_thresholds,
        "evidence": evidence,
        "segmentation": segmentation,
        "complete_boundaries": boundaries,
        "retrieval": retrieval,
        "packing": packing,
    }
    if external_manifest is not None:
        report["external"] = evaluate_external_manifest(external_manifest)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--json-output", dest="output", type=Path)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument(
        "--container-policy",
        choices=[value.value for value in ContainerPolicy],
        default=ContainerPolicy.PRESERVE.value,
    )
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument(
        "--prepare-external",
        action="store_true",
        help="emit generated candidate IDs and binding digests without evaluating rankings",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepare_external:
        if args.external_manifest is None:
            raise SystemExit("--prepare-external requires --external-manifest")
        report = prepare_external_manifest(args.external_manifest)
        exit_code = 0
    else:
        report = run_quality_gate(
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            container_policy=args.container_policy,
            external_manifest=args.external_manifest,
        )
        exit_code = 0 if report["passed"] else 1
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
