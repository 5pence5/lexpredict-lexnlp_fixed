"""Coverage tests for scripts.segmentation_quality_gate edge branches."""

from __future__ import annotations

import copy
import importlib
import json
import runpy
import sys

import pytest

import scripts.segmentation_quality_gate as gate
from scripts import segmentation_quality_gate


def _external_payload() -> dict:
    text = "1. TERM\nThe agreement lasts for three years.\n\n2. PRICE\nThe annual price is £12,500 plus VAT.\n"
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


def _bound_external_payload() -> dict:
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
    payload["documents"][0]["queries"][0]["ranked_chunk_ids"] = [item["chunk_id"] for item in ranked]
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


class TestModuleImport:
    def test_reimport_reinserts_repository_root(self, monkeypatch) -> None:
        root = str(segmentation_quality_gate.REPOSITORY_ROOT)
        assert root
        monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != root])
        assert root not in sys.path
        importlib.reload(segmentation_quality_gate)
        assert sys.path[0] == root


class TestProfileAndPolicyParsing:
    def test_unknown_structure_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown structure profile"):
            gate._profile("not-a-profile")

    def test_unknown_container_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown container policy"):
            gate._policy("not-a-policy")


class TestValidateExternalShape:
    def test_bad_schema_version(self) -> None:
        payload = _external_payload()
        payload["schema_version"] = 2
        with pytest.raises(ValueError, match="schema_version must be 1"):
            gate._validate_external_shape(payload, require_rankings=False)

    @pytest.mark.parametrize("documents", [[], "not-a-list", None])
    def test_documents_must_be_non_empty_list(self, documents) -> None:
        payload = _external_payload()
        payload["documents"] = documents
        with pytest.raises(ValueError, match="non-empty list"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_document_must_be_an_object(self) -> None:
        payload = _external_payload()
        payload["documents"] = [42]
        with pytest.raises(ValueError, match="must be an object"):
            gate._validate_external_shape(payload, require_rankings=False)

    @pytest.mark.parametrize("doc_id", ["", None, 42])
    def test_document_requires_non_empty_string_id(self, doc_id) -> None:
        payload = _external_payload()
        payload["documents"][0]["id"] = doc_id
        with pytest.raises(ValueError, match="non-empty string id"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_duplicate_document_id_rejected(self) -> None:
        payload = _external_payload()
        payload["documents"].append(copy.deepcopy(payload["documents"][0]))
        with pytest.raises(ValueError, match="duplicate external document id"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_query_must_be_an_object(self) -> None:
        payload = _external_payload()
        payload["documents"][0]["queries"] = ["not-an-object"]
        with pytest.raises(ValueError, match="every query must be an object"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_query_requires_non_empty_string_id(self) -> None:
        payload = _external_payload()
        payload["documents"][0]["queries"][0]["id"] = ""
        with pytest.raises(ValueError, match="query id must be a non-empty string"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_duplicate_query_id_rejected(self) -> None:
        payload = _external_payload()
        payload["documents"][0]["queries"].append(copy.deepcopy(payload["documents"][0]["queries"][0]))
        with pytest.raises(ValueError, match="duplicate query id"):
            gate._validate_external_shape(payload, require_rankings=False)

    @pytest.mark.parametrize("gold_spans", [[], "not-a-list", None])
    def test_gold_spans_must_be_non_empty(self, gold_spans) -> None:
        payload = _external_payload()
        payload["documents"][0]["queries"][0]["gold_spans"] = gold_spans
        with pytest.raises(ValueError, match="gold_spans must be non-empty"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_ranking_required_when_requested(self) -> None:
        payload = _external_payload()
        with pytest.raises(ValueError, match="ranked_chunk_ids must be non-empty"):
            gate._validate_external_shape(payload, require_rankings=True)

    def test_ranking_ids_must_be_strings(self) -> None:
        payload = _bound_external_payload()
        payload["documents"][0]["queries"][0]["ranked_chunk_ids"] = ["ok-id", 42]
        with pytest.raises(ValueError, match="must contain strings"):
            gate._validate_external_shape(payload, require_rankings=True)

    def test_candidate_chunking_must_be_an_object(self) -> None:
        payload = _external_payload()
        payload["candidate_chunking"] = ["not", "an", "object"]
        with pytest.raises(ValueError, match="candidate_chunking must be an object"):
            gate._validate_external_shape(payload, require_rankings=False)

    def test_candidate_chunking_fixed_fields_are_pinned(self) -> None:
        payload = _external_payload()
        payload["candidate_chunking"]["structure_profile"] = "statute"
        with pytest.raises(ValueError, match=r"candidate_chunking\.structure_profile"):
            gate._validate_external_shape(payload, require_rankings=False)

    @pytest.mark.parametrize("max_chars", [0, -5, "48", True, 48.5])
    def test_max_chars_must_be_positive_integer(self, max_chars) -> None:
        payload = _external_payload()
        payload["candidate_chunking"]["max_chars"] = max_chars
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            gate._validate_external_shape(payload, require_rankings=False)

    @pytest.mark.parametrize("overlap_chars", [-1, 48, 99, "8", True, 8.0])
    def test_overlap_must_be_within_budget(self, overlap_chars) -> None:
        payload = _external_payload()
        payload["candidate_chunking"]["overlap_chars"] = overlap_chars
        with pytest.raises(ValueError, match=r"overlap_chars must be in"):
            gate._validate_external_shape(payload, require_rankings=False)


class TestRankingProvenance:
    def test_missing_provenance_rejected(self) -> None:
        prepared = gate.prepare_external_manifest(_external_payload())
        with pytest.raises(ValueError, match="ranking_provenance must be an object"):
            gate._validate_ranking_provenance({}, prepared)

    def test_empty_provenance_field_rejected(self) -> None:
        payload = _bound_external_payload()
        prepared = gate.prepare_external_manifest(_external_payload())
        payload["ranking_provenance"]["retriever_id"] = ""
        with pytest.raises(ValueError, match=r"ranking_provenance\.retriever_id"):
            gate._validate_ranking_provenance(payload, prepared)

    def test_bad_provenance_schema_version_rejected(self) -> None:
        payload = _bound_external_payload()
        prepared = gate.prepare_external_manifest(_external_payload())
        payload["ranking_provenance"]["schema_version"] = 2
        with pytest.raises(ValueError, match=r"ranking_provenance\.schema_version must be 1"):
            gate._validate_ranking_provenance(payload, prepared)


class TestEvaluateExternalManifest:
    def test_duplicate_ranked_chunk_id_rejected(self) -> None:
        payload = _bound_external_payload()
        query = payload["documents"][0]["queries"][0]
        assert len(query["ranked_chunk_ids"]) >= 1
        query["ranked_chunk_ids"] = [query["ranked_chunk_ids"][0]] * 2
        with pytest.raises(ValueError, match="duplicate ranked chunk id"):
            gate.evaluate_external_manifest(payload)


class TestRunQualityGateWithExternal:
    def test_external_section_is_evaluated(self) -> None:
        report = gate.run_quality_gate(external_manifest=_bound_external_payload())
        assert report["passed"], report["failures"]
        assert report["external"]["query_count"] == 1
        assert report["external"]["scope"] == "external-generated-character-chunk-ranking"


class TestMain:
    def test_prepare_external_without_manifest_exits(self) -> None:
        with pytest.raises(SystemExit, match="requires --external-manifest"):
            gate.main(["--prepare-external"])

    def test_default_main_writes_report_to_stdout(self, capsys) -> None:
        assert gate.main([]) == 0
        rendered = capsys.readouterr().out
        report = json.loads(rendered)
        assert report["passed"] is True
        assert report["schema_version"] == 1

    def test_main_writes_report_to_output_file(self, tmp_path) -> None:
        output = tmp_path / "nested" / "report.json"
        assert gate.main(["--output", str(output)]) == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["passed"] is True

    def test_module_main_guard_delegates_to_main(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(sys, "argv", ["segmentation_quality_gate.py", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(
                str(gate.REPOSITORY_ROOT / "scripts" / "segmentation_quality_gate.py"),
                run_name="__main__",
            )
        assert exc_info.value.code == 0
