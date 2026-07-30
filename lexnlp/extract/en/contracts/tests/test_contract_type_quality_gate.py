import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.contract_type_quality_gate import (
    evaluate_duplicate_group_holdout,
    verify_fixture_sha256,
)


def holdout_evidence():
    split = {
        "strategy": "lexnlp-global-group-holdout-v1",
        "normalization": "casefold, collapse whitespace, strip, sha256 UTF-8",
        "split_sha256": "trusted-split",
        "ambiguous_cross_label_groups_excluded": 20,
        "ambiguous_cross_label_samples_excluded": 40,
        "train_samples": 1882,
        "test_samples": 465,
        "evaluated_labels": 46,
        "train_only_labels_with_fewer_than_five_groups": 5,
        "normalized_group_overlap": 0,
    }
    metrics = {
        "accuracy_top1": 0.75,
        "accuracy_top3": 0.90,
        "f1_macro": 0.65,
        "f1_weighted": 0.74,
    }
    baseline = {
        "training_recipe": "contract-type-runtime-v2",
        "duplicate_group_holdout": {
            **split,
            "metrics": deepcopy(metrics),
        },
    }
    candidate = {
        "target_tag": "pipeline/contract-type/candidate",
        "training_recipe": "contract-type-runtime-v2",
        "validation_split": deepcopy(split),
        "validation_metrics": deepcopy(metrics),
    }
    return baseline, candidate


def evaluate(baseline, candidate):
    return evaluate_duplicate_group_holdout(
        baseline_payload=baseline,
        candidate_report=candidate,
        candidate_tag="pipeline/contract-type/candidate",
    )


def test_verify_fixture_sha256_rejects_tampering(tmp_path: Path):
    fixture = tmp_path / "fixture.csv"
    fixture.write_bytes(b"trusted")
    expected = hashlib.sha256(fixture.read_bytes()).hexdigest()
    fixture.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Fixture SHA-256 mismatch"):
        verify_fixture_sha256(fixture, expected)


def test_duplicate_group_holdout_gate_accepts_unchanged_candidate():
    baseline, candidate = holdout_evidence()

    result = evaluate(baseline, candidate)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])
    assert result["baseline"]["accuracy_topn"] == 0.90
    assert result["candidate"]["accuracy_topn"] == 0.90


def test_duplicate_group_holdout_gate_rejects_nested_metric_regression():
    baseline, candidate = holdout_evidence()
    candidate["validation_metrics"]["accuracy_top1"] = 0.74

    result = evaluate(baseline, candidate)

    assert result["passed"] is False
    failed = [check for check in result["checks"] if not check["passed"]]
    assert failed == [
        {
            "scope": "holdout-metric",
            "metric": "accuracy_top1",
            "baseline": 0.75,
            "candidate": 0.74,
            "delta": pytest.approx(-0.01),
            "max_regression": 0.0,
            "passed": False,
        }
    ]


@pytest.mark.parametrize(
    "missing_evidence",
    ("validation_metrics", "split_sha256"),
)
def test_duplicate_group_holdout_gate_fails_closed_on_missing_evidence(
    missing_evidence,
):
    baseline, candidate = holdout_evidence()
    if missing_evidence == "validation_metrics":
        del candidate["validation_metrics"]
    else:
        del candidate["validation_split"][missing_evidence]

    with pytest.raises(ValueError, match="evidence"):
        evaluate(baseline, candidate)


@pytest.mark.parametrize(
    ("evidence", "candidate_value"),
    (
        ("split_sha256", "different-split"),
        ("training_recipe", "different-recipe"),
    ),
)
def test_duplicate_group_holdout_gate_rejects_split_or_recipe_mismatch(
    evidence,
    candidate_value,
):
    baseline, candidate = holdout_evidence()
    if evidence == "training_recipe":
        candidate[evidence] = candidate_value
    else:
        candidate["validation_split"][evidence] = candidate_value

    result = evaluate(baseline, candidate)

    assert result["passed"] is False
    failed = [check for check in result["checks"] if not check["passed"]]
    assert len(failed) == 1
    assert failed[0]["evidence"].endswith(evidence)
