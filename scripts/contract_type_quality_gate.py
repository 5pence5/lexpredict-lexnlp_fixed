#!/usr/bin/env python3
"""Quality gate for contract-type model upgrades."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_FIXTURE = Path(
    "test_data/lexnlp/extract/en/contracts/tests/test_contracts/test_contract_type.csv"
)
REQUIRED_METRIC_KEYS = ("accuracy_top1", "accuracy_topn", "f1_macro", "f1_weighted")
HOLDOUT_SPLIT_IDENTITY_KEYS = (
    "strategy",
    "normalization",
    "split_sha256",
    "ambiguous_cross_label_groups_excluded",
    "ambiguous_cross_label_samples_excluded",
    "train_samples",
    "test_samples",
    "evaluated_labels",
    "train_only_labels_with_fewer_than_five_groups",
    "normalized_group_overlap",
)


def resolve_contract_type_model_tag() -> str:
    return (
        os.getenv("LEXNLP_CONTRACT_TYPE_MODEL_TAG")
        or "pipeline/contract-type/0.2-runtime"
    ).strip()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate contract-type models on a labeled fixture.",
    )
    parser.add_argument(
        "--baseline-tag",
        default=resolve_contract_type_model_tag(),
        help="Catalog tag used as baseline model.",
    )
    parser.add_argument(
        "--baseline-metrics-json",
        type=Path,
        help=(
            "Optional path to committed baseline metrics JSON. "
            "When provided, baseline metrics are loaded from this file "
            "instead of executing the baseline model."
        ),
    )
    parser.add_argument(
        "--candidate-tag",
        required=True,
        help="Catalog tag used as candidate model.",
    )
    parser.add_argument(
        "--candidate-training-report-json",
        type=Path,
        help=(
            "Optional training report for enforcing the duplicate-group holdout. "
            "Requires --baseline-metrics-json with duplicate_group_holdout evidence."
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help=f"CSV fixture path (default: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-N used for accuracy_topN checks (default: 3).",
    )
    parser.add_argument(
        "--max-accuracy-top1-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate top-1 accuracy drop vs baseline.",
    )
    parser.add_argument(
        "--max-accuracy-topn-regression",
        "--max-accuracy-top3-regression",
        type=float,
        dest="max_accuracy_topn_regression",
        default=0.0,
        help=(
            "Maximum allowed candidate top-N accuracy drop vs baseline. "
            "--max-accuracy-top3-regression is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--max-f1-macro-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate macro-F1 drop vs baseline.",
    )
    parser.add_argument(
        "--max-f1-weighted-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate weighted-F1 drop vs baseline.",
    )
    parser.add_argument(
        "--max-holdout-accuracy-top1-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate holdout top-1 accuracy drop vs baseline.",
    )
    parser.add_argument(
        "--max-holdout-accuracy-topn-regression",
        "--max-holdout-accuracy-top3-regression",
        type=float,
        dest="max_holdout_accuracy_topn_regression",
        default=0.0,
        help="Maximum allowed candidate holdout top-N accuracy drop vs baseline.",
    )
    parser.add_argument(
        "--max-holdout-f1-macro-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate holdout macro-F1 drop vs baseline.",
    )
    parser.add_argument(
        "--max-holdout-f1-weighted-regression",
        type=float,
        default=0.0,
        help="Maximum allowed candidate holdout weighted-F1 drop vs baseline.",
    )
    parser.add_argument(
        "--min-candidate-accuracy-top1",
        type=float,
        default=0.0,
        help="Absolute minimum candidate top-1 accuracy.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--write-baseline-metrics-json",
        type=Path,
        help=(
            "Optional path to write canonical baseline metrics JSON "
            "(baseline_tag, fixture, top_n, metrics)."
        ),
    )
    return parser.parse_args(argv)


def load_fixture(path: Path) -> Tuple[List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")

    texts: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8", newline="") as fixture_file:
        reader = csv.DictReader(fixture_file)
        for row in reader:
            text = (row.get("Text") or "").strip()
            label = (row.get("Contract_Type") or "").strip()
            if not text:
                raise ValueError("Fixture row missing Text")
            if not label:
                raise ValueError("Fixture row missing Contract_Type")
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"Fixture file contains no rows: {path}")
    return texts, labels


def verify_fixture_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            "Fixture SHA-256 mismatch between --fixture and "
            f"--baseline-metrics-json: {digest} != {expected_sha256}"
        )


def ensure_tag_downloaded(tag: str) -> Path:
    from lexnlp.ml.catalog import get_path_from_catalog
    from lexnlp.ml.catalog.download import download_github_release

    try:
        return get_path_from_catalog(tag)
    except FileNotFoundError:
        download_github_release(tag, prompt_user=False)
        return get_path_from_catalog(tag)


def load_pipeline_for_tag(tag: str):
    from lexnlp.utils.unpickler import load_sklearn_model

    model_path = ensure_tag_downloaded(tag)
    with model_path.open("rb") as model_file:
        return load_sklearn_model(model_file)


def score_pipeline(pipeline, texts: List[str], labels: List[str], *, top_n: int) -> Dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    top_n = max(1, int(top_n))

    pred_top1 = pipeline.predict(texts)
    accuracy_top1 = float(accuracy_score(labels, pred_top1))
    f1_macro = float(f1_score(labels, pred_top1, average="macro", zero_division=0))
    f1_weighted = float(f1_score(labels, pred_top1, average="weighted", zero_division=0))

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("Pipeline does not expose predict_proba; cannot compute top-N accuracy.")

    probas = pipeline.predict_proba(texts)
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise ValueError("Pipeline is missing classes_.")

    classes = np.asarray(classes)
    top_indices = np.argsort(probas, axis=1)[:, -top_n:]
    hits = 0
    for truth, indices in zip(labels, top_indices):
        if truth in set(classes[indices].tolist()):
            hits += 1
    accuracy_topn = float(hits / len(labels))

    return {
        "accuracy_top1": accuracy_top1,
        "accuracy_topn": accuracy_topn,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def parse_metrics(raw: Dict[str, Any], source: str) -> Dict[str, float]:
    raw = dict(raw)
    # Backward compatibility: older baseline JSON used accuracy_top3.
    if "accuracy_topn" not in raw and "accuracy_top3" in raw:
        raw["accuracy_topn"] = raw["accuracy_top3"]

    missing = [key for key in REQUIRED_METRIC_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Missing metric keys in {source}: {', '.join(missing)}")

    return {key: float(raw[key]) for key in REQUIRED_METRIC_KEYS}


def load_json_object(path: Path, *, description: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} JSON must be an object")
    return payload


def load_baseline_metrics(path: Path) -> Dict[str, Any]:
    payload = load_json_object(path, description="Baseline metrics file")

    if "metrics" in payload:
        metrics = parse_metrics(payload["metrics"], f"{path}::metrics")
    elif "baseline" in payload:
        metrics = parse_metrics(payload["baseline"], f"{path}::baseline")
    else:
        raise ValueError(
            f"Baseline metrics JSON must contain either 'metrics' or 'baseline': {path}"
        )

    return {
        "metrics": metrics,
        "baseline_tag": payload.get("baseline_tag"),
        "fixture": payload.get("fixture"),
        "fixture_sha256": payload.get("fixture_sha256"),
        "top_n": payload.get("top_n"),
        "raw": payload,
    }


def require_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must contain object evidence at {key!r}")
    return value


def require_evidence_value(
    payload: Mapping[str, Any],
    key: str,
    *,
    source: str,
) -> Any:
    if key not in payload or payload[key] is None or payload[key] == "":
        raise ValueError(f"{source} is missing required evidence {key!r}")
    return payload[key]


def evaluate_duplicate_group_holdout(
    *,
    baseline_payload: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    candidate_tag: str,
    max_accuracy_top1_regression: float = 0.0,
    max_accuracy_topn_regression: float = 0.0,
    max_f1_macro_regression: float = 0.0,
    max_f1_weighted_regression: float = 0.0,
) -> Dict[str, Any]:
    """Validate and compare deterministic duplicate-group holdout evidence."""
    baseline_source = "baseline metrics::duplicate_group_holdout"
    candidate_source = "candidate training report"
    baseline_holdout = require_mapping(
        baseline_payload,
        "duplicate_group_holdout",
        source="Baseline metrics",
    )
    candidate_split = require_mapping(
        candidate_report,
        "validation_split",
        source=candidate_source,
    )
    baseline_metrics = parse_metrics(
        dict(require_mapping(baseline_holdout, "metrics", source=baseline_source)),
        f"{baseline_source}::metrics",
    )
    candidate_metrics = parse_metrics(
        dict(
            require_mapping(
                candidate_report,
                "validation_metrics",
                source=candidate_source,
            )
        ),
        f"{candidate_source}::validation_metrics",
    )

    baseline_recipe = require_evidence_value(
        baseline_payload,
        "training_recipe",
        source="Baseline metrics",
    )
    candidate_recipe = require_evidence_value(
        candidate_report,
        "training_recipe",
        source=candidate_source,
    )
    reported_candidate_tag = require_evidence_value(
        candidate_report,
        "target_tag",
        source=candidate_source,
    )

    result: Dict[str, Any] = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "checks": [],
        "passed": True,
    }

    def require_equal(evidence: str, baseline_value: Any, candidate_value: Any) -> None:
        passed = candidate_value == baseline_value
        result["checks"].append(
            {
                "scope": "holdout-evidence",
                "evidence": evidence,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "passed": passed,
            }
        )
        result["passed"] = bool(result["passed"] and passed)

    require_equal("target_tag", candidate_tag, reported_candidate_tag)
    require_equal("training_recipe", baseline_recipe, candidate_recipe)
    for key in HOLDOUT_SPLIT_IDENTITY_KEYS:
        baseline_value = require_evidence_value(
            baseline_holdout,
            key,
            source=baseline_source,
        )
        candidate_value = require_evidence_value(
            candidate_split,
            key,
            source=f"{candidate_source}::validation_split",
        )
        require_equal(f"validation_split.{key}", baseline_value, candidate_value)

    max_regressions = {
        "accuracy_top1": max_accuracy_top1_regression,
        "accuracy_topn": max_accuracy_topn_regression,
        "f1_macro": max_f1_macro_regression,
        "f1_weighted": max_f1_weighted_regression,
    }
    for metric_key, max_regression in max_regressions.items():
        baseline_value = float(baseline_metrics[metric_key])
        candidate_value = float(candidate_metrics[metric_key])
        delta = candidate_value - baseline_value
        passed = delta >= -max_regression
        result["checks"].append(
            {
                "scope": "holdout-metric",
                "metric": metric_key,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "max_regression": max_regression,
                "passed": passed,
            }
        )
        result["passed"] = bool(result["passed"] and passed)

    return result


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.top_n <= 0:
        raise ValueError("--top-n must be > 0")

    texts, labels = load_fixture(args.fixture)

    baseline_source = "tag"
    baseline_metrics_source = None
    baseline_metrics_file: Dict[str, Any] | None = None

    if args.baseline_metrics_json:
        baseline_source = "metrics-json"
        baseline_metrics_source = str(args.baseline_metrics_json)
        baseline_metrics_file = load_baseline_metrics(args.baseline_metrics_json)
        baseline_metrics = baseline_metrics_file["metrics"]

        file_baseline_tag = baseline_metrics_file.get("baseline_tag")
        if file_baseline_tag and file_baseline_tag != args.baseline_tag:
            raise ValueError(
                "Baseline tag mismatch between --baseline-tag and --baseline-metrics-json: "
                f"{args.baseline_tag!r} != {file_baseline_tag!r}"
            )

        file_fixture = baseline_metrics_file.get("fixture")
        if file_fixture:
            expected_fixture = str(args.fixture)
            if file_fixture != expected_fixture:
                raise ValueError(
                    "Fixture mismatch between --fixture and --baseline-metrics-json: "
                    f"{expected_fixture!r} != {file_fixture!r}"
                )

        fixture_sha256 = baseline_metrics_file.get("fixture_sha256")
        if not fixture_sha256:
            raise ValueError(
                "Baseline metrics JSON must declare fixture_sha256"
            )
        verify_fixture_sha256(args.fixture, fixture_sha256)

        file_top_n = baseline_metrics_file.get("top_n")
        if file_top_n is not None and int(file_top_n) != int(args.top_n):
            raise ValueError(
                "top_n mismatch between CLI and baseline metrics JSON: "
                f"{args.top_n} != {file_top_n}"
            )
    else:
        baseline_metrics = score_pipeline(
            load_pipeline_for_tag(args.baseline_tag),
            texts,
            labels,
            top_n=args.top_n,
        )

    candidate_metrics = score_pipeline(
        load_pipeline_for_tag(args.candidate_tag),
        texts,
        labels,
        top_n=args.top_n,
    )

    result = {
        "baseline_tag": args.baseline_tag,
        "candidate_tag": args.candidate_tag,
        "fixture": str(args.fixture),
        "top_n": int(args.top_n),
        "baseline_source": baseline_source,
        "baseline_metrics_json": baseline_metrics_source,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "checks": [],
        "passed": True,
    }

    def require(metric_key: str, max_regression: float) -> None:
        baseline_value = float(baseline_metrics[metric_key])
        candidate_value = float(candidate_metrics[metric_key])
        delta = candidate_value - baseline_value
        passed = delta >= -max_regression
        result["checks"].append(
            {
                "metric": metric_key,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "max_regression": max_regression,
                "passed": passed,
            }
        )
        result["passed"] = bool(result["passed"] and passed)

    require("accuracy_top1", args.max_accuracy_top1_regression)
    require("accuracy_topn", args.max_accuracy_topn_regression)
    require("f1_macro", args.max_f1_macro_regression)
    require("f1_weighted", args.max_f1_weighted_regression)

    if candidate_metrics["accuracy_top1"] < args.min_candidate_accuracy_top1:
        result["checks"].append(
            {
                "metric": "min_candidate_accuracy_top1",
                "candidate": float(candidate_metrics["accuracy_top1"]),
                "min_required": float(args.min_candidate_accuracy_top1),
                "passed": False,
            }
        )
        result["passed"] = False

    if args.candidate_training_report_json:
        if baseline_metrics_file is None:
            raise ValueError(
                "--candidate-training-report-json requires "
                "--baseline-metrics-json"
            )
        candidate_training_report = load_json_object(
            args.candidate_training_report_json,
            description="Candidate training report",
        )
        holdout_result = evaluate_duplicate_group_holdout(
            baseline_payload=baseline_metrics_file["raw"],
            candidate_report=candidate_training_report,
            candidate_tag=args.candidate_tag,
            max_accuracy_top1_regression=(
                args.max_holdout_accuracy_top1_regression
            ),
            max_accuracy_topn_regression=(
                args.max_holdout_accuracy_topn_regression
            ),
            max_f1_macro_regression=args.max_holdout_f1_macro_regression,
            max_f1_weighted_regression=(
                args.max_holdout_f1_weighted_regression
            ),
        )
        result["duplicate_group_holdout"] = holdout_result
        result["passed"] = bool(result["passed"] and holdout_result["passed"])

    if args.write_baseline_metrics_json:
        baseline_payload = {
            "schema_version": 2,
            "baseline_tag": args.baseline_tag,
            "fixture": str(args.fixture),
            "fixture_sha256": hashlib.sha256(
                args.fixture.read_bytes()
            ).hexdigest(),
            "top_n": int(args.top_n),
            "metrics": baseline_metrics,
        }
        args.write_baseline_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline_metrics_json.write_text(
            json.dumps(baseline_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
