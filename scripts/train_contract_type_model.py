#!/usr/bin/env python3
"""Train and store a runtime-compatible contract-type classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Collection, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from lexnlp.extract.en.contracts.runtime_model import (
    CONTRACT_TYPE_TRAINING_THREADS,
    CONTRACT_TYPE_CORPUS_TAG,
    RUNTIME_CONTRACT_TYPE_TAG,
    collect_contract_type_samples,
    ensure_tag_downloaded,
    train_contract_type_pipeline,
    write_pipeline_to_catalog,
)
from lexnlp.ml.artifact_abi import (
    artifact_metadata,
    assert_model_artifact_runtime,
)

DUPLICATE_GROUP_HOLDOUT_SEED = "lexnlp-global-group-holdout-v1"
TRAINING_RECIPE = (
    "contract-type-runtime-v2-all-docs-duplicate-holdout-v2-single-thread"
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a contract-type classifier from LexNLP corpus assets and write "
            "it into the local LexNLP catalog."
        )
    )
    parser.add_argument(
        "--target-tag",
        default=RUNTIME_CONTRACT_TYPE_TAG,
        help=f"Catalog tag to write (default: {RUNTIME_CONTRACT_TYPE_TAG}).",
    )
    parser.add_argument(
        "--corpus-tag",
        default=CONTRACT_TYPE_CORPUS_TAG,
        help=f"Corpus tag to read training examples from (default: {CONTRACT_TYPE_CORPUS_TAG}).",
    )
    parser.add_argument(
        "--max-docs-per-label",
        type=int,
        default=0,
        help="Maximum sampled documents per label (0 uses the complete pinned corpus).",
    )
    parser.add_argument(
        "--head-character-n",
        type=int,
        default=4000,
        help="Maximum characters read from each document.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for split/training.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=75_000,
        help="Maximum TF-IDF vocabulary size (default: 75000).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target model if it exists.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/model_training/contract_type_model_training_report.json"),
        help="Path to write training summary JSON.",
    )
    return parser.parse_args(argv)


def score(
    labels: Sequence[str],
    predictions: Sequence[str],
    probabilities,
    classes,
    *,
    top_n: int = 3,
) -> Dict[str, float]:
    classes = np.asarray(classes)
    top_indices = np.argsort(probabilities, axis=1)[:, -top_n:]
    top_n_hits = sum(
        truth in set(classes[indices].tolist())
        for truth, indices in zip(labels, top_indices)
    )
    return {
        "accuracy_top1": float(accuracy_score(labels, predictions)),
        "accuracy_top3": float(top_n_hits / len(labels)),
        "f1_macro": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def normalized_text_group(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_assignment_sha256(
    *,
    group_labels: Mapping[str, Collection[str]],
    train_groups: Collection[str],
    validation_groups: Collection[str],
    excluded_ambiguous_groups: Collection[str],
) -> str:
    """Return a stable identity for group partitions and their exact labels."""
    assigned_partitions: Dict[str, str] = {}
    partitions = {
        "train": train_groups,
        "validation": validation_groups,
        "excluded_ambiguous": excluded_ambiguous_groups,
    }
    for partition, groups in partitions.items():
        for group_hash in groups:
            if group_hash in assigned_partitions:
                raise ValueError(
                    f"Normalized group {group_hash!r} appears in multiple partitions"
                )
            assigned_partitions[group_hash] = partition

    expected_groups = set(group_labels)
    assigned_groups = set(assigned_partitions)
    if assigned_groups != expected_groups:
        raise ValueError(
            "Split partitions must cover every normalized group exactly once: "
            f"missing={sorted(expected_groups - assigned_groups)}, "
            f"unknown={sorted(assigned_groups - expected_groups)}"
        )

    assignment = []
    for group_hash in sorted(expected_groups):
        labels = sorted(group_labels[group_hash])
        if not labels:
            raise ValueError(f"Normalized group {group_hash!r} has no labels")
        assignment.append(
            {
                "group_sha256": group_hash,
                "labels": labels,
                "partition": assigned_partitions[group_hash],
            }
        )

    serialized = json.dumps(
        assignment,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_duplicate_group_holdout(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    validation_size: float = 0.2,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    """Create a deterministic holdout with no normalized-document leakage."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels length mismatch")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    group_indices: Dict[str, List[int]] = defaultdict(list)
    group_labels: Dict[str, set[str]] = defaultdict(set)
    row_groups: List[str] = []
    for index, (text, label) in enumerate(zip(texts, labels)):
        group_hash = normalized_text_group(text)
        row_groups.append(group_hash)
        group_indices[group_hash].append(index)
        group_labels[group_hash].add(label)

    ambiguous_groups = {
        group_hash
        for group_hash, labels_for_group in group_labels.items()
        if len(labels_for_group) > 1
    }
    label_groups: Dict[str, List[str]] = defaultdict(list)
    for group_hash, labels_for_group in group_labels.items():
        if group_hash in ambiguous_groups:
            continue
        label = next(iter(labels_for_group))
        label_groups[label].append(group_hash)

    test_groups: set[str] = set()
    evaluated_labels: set[str] = set()
    train_only_labels: set[str] = set()
    for label, groups in sorted(label_groups.items()):
        ranked_groups = sorted(
            groups,
            key=lambda group_hash: hashlib.sha256(
                (
                    f"{DUPLICATE_GROUP_HOLDOUT_SEED}\0"
                    f"{label}\0{group_hash}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked_groups) < 5:
            train_only_labels.add(label)
            continue
        evaluated_labels.add(label)
        test_group_count = min(
            len(ranked_groups) - 1,
            max(1, round(len(ranked_groups) * validation_size)),
        )
        test_groups.update(ranked_groups[:test_group_count])

    train_indices = [
        index
        for index, group_hash in enumerate(row_groups)
        if group_hash not in ambiguous_groups and group_hash not in test_groups
    ]
    test_indices = [
        index
        for index, group_hash in enumerate(row_groups)
        if group_hash in test_groups
    ]
    if not train_indices or not test_indices:
        raise RuntimeError("Duplicate-group holdout produced an empty partition")

    train_groups = {row_groups[index] for index in train_indices}
    report = {
        "strategy": DUPLICATE_GROUP_HOLDOUT_SEED,
        "normalization": "casefold, collapse whitespace, strip, sha256 UTF-8",
        "split_sha256": split_assignment_sha256(
            group_labels=group_labels,
            train_groups=train_groups,
            validation_groups=test_groups,
            excluded_ambiguous_groups=ambiguous_groups,
        ),
        "validation_size": validation_size,
        "unique_normalized_groups": len(group_indices),
        "ambiguous_cross_label_groups_excluded": len(ambiguous_groups),
        "ambiguous_cross_label_samples_excluded": sum(
            len(group_indices[group_hash])
            for group_hash in ambiguous_groups
        ),
        "ambiguous_groups_excluded_from_evaluation_pipeline": True,
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "evaluated_labels": len(evaluated_labels),
        "train_only_labels_with_fewer_than_five_groups": len(
            train_only_labels
        ),
        "normalized_group_overlap": len(
            train_groups & {row_groups[index] for index in test_indices}
        ),
    }
    return train_indices, test_indices, report


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    assert_model_artifact_runtime()

    corpus_archive = ensure_tag_downloaded(args.corpus_tag)
    texts, labels, counts = collect_contract_type_samples(
        corpus_archive,
        max_docs_per_label=args.max_docs_per_label,
        head_character_n=args.head_character_n,
    )

    train_indices, validation_indices, validation_split = (
        build_duplicate_group_holdout(
            texts,
            labels,
            validation_size=args.validation_size,
        )
    )
    x_train = [texts[index] for index in train_indices]
    y_train = [labels[index] for index in train_indices]
    x_val = [texts[index] for index in validation_indices]
    y_val = [labels[index] for index in validation_indices]

    validation_pipeline = train_contract_type_pipeline(
        x_train,
        y_train,
        random_state=args.random_state,
        max_features=args.max_features,
    )
    validation_predictions = validation_pipeline.predict(x_val)
    validation_metrics = score(
        y_val,
        validation_predictions,
        validation_pipeline.predict_proba(x_val),
        validation_pipeline.classes_,
    )

    final_pipeline = train_contract_type_pipeline(
        texts,
        labels,
        random_state=args.random_state,
        max_features=args.max_features,
    )
    destination = write_pipeline_to_catalog(
        pipeline=final_pipeline,
        target_tag=args.target_tag,
        force=args.force,
    )

    report = {
        "schema_version": 2,
        "training_recipe": TRAINING_RECIPE,
        "target_tag": args.target_tag,
        "target_model_path": str(destination),
        **artifact_metadata(destination),
        "corpus_tag": args.corpus_tag,
        "corpus_path": str(corpus_archive),
        "dataset": {
            "labels": len(counts),
            "samples_total": len(texts),
            "samples_train": len(x_train),
            "samples_validation": len(x_val),
            "max_docs_per_label": args.max_docs_per_label,
            "head_character_n": args.head_character_n,
        },
        "validation_metrics": validation_metrics,
        "validation_split": validation_split,
        "model_configuration": {
            "max_features": args.max_features,
            "random_state": args.random_state,
            "threadpool_limit": CONTRACT_TYPE_TRAINING_THREADS,
            "sampling": (
                "complete-pinned-corpus"
                if args.max_docs_per_label == 0
                else "canonical-archive-order-per-label-cap"
            ),
            "final_training_includes_ambiguous_cross_label_samples": True,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
