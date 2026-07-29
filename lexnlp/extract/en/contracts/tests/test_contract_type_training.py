from scripts.train_contract_type_model import build_duplicate_group_holdout


def test_duplicate_group_holdout_is_deterministic_and_leak_free():
    texts = []
    labels = []
    for label in ("A", "B"):
        for index in range(5):
            texts.append(f"{label} unique document {index}")
            labels.append(label)
    texts.extend([" A   unique DOCUMENT 0 ", "cross-label", "CROSS-LABEL"])
    labels.extend(["A", "A", "B"])
    for index in range(4):
        texts.append(f"C rare document {index}")
        labels.append("C")

    first = build_duplicate_group_holdout(texts, labels)
    second = build_duplicate_group_holdout(texts, labels)

    assert first == second
    train_indices, test_indices, report = first
    assert set(train_indices).isdisjoint(test_indices)
    assert 11 not in train_indices + test_indices
    assert 12 not in train_indices + test_indices
    assert report == {
        "strategy": "lexnlp-global-group-holdout-v1",
        "normalization": "casefold, collapse whitespace, strip, sha256 UTF-8",
        "validation_size": 0.2,
        "unique_normalized_groups": 15,
        "ambiguous_cross_label_groups_excluded": 1,
        "ambiguous_cross_label_samples_excluded": 2,
        "ambiguous_groups_excluded_from_evaluation_pipeline": True,
        "train_groups": 12,
        "test_groups": 2,
        "train_samples": 13,
        "test_samples": 2,
        "evaluated_labels": 2,
        "train_only_labels_with_fewer_than_five_groups": 1,
        "normalized_group_overlap": 0,
    }


def test_duplicate_group_holdout_honors_nondefault_validation_size():
    texts = [
        f"{label} unique document {index}"
        for label in ("A", "B")
        for index in range(5)
    ]
    labels = [label for label in ("A", "B") for _ in range(5)]

    _, test_indices, report = build_duplicate_group_holdout(
        texts,
        labels,
        validation_size=0.4,
    )

    assert len(test_indices) == 4
    assert report["validation_size"] == 0.4
    assert report["test_groups"] == 4
