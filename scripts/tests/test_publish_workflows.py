from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_WORKFLOWS = (
    PROJECT_ROOT / ".github/workflows/publish-contract-model.yml",
    PROJECT_ROOT / ".github/workflows/publish-contract-type-runtime-model.yml",
)


@pytest.mark.parametrize("workflow_path", PUBLISH_WORKFLOWS, ids=lambda path: path.stem)
def test_model_publish_workflow_preserves_release_privilege_boundary(workflow_path: Path):
    workflow = workflow_path.read_text(encoding="utf-8")
    build, publish = workflow.split("\n  publish:\n", maxsplit=1)

    assert workflow.count("contents: write") == 1
    assert workflow.count(
        "github.ref_type == 'branch' && "
        "github.ref_name == github.event.repository.default_branch"
    ) == 2
    assert "persist-credentials: false" in build
    assert "contents: write" not in build

    assert "needs: build" in publish
    assert "contents: write" in publish
    assert "name: model-release" in publish
    assert "actions/checkout@" not in publish
    assert "actions/download-artifact@" in publish
    assert (
        "contents/lexnlp/ml/catalog/release_asset_manifest.json?ref=${GITHUB_SHA}"
        in publish
    )
    assert "application/vnd.github.raw+json" in publish
    assert "steps.trusted_manifest.outputs.model_filename" in publish
    assert "steps.trusted_manifest.outputs.model_size" in publish
    assert "steps.trusted_manifest.outputs.model_sha256" in publish
    assert "needs.build.outputs.model_" not in publish


@pytest.mark.parametrize("workflow_path", PUBLISH_WORKFLOWS, ids=lambda path: path.stem)
def test_model_publish_workflow_never_replaces_existing_release_bytes(workflow_path: Path):
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "--clobber" not in workflow
    assert "Release asset already exists with the verified bytes" in workflow
    assert "Refusing to replace differing release asset" in workflow


def test_contract_type_publish_gates_enforce_training_holdout_metrics():
    workflow = PUBLISH_WORKFLOWS[1].read_text(encoding="utf-8")
    required_arguments = (
        "--candidate-training-report-json",
        "--max-holdout-accuracy-top1-regression",
        "--max-holdout-accuracy-topn-regression",
        "--max-holdout-f1-macro-regression",
        "--max-holdout-f1-weighted-regression",
    )

    for argument in required_arguments:
        assert workflow.count(argument) == 2
