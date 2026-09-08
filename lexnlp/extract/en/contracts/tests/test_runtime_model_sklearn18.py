"""Smoke tests for ``contract-type/0.2-runtime`` under sklearn >=1.8 + skops.

These exercise the real ``train_contract_type_pipeline`` /
``write_pipeline_to_catalog`` / ``load_model`` round-trip on a tiny
synthetic corpus so we catch sklearn API drift (e.g. the 1.7 deprecation
of ``LogisticRegression.multi_class``) without depending on the corpus
GitHub release.
"""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from pathlib import Path

import pytest


@pytest.fixture()
def tiny_corpus() -> tuple[list[str], list[str]]:
    """Three labels, three docs each — just enough to fit + predict."""

    texts = [
        "This master service agreement governs delivery of professional services.",
        "The parties agree to the terms of services rendered under this MSA.",
        "Statement of work for the consulting engagement and its deliverables.",
        "This non-disclosure agreement protects confidential information.",
        "Confidentiality and non-disclosure obligations of the receiving party.",
        "Each party agrees to keep the trade secrets in strict confidence.",
        "This software license agreement permits use of the licensed product.",
        "The licensee may use the software subject to the licensing terms.",
        "End user license agreement and product license grant.",
    ]
    labels = ["MSA"] * 3 + ["NDA"] * 3 + ["LICENSE"] * 3
    return texts, labels


def test_pipeline_fits_under_sklearn_18(tiny_corpus: tuple[list[str], list[str]]) -> None:
    """``train_contract_type_pipeline`` must fit without sklearn deprecations."""
    import warnings

    from lexnlp.extract.en.contracts.runtime_model import train_contract_type_pipeline

    texts, labels = tiny_corpus
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pipeline = train_contract_type_pipeline(texts, labels, random_state=7)
    # Smoke-prediction: make sure the trained pipeline can score a doc.
    predictions = pipeline.predict(texts)
    assert set(predictions) <= set(labels), f"Pipeline emitted unexpected labels: {set(predictions) - set(labels)}"


def test_pipeline_round_trips_via_skops(tmp_path: Path, tiny_corpus: tuple[list[str], list[str]]) -> None:
    """``write_pipeline_to_catalog`` + ``load_model`` must preserve predictions."""

    from lexnlp.extract.en.contracts import runtime_model
    from lexnlp.extract.en.contracts.runtime_model import (
        CONTRACT_TYPE_MODEL_FILENAME,
        train_contract_type_pipeline,
        write_pipeline_to_catalog,
    )
    from lexnlp.ml import catalog as catalog_mod
    from lexnlp.ml.model_io import load_model

    texts, labels = tiny_corpus
    pipeline = train_contract_type_pipeline(texts, labels, random_state=7)
    expected = list(pipeline.predict(texts))

    target_tag = "pipeline/contract-type/0.2-runtime-test"

    # Redirect the catalog to a writable temp dir so we don't touch the user's
    # ``~/.lexnlp/`` cache during tests.
    _MISSING = object()  # sentinel so we can distinguish absent vs None
    original_catalog = catalog_mod.CATALOG
    original_runtime_catalog = getattr(runtime_model, "CATALOG", _MISSING)
    catalog_mod.CATALOG = tmp_path
    runtime_model.CATALOG = tmp_path  # type: ignore[attr-defined] - runtime_model.CATALOG is set dynamically and may not be declared in stubs
    try:
        destination, wrote = write_pipeline_to_catalog(
            pipeline=pipeline,
            target_tag=target_tag,
            force=True,
        )
    finally:
        catalog_mod.CATALOG = original_catalog
        if original_runtime_catalog is _MISSING:
            delattr(runtime_model, "CATALOG")
        else:
            runtime_model.CATALOG = original_runtime_catalog

    assert wrote is True
    assert destination.exists()
    assert destination.suffix == ".skops"
    assert destination.name == CONTRACT_TYPE_MODEL_FILENAME

    reloaded = load_model(destination, trusted=True)
    assert list(reloaded.predict(texts)) == expected
