"""
Utilities to build and load a Python 3.11-compatible contract-type classifier.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


# standard library
import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# third-party imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from lexnlp.ml.artifact_io import atomic_pickle_dump
from lexnlp.utils.unpickler import load_sklearn_model


LOGGER = logging.getLogger(__name__)

LEGACY_CONTRACT_TYPE_TAG = "pipeline/contract-type/0.1"
RUNTIME_CONTRACT_TYPE_TAG = "pipeline/contract-type/0.2-runtime"
CONTRACT_TYPE_CORPUS_TAG = "corpus/contract-types/0.1"
CONTRACT_TYPE_MODEL_FILENAME = "pipeline_contract_type_classifier.cloudpickle"
CONTRACT_TYPE_TRAINING_THREADS = 1


def ensure_tag_downloaded(tag: str) -> Path:
    from lexnlp.ml.catalog import get_path_from_catalog
    from lexnlp.ml.catalog.download import download_github_release

    try:
        return get_path_from_catalog(tag)
    except FileNotFoundError:
        LOGGER.info("Catalog tag missing; downloading release tag=%s", tag)
        download_github_release(tag, prompt_user=False)
        return get_path_from_catalog(tag)


def load_pipeline_for_tag(tag: str) -> Pipeline:
    path = ensure_tag_downloaded(tag)
    with path.open("rb") as model_file:
        return load_sklearn_model(model_file)


def _extract_label(member_name: str) -> str:
    # Expected shape: CONTRACT_TYPES/<LABEL>/<filename>.txt
    parts: Sequence[str] = Path(member_name).parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected corpus member path: {member_name}")
    return parts[-2]


def collect_contract_type_samples(
    archive_path: Path,
    *,
    max_docs_per_label: int,
    head_character_n: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    if max_docs_per_label < 0:
        raise ValueError("max_docs_per_label must be >= 0")
    if head_character_n <= 0:
        raise ValueError("head_character_n must be > 0")

    texts: List[str] = []
    labels: List[str] = []
    counts: Dict[str, int] = defaultdict(int)

    with tarfile.open(archive_path, mode="r:*") as archive:
        # The corpus release is integrity-pinned, including its member order.
        # Preserve that canonical order because the per-label cap makes order
        # part of the published training recipe.
        members: Iterable[tarfile.TarInfo] = archive
        for member in members:
            if not member.isfile() or not member.name.lower().endswith(".txt"):
                continue

            label = _extract_label(member.name)
            if max_docs_per_label and counts[label] >= max_docs_per_label:
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            payload = file_obj.read(head_character_n * 2)
            text = payload.decode("utf-8", errors="ignore").strip()
            if not text:
                continue

            texts.append(text[:head_character_n])
            labels.append(label)
            counts[label] += 1

    if not texts:
        raise RuntimeError(f"No samples collected from {archive_path}")
    if len(set(labels)) < 2:
        raise RuntimeError(
            "Contract-type training requires at least two labels; "
            f"found {len(set(labels))}"
        )

    return texts, labels, dict(counts)


def train_contract_type_pipeline(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    random_state: int,
    max_features: int = 75_000,
) -> Pipeline:
    if len(texts) != len(labels):
        raise ValueError("texts and labels length mismatch")
    if max_features <= 0:
        raise ValueError("max_features must be > 0")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    # BLAS reductions can differ at the last few floating-point bits when their
    # work is scheduled across a variable number of threads.  A fixed
    # single-thread fit makes release artifacts byte-reproducible and also
    # prevents runtime fallback training from oversubscribing small hosts.
    with threadpool_limits(limits=CONTRACT_TYPE_TRAINING_THREADS):
        pipeline.fit(texts, labels)
    return pipeline


def write_pipeline_to_catalog(
    *,
    pipeline: Pipeline,
    target_tag: str,
    force: bool,
) -> Path:
    from lexnlp.ml.catalog import CATALOG

    destination_dir = CATALOG / target_tag
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / CONTRACT_TYPE_MODEL_FILENAME

    if destination_path.exists() and not force:
        try:
            with destination_path.open("rb") as model_file:
                load_sklearn_model(model_file)
            return destination_path
        except Exception:
            LOGGER.warning(
                "Existing contract-type model is invalid; replacing it: %s",
                destination_path,
                exc_info=True,
            )

    atomic_pickle_dump(pipeline, destination_path)
    return destination_path


def ensure_runtime_contract_type_model(
    *,
    target_tag: str = RUNTIME_CONTRACT_TYPE_TAG,
    force: bool = False,
    max_docs_per_label: int = 0,
    head_character_n: int = 4000,
    random_state: int = 7,
    max_features: int = 75_000,
) -> Path:
    from lexnlp.ml.catalog import get_path_from_catalog

    invalid_local_model = False
    if not force:
        try:
            existing_path = get_path_from_catalog(target_tag)
        except FileNotFoundError:
            pass
        else:
            try:
                with existing_path.open("rb") as model_file:
                    load_sklearn_model(model_file)
                return existing_path
            except Exception:
                invalid_local_model = True
                LOGGER.warning(
                    "Existing runtime contract-type model is invalid; rebuilding it: %s",
                    existing_path,
                    exc_info=True,
                )

        # Prefer downloading a published runtime-compatible artifact when
        # available to avoid retraining in CI environments.
        if not invalid_local_model:
            try:
                downloaded_path = ensure_tag_downloaded(target_tag)
                with downloaded_path.open("rb") as model_file:
                    load_sklearn_model(model_file)
                return downloaded_path
            except Exception as exc:
                LOGGER.warning(
                    "Unable to load runtime contract-type model tag=%s; falling back to training. error=%s",
                    target_tag,
                    exc,
                    exc_info=True,
                )

    corpus_archive = ensure_tag_downloaded(CONTRACT_TYPE_CORPUS_TAG)
    texts, labels, _counts = collect_contract_type_samples(
        corpus_archive,
        max_docs_per_label=max_docs_per_label,
        head_character_n=head_character_n,
    )
    pipeline = train_contract_type_pipeline(
        texts,
        labels,
        random_state=random_state,
        max_features=max_features,
    )
    destination_path = write_pipeline_to_catalog(
        pipeline=pipeline,
        target_tag=target_tag,
        force=True,
    )
    LOGGER.info("Trained runtime contract-type model tag=%s at %s", target_tag, destination_path)
    return destination_path
