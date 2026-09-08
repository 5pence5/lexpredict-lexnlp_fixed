"""Model serialization helpers.

LexNLP historically persisted scikit-learn pipelines via ``pickle`` /
``cloudpickle`` / ``joblib``. Those formats are tightly coupled to the
Python / sklearn versions that produced them (see the tree-pickle ABI
break between sklearn 1.2 and 1.3+) and they execute arbitrary code on
load, making them unsuitable as a persistence target for shipped
artifacts.

This module introduces `skops.io <https://skops.readthedocs.io/>`_ as
the forward-looking successor. ``skops`` serializes sklearn estimators
via a restricted schema that does not execute arbitrary code on load.

``CANONICAL_SUFFIX`` is the extension used for new artifacts written by
``dump_model``. ``load_model`` accepts either ``.skops`` files or any
legacy pickle-based file (``.pickle`` / ``.cloudpickle`` / joblib) so
existing bundled assets continue to work while we migrate.
"""

from __future__ import annotations

import logging
import pickle
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from skops.io import dump as _skops_dump
from skops.io import get_untrusted_types
from skops.io import load as _skops_load

LOGGER = logging.getLogger(__name__)

CANONICAL_SUFFIX = ".skops"
_LEGACY_SUFFIXES = frozenset({".pickle", ".pkl", ".cloudpickle", ".joblib"})
_sklearn_patch_lock = threading.Lock()

#: Explicit allow-list of custom-type names accepted when loading a skops
#: artifact with ``trusted=True``. Anything outside this set causes a load
#: failure even with ``trusted=True``, which prevents an attacker-controlled
#: artifact from smuggling unknown types past the skops gate. Callers that
#: legitimately need to load a broader set of types can extend it via the
#: ``extra_trusted`` argument on :func:`load_model` / :func:`_load_skops`.
DEFAULT_TRUSTED_ALLOWLIST: frozenset[str] = frozenset(
    {
        # NumPy ------------------------------------------------------------
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.int8",
        "numpy.int16",
        "numpy.int32",
        "numpy.int64",
        "numpy.float32",
        "numpy.float64",
        "numpy.bool_",
        # scikit-learn core ----------------------------------------------
        "sklearn.pipeline.Pipeline",
        "sklearn.pipeline.FeatureUnion",
        "sklearn.compose._column_transformer.ColumnTransformer",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.preprocessing._data.MinMaxScaler",
        "sklearn.preprocessing._data.RobustScaler",
        "sklearn.preprocessing._label.LabelEncoder",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.feature_extraction.text.TfidfVectorizer",
        "sklearn.feature_extraction.text.CountVectorizer",
        "sklearn.feature_extraction.text.TfidfTransformer",
        # sklearn estimators LexNLP actually ships ------------------------
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.linear_model._logistic.LogisticRegressionCV",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble._forest.ExtraTreesClassifier",
        "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.svm._classes.LinearSVC",
        "sklearn.feature_selection._univariate_selection.SelectKBest",
        "sklearn.feature_selection._univariate_selection.f_classif",
        "sklearn.feature_selection._univariate_selection.chi2",
        "sklearn.feature_selection._univariate_selection.f_regression",
        # NLTK tokenizer types embedded in bundled segmenter artifacts ----
        "nltk.tokenize.punkt.PunktSentenceTokenizer",
        "nltk.tokenize.punkt.PunktParameters",
        "nltk.tokenize.punkt.PunktLanguageVars",
        "nltk.tokenize.punkt.PunktToken",
        # LexNLP's own token-sequence classifier, embedded in the bundled
        # definition/term detector artifacts. Trusting our own class is safe
        # and keeps the skops gate closed to everything else.
        "lexnlp.extract.ml.classifier.token_sequence_model.TokenSequenceClassifierModel",
    }
)


def is_skops_path(path: Path) -> bool:
    """Return ``True`` when ``path`` is a skops artifact (by suffix)."""

    return path.suffix.lower() == CANONICAL_SUFFIX


def load_bundled_model(legacy_path: str | Path, **load_kwargs: Any) -> Any:
    """Load a bundled artifact, preferring its ``.skops`` sibling.

    LexNLP ships several legacy ``.pickle`` artifacts that have a re-exported
    ``.skops`` sibling. Loaders that want the safer skops path with no risk
    of falling through to the pickle ABI break call this helper, which:

    1. checks for ``Path(legacy_path).with_suffix('.skops')`` and loads that
       (with ``trusted=True`` against :data:`DEFAULT_TRUSTED_ALLOWLIST`);
    2. otherwise falls back to ``load_model(legacy_path)`` so callers can
       keep the bundled pickle for development setups that did not run
       ``scripts/reexport_bundled_sklearn_models.py``.
    """

    legacy = Path(legacy_path)
    skops_sibling = legacy.with_suffix(CANONICAL_SUFFIX)
    if skops_sibling.exists():
        return load_model(skops_sibling, trusted=True, **load_kwargs)
    return load_model(legacy, **load_kwargs)


def dump_model(obj: Any, path: Path) -> Path:
    """Persist ``obj`` at ``path`` using skops.

    ``path`` is normalized to use :data:`CANONICAL_SUFFIX` so callers
    cannot accidentally round-trip a skops artifact through a legacy
    pickle extension.
    """

    path = Path(path)
    if not is_skops_path(path):
        path = path.with_suffix(CANONICAL_SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    _skops_dump(obj, path)
    return path


def _load_legacy(path: Path) -> Any:
    """Load a legacy pickle / cloudpickle / joblib artifact.

    Legacy loaders are imported lazily because ``cloudpickle`` and
    ``joblib`` are optional for skops-only consumers.
    """

    suffix = path.suffix.lower()
    if suffix in (".pickle", ".pkl"):
        with path.open("rb") as file:
            # Run the raw-pickle path inside the same sklearn-tree patch as
            # the joblib path so pre-1.3 Tree node dtypes upgrade on load
            # even for uncompressed pickles (e.g. ``addresses_clf.pickle``
            # which starts with the raw protocol-4 framing byte 0x80).
            try:
                with _patched_sklearn_tree_loader():
                    return _patch_legacy_sklearn_estimator(pickle.load(file))
            except (pickle.UnpicklingError, ValueError, EOFError, AttributeError) as exc:
                # Older joblib payloads — including ``compress=0`` ones that do
                # NOT start with the zlib framing byte — can fail the raw
                # ``pickle.load`` path because joblib prepends its own header.
                # Always retry via the joblib loader and only re-raise the
                # original exception if joblib also rejects the file.
                LOGGER.debug("Falling back to joblib-compatible legacy loader: %s", path)
                try:
                    return _load_legacy_joblib(path)
                except Exception:
                    raise exc
    if suffix == ".cloudpickle":
        from cloudpickle import load as cloudpickle_load

        with path.open("rb") as file:
            return cloudpickle_load(file)
    if suffix == ".joblib":
        return _load_legacy_joblib(path)
    # Reject unknown suffixes instead of attempting unsafe deserialization
    raise ValueError(
        f"Unsupported file suffix '{suffix}' for legacy model loading. "
        f"Expected one of: {', '.join(sorted(_LEGACY_SUFFIXES))}"
    )


def _load_legacy_joblib(path: Path) -> Any:
    """Load a legacy joblib artifact with sklearn tree ABI shims when needed."""

    import joblib

    with _patched_sklearn_tree_loader():
        return _patch_legacy_sklearn_estimator(joblib.load(path))


def _patch_legacy_sklearn_estimator(obj: Any, _seen: set[int] | None = None) -> Any:
    """Populate sklearn runtime attrs that older pickles did not persist.

    Walks containers (dict/list/tuple/set), sklearn pipelines (``steps``),
    ensembles (``estimators_``) and finally any arbitrary ``__dict__`` so
    classifiers nested inside non-sklearn wrapper classes (e.g.
    :class:`BaseTokenSequenceClassifierModel`) also receive the shims.
    ``_seen`` breaks cycles by object id.
    """

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return obj
    _seen.add(oid)

    if isinstance(obj, dict):
        for value in obj.values():
            _patch_legacy_sklearn_estimator(value, _seen)
        return obj
    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            _patch_legacy_sklearn_estimator(value, _seen)
        return obj

    if hasattr(obj, "steps"):
        for _, step in obj.steps:
            _patch_legacy_sklearn_estimator(step, _seen)

    if hasattr(obj, "base_estimator") and not hasattr(obj, "estimator"):
        obj.estimator = obj.base_estimator
    if hasattr(obj, "tree_") and not hasattr(obj, "monotonic_cst"):
        obj.monotonic_cst = None
    if hasattr(obj, "estimators_") and not hasattr(obj, "monotonic_cst"):
        obj.monotonic_cst = None
        for estimator in obj.estimators_:
            _patch_legacy_sklearn_estimator(estimator, _seen)
        if hasattr(obj, "estimator"):
            _patch_legacy_sklearn_estimator(obj.estimator, _seen)

    # Recurse into plain Python wrapper objects (e.g. a
    # ``BaseTokenSequenceClassifierModel`` that stores a real sklearn
    # ``Pipeline`` on ``.model``) so their nested estimators also pick up
    # the shims. Guard with ``__dict__`` so we skip Cython / built-in
    # types whose attribute storage is opaque.
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            attrs = vars(obj).values()
        except TypeError:  # pragma: no cover — __dict__ proxy objects
            attrs = ()
        for value in attrs:
            # Skip primitives to avoid pointless recursion.
            if isinstance(value, (str, bytes, int, float, bool, type(None))):
                continue
            _patch_legacy_sklearn_estimator(value, _seen)

    return obj


@contextmanager
def _patched_sklearn_tree_loader():
    """Patch sklearn's tree-node validator so pre-1.3 models still load."""

    _sklearn_patch_lock.acquire()
    try:
        try:
            import numpy
            from sklearn.tree import _tree
        except ImportError:
            yield
            return

        original = _tree._check_node_ndarray

        def compat(node_ndarray, expected_dtype):
            names = getattr(getattr(node_ndarray, "dtype", None), "names", None)
            expected_names = getattr(expected_dtype, "names", None)
            if (
                names
                and expected_names
                and "missing_go_to_left" in expected_names
                and "missing_go_to_left" not in names
            ):
                patched = numpy.empty(node_ndarray.shape, dtype=expected_dtype)
                for name in names:
                    patched[name] = node_ndarray[name]
                patched["missing_go_to_left"] = 0
                node_ndarray = patched
            return original(node_ndarray, expected_dtype)

        _tree._check_node_ndarray = compat
        try:
            yield
        finally:
            _tree._check_node_ndarray = original
    finally:
        _sklearn_patch_lock.release()


def load_model(
    path: Path,
    *,
    trusted: bool = False,
    extra_trusted: tuple[str, ...] = (),
) -> Any:
    """Load a model written by :func:`dump_model` or by a legacy dumper.

    ``trusted=True`` intersects the artifact's declared custom types with
    :data:`DEFAULT_TRUSTED_ALLOWLIST` (plus ``extra_trusted`` if
    supplied) before handing them to :func:`skops.io.load`. Anything
    outside the allow-list triggers a load failure, so an
    attacker-controlled artifact cannot smuggle unknown types past the
    skops gate even with ``trusted=True``. The default (``trusted=False``)
    keeps the original fail-closed path.
    """

    path = Path(path)
    if is_skops_path(path):
        return _load_skops(path, trusted=trusted, extra_trusted=extra_trusted)

    if path.suffix.lower() in _LEGACY_SUFFIXES:
        LOGGER.debug("Loading legacy model via pickle-family loader: %s", path)
        return _load_legacy(path)

    # Unknown suffixes must not silently fall back to pickle-family loaders
    # because that broadens the unsafe deserialization surface. Try skops
    # (some skops artifacts may not use the canonical suffix); otherwise
    # surface a ValueError listing the supported suffixes.
    try:
        return _load_skops(path, trusted=trusted, extra_trusted=extra_trusted)
    except Exception as exc:
        raise ValueError(
            f"Unsupported model suffix '{path.suffix}'. Use '{CANONICAL_SUFFIX}' or one of {sorted(_LEGACY_SUFFIXES)}."
        ) from exc


def _load_skops(
    path: Path,
    *,
    trusted: bool,
    extra_trusted: tuple[str, ...] = (),
) -> Any:
    """Invoke :func:`skops.io.load` with a ``trusted`` argument that
    matches the current skops API (list of allowed custom type names).

    When ``trusted=True`` we intersect the artifact's declared custom
    types with :data:`DEFAULT_TRUSTED_ALLOWLIST` (plus ``extra_trusted``)
    so an attacker-controlled artifact cannot smuggle unknown types
    past the skops gate even with ``trusted=True``.
    """

    if not trusted:
        # Fail-closed path: hand an empty list so skops enforces its own
        # default trusted set and raises UntrustedTypesFoundException.
        # Avoid scanning the artifact's declared types on this common path.
        return _skops_load(path, trusted=[])

    declared = list(get_untrusted_types(file=path) or [])
    allowed = DEFAULT_TRUSTED_ALLOWLIST | frozenset(extra_trusted)
    rejected = [name for name in declared if name not in allowed]
    if rejected:
        raise ValueError(
            f"Refusing to load skops artifact {path}: contains types outside "
            f"the trusted allow-list: {sorted(rejected)}. Extend the "
            "allow-list via the ``extra_trusted`` argument if these are "
            "known-safe for your deployment."
        )
    return _skops_load(path, trusted=declared)
