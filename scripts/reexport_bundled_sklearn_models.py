#!/usr/bin/env python3
"""Re-export bundled sklearn/joblib artifacts to match the current runtime.

The legacy bundled artifacts under ``lexnlp/`` were pickled with sklearn 1.2
and trip an ``InconsistentVersionWarning`` (and a hard ``ValueError`` for
tree-based estimators) when loaded under sklearn >=1.3. ``--format skops``
re-emits each artifact as a ``skops.io`` file alongside the original pickle,
upgrading tree node arrays through :func:`lexnlp.ml.model_io.load_model`'s
``_patched_sklearn_tree_loader`` context. ``--format pickle`` keeps the legacy
behaviour (joblib re-dump) for downstream consumers that have not yet
migrated.

The companion zipped layered-definition payload uses an internal pickle pair
so we re-emit it as a ``.skops.zip`` archive when ``--format skops`` is
selected; in pickle mode we just re-pickle in place.
"""

from __future__ import annotations

import argparse
import io
import pickle
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import joblib

from lexnlp.ml.model_io import dump_model, load_model

BUNDLED_MODEL_PATHS: tuple[Path, ...] = (
    Path("lexnlp/extract/de/date_model.pickle"),
    Path("lexnlp/extract/de/model.pickle"),
    Path("lexnlp/extract/en/date_model.pickle"),
    Path("lexnlp/extract/en/addresses/addresses_clf.pickle"),
    Path("lexnlp/extract/ml/en/data/definition_model_layered.pickle.gzip"),
    Path("lexnlp/nlp/en/segments/page_segmenter.pickle"),
    Path("lexnlp/nlp/en/segments/paragraph_segmenter.pickle"),
    Path("lexnlp/nlp/en/segments/section_segmenter.pickle"),
    Path("lexnlp/nlp/en/segments/sentence_segmenter.pickle"),
    Path("lexnlp/nlp/en/segments/title_locator.pickle"),
)

LEGACY_WARNING_TOKEN = "Trying to unpickle estimator"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-export bundled sklearn/joblib model artifacts in-place.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional explicit list of paths to re-export (defaults to known bundled models).",
    )
    parser.add_argument(
        "--compress",
        type=int,
        default=3,
        help="joblib compression level when --format=pickle (default: 3).",
    )
    parser.add_argument(
        "--format",
        choices=("skops", "pickle"),
        default="skops",
        help=(
            "Output format. 'skops' (default) writes ``.skops`` siblings via "
            "skops.io; 'pickle' re-dumps the existing pickle in place via joblib."
        ),
    )
    parser.add_argument(
        "--remove-legacy",
        action="store_true",
        help=(
            "When --format=skops, delete the old ``.pickle`` after a successful "
            "skops export. Off by default so loaders can keep falling back to "
            "the pickle while consumers migrate."
        ),
    )
    return parser.parse_args(argv)


def iter_paths(args: argparse.Namespace) -> Iterable[Path]:
    if args.paths:
        for raw in args.paths:
            yield Path(raw)
    else:
        yield from BUNDLED_MODEL_PATHS


def _load_addresses_clf(path: Path):
    """Load the addresses classifier through the legacy RenameUnpickler.

    Older releases stored the artifact as a raw pickle that references
    ``sklearn.tree.tree`` (pre-1.0 module path). RenameUnpickler rewrites
    that to ``sklearn.tree._classes`` so :class:`load_model` can pick it up.
    """

    from lexnlp.utils.unpickler import renamed_load

    with path.open("rb") as f:
        try:
            return renamed_load(f)
        except (pickle.UnpicklingError, ValueError, KeyError):
            # Fall back to joblib if a previous run already saved it as a
            # joblib pickle.
            return joblib.load(path)


def _load_for_reexport(path: Path):
    """Load ``path`` using the loader best suited to its on-disk shape."""

    if path.name == "addresses_clf.pickle":
        return _load_addresses_clf(path)
    if path.name == "definition_model_layered.pickle.gzip":
        return load_layered_definition_models(path)
    # Everything else is a joblib pickle whose tree-node dtype may be older
    # than sklearn's current expectation. ``lexnlp.ml.model_io.load_model``
    # wraps ``joblib.load`` in ``_patched_sklearn_tree_loader``, which adds
    # the missing ``missing_go_to_left`` column on the fly.
    return load_model(path)


def load_layered_definition_models(path: Path):
    from lexnlp.utils.unpickler import renamed_load

    with ZipFile(path) as archive:
        payload = {}
        for name in ("term.pickle", "definition.pickle"):
            raw = archive.read(name)
            payload[name] = renamed_load(io.BytesIO(raw))
    return payload


def reexport_layered_definition_models_pickle(path: Path) -> None:
    payload = load_layered_definition_models(path)
    tmp_path = path.with_name(path.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with ZipFile(tmp_path, mode="w", compression=ZIP_STORED) as archive:
            for name in ("term.pickle", "definition.pickle"):
                archive.writestr(
                    name,
                    pickle.dumps(payload[name], protocol=pickle.HIGHEST_PROTOCOL),
                )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def reexport_layered_definition_models_skops(path: Path) -> Path:
    """Write a ``.skops.zip`` sibling holding two ``.skops`` payloads.

    Returns the path that was written.
    """

    from skops.io import dump as _skops_dump

    payload = load_layered_definition_models(path)
    target = path.with_name("definition_model_layered.skops.zip")
    tmp_path = target.with_name(target.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with ZipFile(tmp_path, mode="w", compression=ZIP_STORED) as archive:
            for src_name, obj in payload.items():
                # ``term.pickle`` -> ``term.skops``; same for definition.
                target_name = src_name.replace(".pickle", ".skops")
                buffer = io.BytesIO()
                _skops_dump(obj, buffer)
                archive.writestr(target_name, buffer.getvalue())
        tmp_path.replace(target)
        return target
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def legacy_warning_count_for_load(path: Path) -> int:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            _ = _load_for_reexport(path)
        except Exception:  # noqa: BLE001 - report load failures as 0 baseline warns
            return 0
    return sum(1 for item in captured if LEGACY_WARNING_TOKEN in str(item.message))


def _reexport_single_pickle(path: Path, compress: int) -> Path:
    """Re-pickle ``path`` in place via joblib (or plain pickle for addresses)."""
    obj = _load_for_reexport(path)
    if path.name == "addresses_clf.pickle":
        # Keep as a plain pickle so lexnlp.extract.en.addresses can keep using
        # RenameUnpickler for older module-path compatibility.
        with path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(obj, path, compress=compress)
    return path


def _sanitize_pandas_indices(obj, _seen: set[int] | None = None):
    """Replace ``pandas.Index`` attributes with plain ``list[str]``.

    Some legacy pipelines pickled a ``pandas.Index`` directly on the
    estimator (e.g. ``MODEL_DATE.columns``). Pandas 2 stores a Cython
    ``BlockValuesRefs`` inside that Index that can't be reduced through
    ``skops.io``. We don't need the Index semantics — runtime code only
    iterates with ``for i, col in enumerate(MODEL_DATE.columns)`` — so the
    safest fix is to replace the Index with a plain Python list before
    serialization. This walks dicts/lists/tuples/dataclasses/objects and
    rewrites any pandas Index it encounters in place.
    """

    try:
        import pandas as _pd
    except ImportError:  # pragma: no cover - pandas is a hard dep at runtime
        return obj

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return obj
    _seen.add(oid)

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, _pd.Index):
                obj[key] = list(value)
            else:
                _sanitize_pandas_indices(value, _seen)
        return obj
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, _pd.Index):
                obj[i] = list(item)
            else:
                _sanitize_pandas_indices(item, _seen)
        return obj
    if isinstance(obj, (set, tuple)):
        # Tuples and sets are immutable / unordered respectively, so we can't
        # rewrite a contained ``pd.Index`` in place. Recurse into each item
        # so any mutable sub-objects are still sanitized.
        for item in obj:
            _sanitize_pandas_indices(item, _seen)
        return obj

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            attrs = vars(obj)
        except TypeError:  # pragma: no cover
            attrs = None
        if attrs is not None:
            for key, value in list(attrs.items()):
                if isinstance(value, _pd.Index):
                    setattr(obj, key, list(value))
                else:
                    _sanitize_pandas_indices(value, _seen)
    return obj


def _reexport_single_skops(path: Path) -> Path:
    """Write a ``.skops`` sibling for ``path`` and return the new path."""
    obj = _load_for_reexport(path)
    _sanitize_pandas_indices(obj)
    target = path.with_suffix(".skops")
    return dump_model(obj, target)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    failures: list[str] = []

    for path in iter_paths(args):
        if not path.exists():
            failures.append(f"missing: {path}")
            continue

        before = legacy_warning_count_for_load(path)

        if args.format == "skops":
            if path.name == "definition_model_layered.pickle.gzip":
                target = reexport_layered_definition_models_skops(path)
            else:
                target = _reexport_single_skops(path)
            after = 0 if target.suffix.lower() == ".skops" else legacy_warning_count_for_load(target)
            extra = ""
            if args.remove_legacy:
                if target.exists():
                    path.unlink()
                    extra = " removed-legacy=yes"
                else:
                    extra = " removed-legacy=SKIPPED (target missing)"
            print(f"reexport: {path} -> {target} legacy_warnings before={before} after={after}{extra}")
        else:
            if path.name == "definition_model_layered.pickle.gzip":
                reexport_layered_definition_models_pickle(path)
                target = path
            else:
                target = _reexport_single_pickle(path, args.compress)
            after = legacy_warning_count_for_load(target)
            print(f"reexport: {path} legacy_warnings before={before} after={after}")

    if failures:
        for failure in failures:
            print(f"reexport: ERROR {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
