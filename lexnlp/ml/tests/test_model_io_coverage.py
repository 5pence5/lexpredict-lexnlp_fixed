"""Coverage tests for lexnlp/ml/model_io.py.

Covers load_bundled_model's skops-sibling preference, the
_patch_legacy_sklearn_estimator container/attribute shims, the
_patched_sklearn_tree_loader ImportError path, and the old-dtype compat
rewrite inside the tree-node validator patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree

from lexnlp.ml.model_io import (
    _load_legacy,
    _patch_legacy_sklearn_estimator,
    _patched_sklearn_tree_loader,
    dump_model,
    load_bundled_model,
    load_model,
)


class _OldStyleEstimator:
    """Mimics a pre-1.3 estimator exposing base_estimator instead of estimator."""

    def __init__(self, inner) -> None:
        self.base_estimator = inner


class _TreeHolder:
    """Mimics a legacy estimator pickled without monotonic_cst."""

    def __init__(self) -> None:
        self.tree_ = object()


class _ForestHolder:
    """Mimics a legacy ensemble pickled without monotonic_cst."""

    def __init__(self, members: list, estimator=None) -> None:
        self.estimators_ = members
        if estimator is not None:
            self.estimator = estimator


class _Wrapper:
    """Plain wrapper holding nested containers, primitives, and a legacy shim target."""

    def __init__(self) -> None:
        self.name = "wrapper"
        self.count = 3
        self.child = {"holder": _TreeHolder()}
        self.items = [_TreeHolder()]


class TestLoadBundledModel:
    def test_prefers_skops_sibling(self, tmp_path: Path) -> None:
        import pickle

        legacy = tmp_path / "model.pickle"
        with legacy.open("wb") as fh:
            pickle.dump({"v": "legacy"}, fh)
        dump_model({"v": "skops"}, tmp_path / "model.skops")
        assert load_bundled_model(legacy) == {"v": "skops"}

    def test_prefers_skops_sibling_with_string_path(self, tmp_path: Path) -> None:
        import pickle

        legacy = tmp_path / "model.pickle"
        with legacy.open("wb") as fh:
            pickle.dump({"v": "legacy"}, fh)
        dump_model({"v": "skops"}, tmp_path / "model.skops")
        assert load_bundled_model(str(legacy)) == {"v": "skops"}

    def test_falls_back_to_legacy_without_sibling(self, tmp_path: Path) -> None:
        import pickle

        legacy = tmp_path / "solo.pickle"
        with legacy.open("wb") as fh:
            pickle.dump({"v": "legacy-only"}, fh)
        assert load_bundled_model(legacy) == {"v": "legacy-only"}
        assert load_model(legacy) == {"v": "legacy-only"}


class TestPatchLegacyEstimator:
    def test_walks_pipeline_steps(self) -> None:
        pipe = Pipeline([("scaler", StandardScaler())])
        assert _patch_legacy_sklearn_estimator(pipe) is pipe
        assert pipe.steps[0][1] is not None

    def test_dict_and_list_containers_walked(self) -> None:
        holder = _TreeHolder()
        obj = {"a": [holder], "b": (holder,)}
        assert _patch_legacy_sklearn_estimator(obj) is obj
        assert holder.monotonic_cst is None

    def test_set_container_walked(self) -> None:
        holder = _TreeHolder()
        obj = {holder}
        assert _patch_legacy_sklearn_estimator(obj) is obj
        assert holder.monotonic_cst is None

    def test_cycle_does_not_recurse_forever(self) -> None:
        holder = _TreeHolder()
        obj: list = [holder]
        obj.append(obj)
        assert _patch_legacy_sklearn_estimator(obj) is obj
        assert holder.monotonic_cst is None

    def test_base_estimator_aliased_to_estimator(self) -> None:
        inner = DecisionTreeClassifier()
        obj = _OldStyleEstimator(inner)
        assert not hasattr(obj, "estimator")
        _patch_legacy_sklearn_estimator(obj)
        assert obj.estimator is inner

    def test_tree_holder_gains_monotonic_cst(self) -> None:
        holder = _TreeHolder()
        assert not hasattr(holder, "monotonic_cst")
        assert _patch_legacy_sklearn_estimator(holder) is holder
        assert holder.monotonic_cst is None

    def test_estimators_list_walked_and_patched(self) -> None:
        first = _TreeHolder()
        second = _TreeHolder()
        forest = _ForestHolder([first, second])
        _patch_legacy_sklearn_estimator(forest)
        assert forest.monotonic_cst is None
        assert first.monotonic_cst is None
        assert second.monotonic_cst is None

    def test_nested_estimator_attribute_walked(self) -> None:
        inner_holder = _TreeHolder()
        inner = _OldStyleEstimator(inner_holder)
        forest = _ForestHolder([_TreeHolder()], estimator=inner)
        _patch_legacy_sklearn_estimator(forest)
        assert inner_holder.monotonic_cst is None
        assert inner.estimator is inner_holder

    def test_wrapper_dict_attributes_walked_and_primitives_skipped(self) -> None:
        wrapper = _Wrapper()
        assert _patch_legacy_sklearn_estimator(wrapper) is wrapper
        assert wrapper.name == "wrapper"
        assert wrapper.count == 3
        assert wrapper.child["holder"].monotonic_cst is None
        assert wrapper.items[0].monotonic_cst is None

    def test_class_objects_not_entered(self) -> None:
        assert _patch_legacy_sklearn_estimator(_TreeHolder) is _TreeHolder

    def test_legacy_pickle_with_pipeline_loads(self, tmp_path: Path) -> None:
        import pickle

        pipe = Pipeline([("scaler", StandardScaler())])
        path = tmp_path / "pipe.pickle"
        with path.open("wb") as fh:
            pickle.dump(pipe, fh)
        assert isinstance(_load_legacy(path), Pipeline)


class TestPatchedSklearnTreeLoader:
    def test_import_error_path_yields_without_patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "numpy", None)
        with _patched_sklearn_tree_loader():
            pass

    def test_patch_restores_original(self) -> None:
        original = _tree._check_node_ndarray
        with _patched_sklearn_tree_loader():
            assert _tree._check_node_ndarray is not original
        assert _tree._check_node_ndarray is original

    def test_compat_rewrites_old_node_dtype(self) -> None:
        expected = numpy.dtype(_tree.NODE_DTYPE)
        old_descr = [(n, expected.fields[n][0]) for n in expected.names if n != "missing_go_to_left"]
        old_dtype = numpy.dtype(old_descr)
        assert "missing_go_to_left" not in (old_dtype.names or ())
        node_old = numpy.zeros(2, dtype=old_dtype)
        node_old["left_child"] = [1, 2]
        original = _tree._check_node_ndarray
        with _patched_sklearn_tree_loader():
            compat = _tree._check_node_ndarray
            assert compat is not original
            result = compat(node_old, expected)
            assert result.dtype.names == expected.names
            assert list(result["left_child"]) == [1, 2]
            assert list(result["missing_go_to_left"]) == [0, 0]
            node_new = numpy.zeros(3, dtype=expected)
            node_new["missing_go_to_left"] = [0, 1, 0]
            passthrough = compat(node_new, expected)
            assert passthrough.dtype.names == expected.names
            assert list(passthrough["missing_go_to_left"]) == [0, 1, 0]
        assert _tree._check_node_ndarray is original
