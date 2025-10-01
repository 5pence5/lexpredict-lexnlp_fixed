"""Runtime patching for scikit-learn compatibility.

The historical models bundled with LexNLP were trained on scikit-learn
releases prior to the introduction of the ``missing_go_to_left`` field in the
tree node structure.  When these pickles are loaded under modern versions of
scikit-learn the stricter :meth:`Tree.__setstate__` implementation raises a
``ValueError`` because the expected field is absent.

This module intercepts the unpickling process and injects a default column so
that legacy models remain usable while we modernise the training pipeline.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency during some builds
    from sklearn.tree import _tree
except Exception:  # pragma: no cover - bail out if scikit-learn unavailable
    _tree = None

try:  # pragma: no cover
    from joblib import numpy_pickle
except Exception:  # pragma: no cover
    numpy_pickle = None


def _augment_nodes(nodes: np.ndarray) -> np.ndarray:
    """Return a node array with a ``missing_go_to_left`` column."""

    if not isinstance(nodes, np.ndarray) or nodes.dtype.names is None:
        return nodes
    if "missing_go_to_left" in nodes.dtype.names:
        return nodes

    from numpy.lib import recfunctions as rfn

    missing = np.zeros(nodes.shape[0], dtype=np.uint8)
    return rfn.append_fields(
        nodes,
        "missing_go_to_left",
        data=missing,
        dtypes=np.uint8,
        usemask=False,
    )


PATCHED_TREE_CLASS = None

if _tree is not None:

    class PatchedTree(_tree.Tree):
        def __setstate__(self, state):  # type: ignore[override]
            if isinstance(state, dict) and "nodes" in state:
                nodes = _augment_nodes(state["nodes"])
                if nodes is not state["nodes"]:
                    state = dict(state)
                    state["nodes"] = nodes
            elif isinstance(state, tuple) and state:
                nodes = _augment_nodes(state[0])
                if nodes is not state[0]:
                    state = (nodes,) + tuple(state[1:])
            return _tree.Tree.__setstate__(self, state)

    PatchedTree.__name__ = "Tree"
    PatchedTree.__qualname__ = "Tree"
    PatchedTree.__module__ = "sklearn.tree._tree"
    PATCHED_TREE_CLASS = PatchedTree

if numpy_pickle is not None:
    _original_find_class = numpy_pickle.NumpyUnpickler.find_class

    def _patched_find_class(self, module, name):
        if module == "sklearn.tree.tree":
            module = "sklearn.tree"
        if module == "sklearn.ensemble.forest":
            module = "sklearn.ensemble._forest"
        if module == "sklearn.tree._tree" and name == "Tree" and PATCHED_TREE_CLASS is not None:
            return PATCHED_TREE_CLASS
        return _original_find_class(self, module, name)

    numpy_pickle.NumpyUnpickler.find_class = _patched_find_class
