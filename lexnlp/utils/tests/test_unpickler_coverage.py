"""Coverage tests for ``lexnlp.utils.unpickler``.

Exercises ``RenameUnpickler.find_class`` (line 38) both directly and via
``renamed_load``, plus the lazy-import / context-manager / patch body of
``renamed_load`` (lines 49, 54, 55).
"""

from __future__ import annotations

import io
import pickle
from collections import OrderedDict
from decimal import Decimal

from lexnlp.utils.unpickler import RenameUnpickler, renamed_load


class _LegacyLike:
    """Picklable stand-in with a legacy ``base_estimator`` attribute."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator


class TestRenameUnpicklerFindClass:
    def test_passthrough_module(self) -> None:
        unpickler = RenameUnpickler(io.BytesIO(b""))
        assert unpickler.find_class("collections", "OrderedDict") is OrderedDict

    def test_legacy_sklearn_module_rename(self) -> None:
        from sklearn.tree import DecisionTreeClassifier

        unpickler = RenameUnpickler(io.BytesIO(b""))
        assert unpickler.find_class("sklearn.tree.tree", "DecisionTreeClassifier") is DecisionTreeClassifier

    def test_legacy_ensemble_module_rename(self) -> None:
        from sklearn.ensemble._forest import RandomForestClassifier

        unpickler = RenameUnpickler(io.BytesIO(b""))
        assert unpickler.find_class("sklearn.ensemble.forest", "RandomForestClassifier") is RandomForestClassifier


class TestRenamedLoad:
    def test_roundtrip_global_requiring_object(self) -> None:
        payload = OrderedDict([("a", 1)])
        result = renamed_load(io.BytesIO(pickle.dumps(payload)))
        assert result == OrderedDict([("a", 1)])
        assert isinstance(result, OrderedDict)

    def test_roundtrip_decimal(self) -> None:
        result = renamed_load(io.BytesIO(pickle.dumps(Decimal("1.5"))))
        assert result == Decimal("1.5")
        assert isinstance(result, Decimal)

    def test_roundtrip_plain_dict(self) -> None:
        assert renamed_load(io.BytesIO(pickle.dumps({"a": 1}))) == {"a": 1}

    def test_applies_legacy_estimator_patch(self) -> None:
        """``renamed_load`` runs ``_patch_legacy_sklearn_estimator`` (line 55)."""
        obj = _LegacyLike(base_estimator="sentinel")
        assert not hasattr(obj, "estimator")
        result = renamed_load(io.BytesIO(pickle.dumps(obj)))
        assert isinstance(result, _LegacyLike)
        assert result.base_estimator == "sentinel"
        assert result.estimator == "sentinel"
