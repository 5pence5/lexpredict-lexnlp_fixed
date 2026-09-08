"""Coverage tests for :mod:`lexnlp.utils.pandas_config` pyarrow paths."""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from lexnlp.utils.pandas_config import convert_to_arrow, read_csv_arrow


@pytest.fixture
def fake_pyarrow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Satisfy the ``import pyarrow`` availability probes without pyarrow installed."""
    module = ModuleType("pyarrow")
    monkeypatch.setitem(sys.modules, "pyarrow", module)
    return module


class _RecordingFrame:
    """Minimal stand-in for a DataFrame supporting ``convert_dtypes``."""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def convert_dtypes(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.result


class _FailingFrame(_RecordingFrame):
    def convert_dtypes(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        raise TypeError("dtype_backend not supported by this pandas build")


def test_convert_to_arrow_forwards_pyarrow_backend(fake_pyarrow: ModuleType) -> None:
    sentinel = object()
    frame = _RecordingFrame(result=sentinel)
    assert convert_to_arrow(frame) is sentinel  # type: ignore[arg-type]
    assert frame.calls == [{"dtype_backend": "pyarrow"}]


def test_convert_to_arrow_type_error_returns_frame_unchanged(fake_pyarrow: ModuleType) -> None:
    frame = _FailingFrame()
    assert convert_to_arrow(frame) is frame  # type: ignore[arg-type]
    assert frame.calls == [{"dtype_backend": "pyarrow"}]


def test_read_csv_arrow_forwards_pyarrow_backend(
    fake_pyarrow: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("alias,name\nalias-a,Name A\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    sentinel = pd.DataFrame({"alias": ["alias-a"]})
    original_read_csv = pd.read_csv

    def spy_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        assert args[0] == path or str(args[0]) == str(path)
        return sentinel

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    assert read_csv_arrow(path) is sentinel
    assert captured.get("dtype_backend") == "pyarrow"
    assert original_read_csv is not None


def test_read_csv_arrow_preserves_explicit_backend(
    fake_pyarrow: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("alias,name\nalias-a,Name A\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    sentinel = pd.DataFrame({"alias": ["alias-a"]})

    def spy_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    assert read_csv_arrow(path, dtype_backend="numpy_nullable") is sentinel
    assert captured.get("dtype_backend") == "numpy_nullable"
