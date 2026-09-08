"""Coverage tests for the old-pandas TypeError fallback in pandas_output."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

pd = pytest.importorskip("pandas")

from lexnlp.extract.batch.pandas_output import _maybe_convert_to_arrow, annotations_to_dataframe


@dataclass
class _StubAnnotation:
    coords: tuple[int, int]
    text: str
    locale: str = "en"
    record_type: str = "stub"


def _install_fake_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", types.ModuleType("pyarrow"))


class TestMaybeConvertToArrowTypeErrorFallback:
    def test_convert_success_path_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pyarrow(monkeypatch)
        seen: dict[str, object] = {}
        sentinel = pd.DataFrame({"converted": [1, 2]})

        def fake_convert(self: object, *args: object, **kwargs: object) -> object:
            seen.update(kwargs)
            return sentinel

        monkeypatch.setattr(pd.DataFrame, "convert_dtypes", fake_convert)
        frame = pd.DataFrame({"a": [1, 2]})
        result = _maybe_convert_to_arrow(frame, prefer_arrow=True)
        assert result is sentinel
        assert seen.get("dtype_backend") == "pyarrow"

    def test_type_error_returns_original_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pyarrow(monkeypatch)

        def fake_convert_raise(self: object, *args: object, **kwargs: object) -> object:
            raise TypeError("old pandas without dtype_backend support")

        monkeypatch.setattr(pd.DataFrame, "convert_dtypes", fake_convert_raise)
        frame = pd.DataFrame({"a": [1, 2]})
        result = _maybe_convert_to_arrow(frame, prefer_arrow=True)
        assert result is frame
        assert list(result["a"]) == [1, 2]

    def test_annotations_to_dataframe_survives_type_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pyarrow(monkeypatch)

        def fake_convert_raise(self: object, *args: object, **kwargs: object) -> object:
            raise TypeError("old pandas without dtype_backend support")

        monkeypatch.setattr(pd.DataFrame, "convert_dtypes", fake_convert_raise)
        annots = [_StubAnnotation(coords=(0, 3), text="abc"), _StubAnnotation(coords=(4, 7), text="xyz")]
        df = annotations_to_dataframe(annots, prefer_arrow=True)
        assert list(df["text"]) == ["abc", "xyz"]
        assert list(df["start"]) == [0, 4]
        assert list(df["end"]) == [3, 7]

    def test_annotations_to_dataframe_uses_converted_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_pyarrow(monkeypatch)
        sentinel = pd.DataFrame({"text": ["abc"], "start": [0], "end": [3]})

        def fake_convert(self: object, *args: object, **kwargs: object) -> object:
            assert kwargs.get("dtype_backend") == "pyarrow"
            return sentinel

        monkeypatch.setattr(pd.DataFrame, "convert_dtypes", fake_convert)
        annots = [_StubAnnotation(coords=(0, 3), text="abc")]
        result = annotations_to_dataframe(annots, prefer_arrow=True)
        assert result is sentinel
