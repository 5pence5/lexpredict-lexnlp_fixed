"""Coverage tests for :mod:`lexnlp.utils.caching` default cache dir."""

from __future__ import annotations

from pathlib import Path

import pytest

import lexnlp.utils.caching as caching_module
from lexnlp.utils.caching import DEFAULT_CACHE_ENV


class TestDefaultCacheDirFallback:
    def test_returns_home_cache_without_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DEFAULT_CACHE_ENV, raising=False)
        assert caching_module._default_cache_dir() == Path.home() / ".cache" / "lexnlp"

    def test_explicit_empty_override_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DEFAULT_CACHE_ENV, "")
        assert caching_module._default_cache_dir() == Path.home() / ".cache" / "lexnlp"
