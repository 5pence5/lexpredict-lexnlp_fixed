"""Coverage tests for CancelledError propagation in async batch extraction."""

from __future__ import annotations

import asyncio

import pytest

from lexnlp.extract.batch.async_extract import _run_one


def _cancelling_extractor(text: str) -> list[str]:
    raise asyncio.CancelledError("stop")


class TestCancelledErrorPropagation:
    def test_cancelled_error_propagates_when_not_raising(self) -> None:
        async def _run() -> None:
            await _run_one(0, "x", _cancelling_extractor, asyncio.Semaphore(1), False)

        with pytest.raises(asyncio.CancelledError, match="stop"):
            asyncio.run(_run())

    def test_cancelled_error_propagates_when_raising(self) -> None:
        async def _run() -> None:
            await _run_one(0, "x", _cancelling_extractor, asyncio.Semaphore(1), True)

        with pytest.raises(asyncio.CancelledError, match="stop"):
            asyncio.run(_run())
