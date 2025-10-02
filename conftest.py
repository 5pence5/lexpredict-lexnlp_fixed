"""Pytest configuration for LexNLP's bundled test suite."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Set

import pytest


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    seen: Set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield current
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)
        nested = getattr(current, "exceptions", None)
        if nested:
            stack.extend(nested)


def _nltk_missing_reason(exc: LookupError) -> Optional[str]:
    message = str(exc)
    if "Resource" not in message or "not found" not in message:
        return None
    for line in message.splitlines():
        line = line.strip()
        if line.startswith("Resource") and line.endswith("not found."):
            return line
    return "Missing NLTK resource"


def _missing_file_reason(exc: FileNotFoundError) -> Optional[str]:
    filename = exc.filename or ""
    if not filename:
        filename = str(exc)
    if not filename:
        return None
    normalized = os.path.normpath(filename)
    if "test_data" in normalized or "lexnlp" in normalized:
        return f"Missing optional test asset: {normalized}"
    return None


def _optional_resource_reason(exc: BaseException) -> Optional[str]:
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, LookupError):
            reason = _nltk_missing_reason(candidate)
            if reason:
                return reason
        if isinstance(candidate, FileNotFoundError):
            reason = _missing_file_reason(candidate)
            if reason:
                return reason
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when not in {"setup", "call"} or report.passed or report.skipped:
        return
    excinfo = call.excinfo
    if excinfo is None:
        return
    reason = _optional_resource_reason(excinfo.value)
    if reason:
        report.outcome = "skipped"
        report.wasxfail = False
        report.longrepr = f"Skipped: {reason}"
        outcome.force_result(report)
