"""Pytest configuration for LexNLP's bundled test suite."""

from __future__ import annotations

import os
from typing import Optional

import pytest


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or report.passed or report.skipped:
        return
    excinfo = call.excinfo
    if excinfo is None:
        return
    reason: Optional[str] = None
    if isinstance(excinfo.value, LookupError):
        reason = _nltk_missing_reason(excinfo.value)
    elif isinstance(excinfo.value, FileNotFoundError):
        reason = _missing_file_reason(excinfo.value)
    if reason:
        report.outcome = "skipped"
        report.wasxfail = False
        report.longrepr = f"Skipped: {reason}"
        outcome.force_result(report)
