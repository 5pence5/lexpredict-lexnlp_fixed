"""Ensure legacy nose expectations still work on modern Python."""

from __future__ import annotations

import unittest
import unittest.runner

# Nose imports ``_TextTestResult`` from both ``unittest`` and
# ``unittest.runner``.  The private alias was dropped in Python 3.12 so we
# recreate it pointing at the public ``TextTestResult`` implementation.
if not hasattr(unittest, "_TextTestResult"):
    unittest._TextTestResult = unittest.TextTestResult  # type: ignore[attr-defined]

if not hasattr(unittest.runner, "_TextTestResult"):
    unittest.runner._TextTestResult = unittest.TextTestResult  # type: ignore[attr-defined]
