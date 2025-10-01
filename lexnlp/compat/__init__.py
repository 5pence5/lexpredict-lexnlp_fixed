"""Compatibility shims applied at import time."""

# Re-export modules so that side-effecting imports run eagerly.
from . import sklearn_patch  # noqa: F401
from . import unittest_patch  # noqa: F401
