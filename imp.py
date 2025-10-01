"""Compatibility shim for the removed :mod:`imp` module on Python 3.12+.

This minimal implementation provides the subset of the legacy API that the
`nose` test runner still relies on so that the existing test suite can execute
without pulling in an alternative dependency.  The behaviour is intentionally
limited to the common module loading patterns exercised by the project tests.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional, Tuple

# Legacy constants preserved for consumers of :mod:`imp`.
PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
PY_RESOURCE = 4
PKG_DIRECTORY = 5
C_BUILTIN = 6
PY_FROZEN = 7

_LOCK = threading.RLock()


def acquire_lock() -> None:
    """Acquire the global import lock."""

    _LOCK.acquire()


def release_lock() -> None:
    """Release the global import lock."""

    _LOCK.release()


def lock_held() -> bool:
    """Return ``True`` if the import lock is currently held."""

    return _LOCK.locked()


def _find_spec(name: str, path: Optional[Iterable[str]]) -> importlib.machinery.ModuleSpec:
    if path is None:
        spec = importlib.util.find_spec(name)
    else:
        spec = importlib.machinery.PathFinder.find_spec(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find module {name!r}")
    return spec


def find_module(name: str, path: Optional[Iterable[str]] = None) -> Tuple[Optional[object], str, Tuple[str, str, int]]:
    """Return a module loader triple mimicking :func:`imp.find_module`."""

    spec = _find_spec(name, path)

    if spec.submodule_search_locations:
        # Package: return the directory path and signal PKG_DIRECTORY.
        pathname = spec.submodule_search_locations[0]
        return None, pathname, ("", "", PKG_DIRECTORY)

    if not spec.has_location or spec.origin is None:
        raise ImportError(f"Loader for {name!r} lacks file location")

    pathname = spec.origin
    suffix = Path(pathname).suffix
    if suffix in importlib.machinery.SOURCE_SUFFIXES:
        file_obj = open(pathname, "r", encoding="utf-8")
        desc = (suffix, "r", PY_SOURCE)
    elif suffix in importlib.machinery.BYTECODE_SUFFIXES:
        file_obj = open(pathname, "rb")
        desc = (suffix, "rb", PY_COMPILED)
    elif suffix in importlib.machinery.EXTENSION_SUFFIXES:
        file_obj = None
        desc = (suffix, "", C_EXTENSION)
    else:
        file_obj = open(pathname, "rb")
        desc = (suffix, "rb", PY_SOURCE)

    return file_obj, pathname, desc


def load_module(name: str, file, pathname: str, description) -> ModuleType:
    """Load a module given the triple returned by :func:`find_module`."""

    if description and len(description) >= 3 and description[2] == PKG_DIRECTORY:
        spec = _find_spec(name, [pathname])
    else:
        spec = importlib.util.spec_from_file_location(name, pathname)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {name!r} from {pathname!r}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def new_module(name: str) -> ModuleType:
    """Create a new empty module instance."""

    module = ModuleType(name)
    module.__file__ = None
    module.__loader__ = None
    module.__package__ = name.rpartition('.')[0]
    return module
