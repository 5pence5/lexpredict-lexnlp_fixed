"""Tests for lexnlp/ml/model_io.py.

Covers the new skops-based model serialization helpers introduced in this PR:
- is_skops_path
- dump_model
- load_model (skops and legacy paths)
- _load_legacy (pickle / joblib dispatch)
- _load_skops (trusted type resolution)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from lexnlp.ml.model_io import (
    _LEGACY_SUFFIXES,
    CANONICAL_SUFFIX,
    DEFAULT_TRUSTED_ALLOWLIST,
    _load_legacy,
    _load_skops,
    dump_model,
    is_skops_path,
    load_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pickle(path: Path, obj: object) -> Path:
    """Write *obj* to *path* using stdlib pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


# ---------------------------------------------------------------------------
# is_skops_path
# ---------------------------------------------------------------------------


class TestIsSkopsPath:
    def test_true_for_skops_extension(self, tmp_path: Path) -> None:
        assert is_skops_path(Path("model.skops")) is True

    def test_true_for_uppercase_skops_extension(self, tmp_path: Path) -> None:
        assert is_skops_path(Path("model.SKOPS")) is True

    def test_false_for_pickle_extension(self) -> None:
        assert is_skops_path(Path("model.pickle")) is False

    def test_false_for_pkl_extension(self) -> None:
        assert is_skops_path(Path("model.pkl")) is False

    def test_false_for_cloudpickle_extension(self) -> None:
        assert is_skops_path(Path("model.cloudpickle")) is False

    def test_false_for_joblib_extension(self) -> None:
        assert is_skops_path(Path("model.joblib")) is False

    def test_false_for_no_extension(self) -> None:
        assert is_skops_path(Path("model")) is False

    def test_false_for_txt_extension(self) -> None:
        assert is_skops_path(Path("model.txt")) is False

    def test_accepts_path_string_coercion(self) -> None:
        # dump_model converts str→Path; test that is_skops_path itself handles Path objects.
        assert is_skops_path(Path("/some/deep/dir/artifact.skops")) is True


# ---------------------------------------------------------------------------
# dump_model
# ---------------------------------------------------------------------------


class TestDumpModel:
    """Tests for dump_model: suffix normalization, parent directory creation,
    return value, and round-trip fidelity."""

    def test_returns_path_with_skops_suffix(self, tmp_path: Path) -> None:
        obj = {"key": "value"}
        result = dump_model(obj, tmp_path / "model.skops")
        assert result.suffix == CANONICAL_SUFFIX

    def test_normalizes_non_skops_suffix_to_skops(self, tmp_path: Path) -> None:
        """If the caller passes a .pickle path, dump_model rewrites it to .skops."""
        result = dump_model({"x": 1}, tmp_path / "model.pickle")
        assert result.suffix == CANONICAL_SUFFIX
        assert result.stem == "model"

    def test_normalizes_pkl_suffix_to_skops(self, tmp_path: Path) -> None:
        result = dump_model({"x": 1}, tmp_path / "artifact.pkl")
        assert result.suffix == CANONICAL_SUFFIX

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "model.skops"
        dump_model({"y": 2}, deep_path)
        assert deep_path.parent.is_dir()

    def test_written_file_exists(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.skops"
        dump_model(42, dest)
        assert dest.exists()

    def test_roundtrip_dict(self, tmp_path: Path) -> None:
        obj = {"a": 1, "b": [2, 3]}
        path = dump_model(obj, tmp_path / "model.skops")
        loaded = load_model(path, trusted=True)
        assert loaded == obj

    def test_roundtrip_list(self, tmp_path: Path) -> None:
        obj = [1, "two", 3.0]
        path = dump_model(obj, tmp_path / "model.skops")
        loaded = load_model(path, trusted=True)
        assert loaded == obj

    def test_roundtrip_integer(self, tmp_path: Path) -> None:
        path = dump_model(42, tmp_path / "model.skops")
        loaded = load_model(path, trusted=True)
        assert loaded == 42

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """dump_model should coerce str paths to Path internally."""
        str_path = str(tmp_path / "model.skops")
        result = dump_model({"z": 0}, str_path)
        assert isinstance(result, Path)
        assert result.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "model.skops"
        dump_model({"v": 1}, dest)
        dump_model({"v": 99}, dest)
        loaded = load_model(dest, trusted=True)
        assert loaded == {"v": 99}

    def test_returns_normalized_path(self, tmp_path: Path) -> None:
        """Even when the input uses a non-skops suffix, the returned path has .skops."""
        result = dump_model({"q": 1}, tmp_path / "something.bin")
        assert result == tmp_path / "something.skops"


# ---------------------------------------------------------------------------
# _load_legacy
# ---------------------------------------------------------------------------


class TestLoadLegacy:
    """Tests for the _load_legacy private helper that dispatches on suffix."""

    def test_loads_pickle_file(self, tmp_path: Path) -> None:
        path = _write_pickle(tmp_path / "model.pickle", {"a": 1})
        assert _load_legacy(path) == {"a": 1}

    def test_loads_pkl_file(self, tmp_path: Path) -> None:
        path = _write_pickle(tmp_path / "model.pkl", [1, 2, 3])
        assert _load_legacy(path) == [1, 2, 3]

    def test_loads_joblib_file(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "model.joblib"
        joblib.dump({"jl": True}, path)
        assert _load_legacy(path) == {"jl": True}

    def test_loads_joblib_compressed_pickle_file(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "model.pickle"
        joblib.dump({"jl_pickle": True}, path, compress=3)
        assert _load_legacy(path) == {"jl_pickle": True}

    def test_loads_joblib_uncompressed_pickle_file(self, tmp_path: Path) -> None:
        """Regression: a ``.pickle`` written by ``joblib.dump(..., compress=0)``
        must still load via the joblib fallback even though it does not
        start with the zlib framing byte."""
        import joblib

        path = tmp_path / "model_uncompressed.pickle"
        joblib.dump({"jl_pickle_raw": True}, path, compress=0)
        assert _load_legacy(path) == {"jl_pickle_raw": True}

    def test_unknown_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """Unknown extensions must be rejected rather than blindly pickle-loaded."""
        path = tmp_path / "model.bin"
        _write_pickle(path, "hello")
        with pytest.raises(ValueError, match="Unsupported file suffix"):
            _load_legacy(path)

    def test_pickle_file_invalid_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pickle"
        path.write_bytes(b"not a pickle")
        with pytest.raises(pickle.UnpicklingError):
            _load_legacy(path)

    def test_loads_cloudpickle_file(self, tmp_path: Path) -> None:
        import cloudpickle

        path = tmp_path / "model.cloudpickle"
        with path.open("wb") as fh:
            cloudpickle.dump({"cp": 1}, fh)
        assert _load_legacy(path) == {"cp": 1}


# ---------------------------------------------------------------------------
# load_model - routing logic
# ---------------------------------------------------------------------------


class TestLoadModel:
    """Tests for load_model's routing logic (skops vs legacy vs unknown suffix)."""

    def test_loads_skops_file(self, tmp_path: Path) -> None:
        obj = {"model": "data"}
        path = dump_model(obj, tmp_path / "model.skops")
        loaded = load_model(path, trusted=True)
        assert loaded == obj

    def test_loads_skops_file_without_trusted_flag(self, tmp_path: Path) -> None:
        """Calling with trusted=False should still load a simple object."""
        path = dump_model({"safe": True}, tmp_path / "model.skops")
        loaded = load_model(path, trusted=False)
        assert loaded == {"safe": True}

    def test_loads_pickle_file(self, tmp_path: Path) -> None:
        path = _write_pickle(tmp_path / "model.pickle", {"legacy": True})
        loaded = load_model(path)
        assert loaded == {"legacy": True}

    def test_loads_pkl_file(self, tmp_path: Path) -> None:
        path = _write_pickle(tmp_path / "model.pkl", 99)
        loaded = load_model(path)
        assert loaded == 99

    def test_loads_joblib_file(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "model.joblib"
        joblib.dump({"jl": "data"}, path)
        loaded = load_model(path)
        assert loaded == {"jl": "data"}

    def test_no_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """A path with no suffix must be rejected (no silent pickle fallback)."""
        path = tmp_path / "modelfile"
        _write_pickle(path, "bare_pickle")
        with pytest.raises(ValueError, match="Unsupported model suffix"):
            load_model(path)

    def test_unknown_suffix_raises_value_error(self, tmp_path: Path) -> None:
        """An unrecognised suffix must be rejected to avoid widening
        unsafe deserialization surface."""
        path = tmp_path / "model.bin"
        _write_pickle(path, "fallback")
        with pytest.raises(ValueError, match="Unsupported model suffix"):
            load_model(path, trusted=True)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        obj = [10, 20]
        skops_path = dump_model(obj, tmp_path / "model.skops")
        loaded = load_model(str(skops_path), trusted=True)
        assert loaded == obj

    def test_skops_path_dispatches_to_skops_loader(self, tmp_path: Path) -> None:
        """Verify _load_skops is called (not _load_legacy) for .skops paths."""
        obj = {"check": "dispatch"}
        path = dump_model(obj, tmp_path / "model.skops")

        with patch("lexnlp.ml.model_io._load_legacy") as mock_legacy:
            result = load_model(path, trusted=True)
        mock_legacy.assert_not_called()
        assert result == obj

    def test_pickle_path_dispatches_to_legacy_loader(self, tmp_path: Path) -> None:
        """Verify _load_skops is NOT called for .pickle paths."""
        path = _write_pickle(tmp_path / "model.pickle", {"p": 1})

        with patch("lexnlp.ml.model_io._load_skops") as mock_skops:
            result = load_model(path)
        mock_skops.assert_not_called()
        assert result == {"p": 1}


# ---------------------------------------------------------------------------
# _load_skops - trusted type resolution
# ---------------------------------------------------------------------------


class TestLoadSkops:
    """Tests for the _load_skops helper that resolves the trusted type list."""

    def test_loads_simple_object_trusted(self, tmp_path: Path) -> None:
        path = dump_model({"simple": 42}, tmp_path / "model.skops")
        result = _load_skops(path, trusted=True)
        assert result == {"simple": 42}

    def test_loads_simple_object_untrusted(self, tmp_path: Path) -> None:
        """trusted=False should still work for basic built-in types."""
        path = dump_model([1, 2, 3], tmp_path / "model.skops")
        result = _load_skops(path, trusted=False)
        assert result == [1, 2, 3]

    def test_get_untrusted_types_called(self, tmp_path: Path) -> None:
        """get_untrusted_types must be called to enumerate the artifact's types."""
        path = dump_model({"x": 1}, tmp_path / "model.skops")
        with patch("lexnlp.ml.model_io.get_untrusted_types", return_value=[]) as mock_gut:
            _load_skops(path, trusted=True)
        mock_gut.assert_called_once_with(file=path)


# ---------------------------------------------------------------------------
# CANONICAL_SUFFIX / _LEGACY_SUFFIXES constants
# ---------------------------------------------------------------------------


class TestTrustedAllowlist:
    """Tests for the explicit allow-list-based trusted loading path."""

    def test_default_allowlist_is_non_empty(self) -> None:
        assert len(DEFAULT_TRUSTED_ALLOWLIST) > 0

    def test_default_allowlist_includes_common_sklearn_types(self) -> None:
        assert "sklearn.pipeline.Pipeline" in DEFAULT_TRUSTED_ALLOWLIST
        assert "numpy.ndarray" in DEFAULT_TRUSTED_ALLOWLIST

    def test_default_allowlist_is_frozenset(self) -> None:
        assert isinstance(DEFAULT_TRUSTED_ALLOWLIST, frozenset)

    def test_load_rejects_type_outside_allowlist(self, tmp_path: Path) -> None:
        """trusted=True without override rejects artifacts containing a
        type that is not in DEFAULT_TRUSTED_ALLOWLIST, raising
        ``ValueError`` whose message names the rejected type."""

        path = dump_model({"x": 1}, tmp_path / "m.skops")
        # Inject a fake untrusted type by patching get_untrusted_types.
        with patch(
            "lexnlp.ml.model_io.get_untrusted_types",
            return_value=["evil.Module.RemoteCodeExecution"],
        ):
            with pytest.raises(ValueError, match="trusted allow-list") as excinfo:
                _load_skops(path, trusted=True)
        assert "evil.Module.RemoteCodeExecution" in str(excinfo.value)

    def test_load_accepts_additional_allowed_type(self, tmp_path: Path) -> None:
        """Callers may extend the allow-list with extra type names so a
        previously rejected custom type is accepted."""

        path = dump_model({"x": 1}, tmp_path / "m.skops")
        captured: dict[str, list[str]] = {}

        # Real artifact contains no custom types, so to prove ``extra_trusted``
        # actually flows into the skops gate we (a) stub ``get_untrusted_types``
        # to return the allow-listed name and (b) intercept ``_skops_load`` to
        # capture the trusted list it was invoked with.
        def fake_skops_load(p, trusted):  # type: ignore[no-untyped-def]  # test stub: simple dict return, typing not needed
            captured["trusted"] = list(trusted)
            return {"x": 1}

        with (
            patch(
                "lexnlp.ml.model_io.get_untrusted_types",
                return_value=["my.Custom.Class"],
            ),
            patch("lexnlp.ml.model_io._skops_load", side_effect=fake_skops_load),
        ):
            result = _load_skops(path, trusted=True, extra_trusted=("my.Custom.Class",))
        assert result == {"x": 1}
        assert captured["trusted"] == ["my.Custom.Class"]

    def test_load_skips_get_untrusted_types_when_not_trusted(self, tmp_path: Path) -> None:
        """The fail-closed path must not pay the cost of scanning declared
        types — ``get_untrusted_types`` is only called when ``trusted=True``."""

        path = dump_model({"x": 1}, tmp_path / "m.skops")
        with patch("lexnlp.ml.model_io.get_untrusted_types") as mock_gut:
            _load_skops(path, trusted=False)
        mock_gut.assert_not_called()


class TestConstants:
    def test_canonical_suffix_is_skops(self) -> None:
        assert CANONICAL_SUFFIX == ".skops"

    def test_legacy_suffixes_includes_pickle(self) -> None:
        assert ".pickle" in _LEGACY_SUFFIXES

    def test_legacy_suffixes_includes_pkl(self) -> None:
        assert ".pkl" in _LEGACY_SUFFIXES

    def test_legacy_suffixes_includes_cloudpickle(self) -> None:
        assert ".cloudpickle" in _LEGACY_SUFFIXES

    def test_legacy_suffixes_includes_joblib(self) -> None:
        assert ".joblib" in _LEGACY_SUFFIXES

    def test_legacy_suffixes_is_frozenset(self) -> None:
        assert isinstance(_LEGACY_SUFFIXES, frozenset)


# ---------------------------------------------------------------------------
# Additional tests for PR changes to _load_legacy and _load_skops
# ---------------------------------------------------------------------------


class TestLoadLegacyAdditional:
    """Additional tests for the PR-refactored _load_legacy helper."""

    def test_double_failure_reraises_original_pickle_exception(self, tmp_path: Path) -> None:
        """When both the raw pickle path AND the joblib fallback fail, the
        original pickle exception (not the joblib one) must propagate.

        The PR removed _looks_like_joblib_pickle and always retries via joblib.
        When joblib also fails, the original pickle exception is re-raised.
        """
        path = tmp_path / "corrupt.pickle"
        path.write_bytes(b"not a pickle and not a joblib file")
        with pytest.raises(pickle.UnpicklingError):
            _load_legacy(path)

    def test_joblib_compress_0_round_trips_via_pickle_extension(self, tmp_path: Path) -> None:
        """Regression: joblib.dump(obj, path, compress=0) produces a file that
        does NOT start with the zlib framing byte.  The old guard (_looks_like_
        joblib_pickle) would have blocked the fallback; the PR always retries."""
        import joblib

        path = tmp_path / "raw_joblib.pickle"
        obj = {"compress_level": 0, "data": list(range(10))}
        joblib.dump(obj, path, compress=0)
        result = _load_legacy(path)
        assert result == obj

    def test_loads_pkl_extension_via_joblib_fallback(self, tmp_path: Path) -> None:
        """A .pkl file written by joblib must load via the joblib fallback."""
        import joblib

        path = tmp_path / "model.pkl"
        joblib.dump({"via": "joblib"}, path, compress=3)
        result = _load_legacy(path)
        assert result == {"via": "joblib"}

    def test_looks_like_joblib_pickle_function_removed(self) -> None:
        """The PR removed _looks_like_joblib_pickle — it must not exist."""
        import lexnlp.ml.model_io as model_io_module

        assert not hasattr(model_io_module, "_looks_like_joblib_pickle"), (
            "_looks_like_joblib_pickle was removed by the PR and must not be present"
        )


class TestLoadSkopsAdditional:
    """Additional tests for the PR-refactored _load_skops helper."""

    def test_trusted_false_does_not_call_get_untrusted_types(self, tmp_path: Path) -> None:
        """The fail-closed (trusted=False) path must skip the type scan entirely."""
        path = dump_model({"ok": True}, tmp_path / "m.skops")
        with patch("lexnlp.ml.model_io.get_untrusted_types") as mock_gut:
            result = _load_skops(path, trusted=False)
        mock_gut.assert_not_called()
        assert result == {"ok": True}

    def test_trusted_true_calls_get_untrusted_types_exactly_once(self, tmp_path: Path) -> None:
        """When trusted=True the artifact is scanned exactly once."""
        path = dump_model({"ok": True}, tmp_path / "m.skops")
        with patch("lexnlp.ml.model_io.get_untrusted_types", return_value=[]) as mock_gut:
            _load_skops(path, trusted=True)
        mock_gut.assert_called_once_with(file=path)

    def test_rejection_message_lists_all_rejected_types(self, tmp_path: Path) -> None:
        """When multiple types are outside the allowlist, all must appear
        in the ValueError message."""
        path = dump_model({"ok": 1}, tmp_path / "m.skops")
        bad_types = ["evil.A", "evil.B", "evil.C"]
        with patch(
            "lexnlp.ml.model_io.get_untrusted_types",
            return_value=bad_types,
        ):
            with pytest.raises(ValueError) as excinfo:
                _load_skops(path, trusted=True)
        msg = str(excinfo.value)
        for t in bad_types:
            assert t in msg, f"Type '{t}' missing from rejection message: {msg}"

    def test_extra_trusted_extends_allow_list(self, tmp_path: Path) -> None:
        """extra_trusted types not in DEFAULT_TRUSTED_ALLOWLIST must be accepted
        AND forwarded to ``skops.io.load`` so the artifact actually gets loaded
        with the augmented trusted set.
        """
        path = dump_model({"ok": 1}, tmp_path / "m.skops")
        custom_type = "my.domain.SpecialEncoder"
        assert custom_type not in DEFAULT_TRUSTED_ALLOWLIST

        captured: dict[str, list[str]] = {}

        def fake_skops_load(p, trusted):  # type: ignore[no-untyped-def]  # test stub
            captured["trusted"] = list(trusted)
            return {"ok": 1}

        with (
            patch(
                "lexnlp.ml.model_io.get_untrusted_types",
                return_value=[custom_type],
            ),
            patch("lexnlp.ml.model_io._skops_load", side_effect=fake_skops_load),
        ):
            result = _load_skops(path, trusted=True, extra_trusted=(custom_type,))
        assert result == {"ok": 1}
        # The custom type must have actually flowed into skops.io.load — not
        # just survived the allow-list check.
        assert custom_type in captured["trusted"], f"extra_trusted type was filtered before reaching skops: {captured}"

    def test_type_in_default_allowlist_not_rejected(self, tmp_path: Path) -> None:
        """A type already in DEFAULT_TRUSTED_ALLOWLIST must not be rejected."""
        path = dump_model({"ok": 1}, tmp_path / "m.skops")
        known_safe = "numpy.ndarray"
        assert known_safe in DEFAULT_TRUSTED_ALLOWLIST

        def fake_skops_load(p, trusted):  # type: ignore[no-untyped-def]  # test stub: simple dict return, typing not needed
            return {"ok": 1}

        with (
            patch(
                "lexnlp.ml.model_io.get_untrusted_types",
                return_value=[known_safe],
            ),
            patch("lexnlp.ml.model_io._skops_load", side_effect=fake_skops_load),
        ):
            result = _load_skops(path, trusted=True)
        assert result == {"ok": 1}
