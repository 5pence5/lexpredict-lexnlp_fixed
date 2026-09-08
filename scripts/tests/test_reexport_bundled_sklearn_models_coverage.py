"""Coverage tests for scripts/reexport_bundled_sklearn_models.py."""

from __future__ import annotations

import io
import pickle
import sys
import warnings
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import reexport_bundled_sklearn_models as script_mod


def _tiny_estimator() -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    return clf


def _make_layered_zip(term_obj: object, definition_obj: object) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", compression=ZIP_STORED) as archive:
        archive.writestr("term.pickle", pickle.dumps(term_obj, protocol=pickle.HIGHEST_PROTOCOL))
        archive.writestr("definition.pickle", pickle.dumps(definition_obj, protocol=pickle.HIGHEST_PROTOCOL))
    return buf.getvalue()


def _write_layered(path: Path, term_obj: object | None = None, definition_obj: object | None = None) -> Path:
    path.write_bytes(_make_layered_zip(term_obj or {"term": 1}, definition_obj or {"definition": 2}))
    return path


class TestParseArgs:
    def test_defaults(self) -> None:
        args = script_mod.parse_args([])
        assert args.paths is None
        assert args.compress == 3
        assert args.format == "skops"
        assert args.remove_legacy is False

    def test_explicit_flags(self, tmp_path: Path) -> None:
        model = tmp_path / "model.pickle"
        args = script_mod.parse_args(
            [
                "--paths",
                str(model),
                "--compress",
                "0",
                "--format",
                "pickle",
                "--remove-legacy",
            ]
        )
        assert args.paths == [str(model)]
        assert args.compress == 0
        assert args.format == "pickle"
        assert args.remove_legacy is True

    def test_invalid_format_exits(self) -> None:
        with pytest.raises(SystemExit):
            script_mod.parse_args(["--format", "onnx"])


class TestIterPaths:
    def test_explicit_paths_win(self, tmp_path: Path) -> None:
        first = tmp_path / "a.pickle"
        second = tmp_path / "b.pickle"
        args = Namespace(paths=[str(first), str(second)])
        assert list(script_mod.iter_paths(args)) == [first, second]

    def test_empty_paths_falls_back_to_bundled(self) -> None:
        args = Namespace(paths=[])
        assert list(script_mod.iter_paths(args)) == list(script_mod.BUNDLED_MODEL_PATHS)

    def test_missing_paths_falls_back_to_bundled(self) -> None:
        args = Namespace(paths=None)
        bundled = list(script_mod.iter_paths(args))
        assert bundled == list(script_mod.BUNDLED_MODEL_PATHS)
        assert any(path.name == "addresses_clf.pickle" for path in bundled)


class TestLoadAddressesClf:
    def test_raw_pickle_loads_through_renamed_load(self, tmp_path: Path) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"kind": "addresses", "n": 3}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        assert script_mod._load_addresses_clf(path) == payload

    def test_joblib_payload_falls_back_after_unpickling_error(self, tmp_path: Path) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"kind": "joblib-addresses"}
        joblib.dump(payload, path)
        assert script_mod._load_addresses_clf(path) == payload

    def test_value_error_falls_back_to_joblib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"kind": "value-error"}
        joblib.dump(payload, path)

        def boom(_fh):
            raise ValueError("version mismatch")

        monkeypatch.setattr("lexnlp.utils.unpickler.renamed_load", boom)
        assert script_mod._load_addresses_clf(path) == payload

    def test_key_error_falls_back_to_joblib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"kind": "key-error"}
        joblib.dump(payload, path)

        def boom(_fh):
            raise KeyError("missing class")

        monkeypatch.setattr("lexnlp.utils.unpickler.renamed_load", boom)
        assert script_mod._load_addresses_clf(path) == payload


class TestLoadForReexport:
    def test_addresses_name_uses_addresses_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"via": "addresses"}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        assert script_mod._load_for_reexport(path) == payload

    def test_layered_gzip_name_uses_layered_loader(self, tmp_path: Path) -> None:
        path = _write_layered(tmp_path / "definition_model_layered.pickle.gzip")
        loaded = script_mod._load_for_reexport(path)
        assert loaded["term.pickle"] == {"term": 1}
        assert loaded["definition.pickle"] == {"definition": 2}

    def test_other_pickle_uses_load_model(self, tmp_path: Path) -> None:
        path = tmp_path / "section_segmenter.pickle"
        clf = _tiny_estimator()
        joblib.dump(clf, path)
        loaded = script_mod._load_for_reexport(path)
        assert isinstance(loaded, LogisticRegression)
        assert loaded.predict(np.array([[1.0]]))[0] == clf.predict(np.array([[1.0]]))[0]


class TestLoadLayeredDefinitionModels:
    def test_reads_both_pickle_members(self, tmp_path: Path) -> None:
        path = _write_layered(
            tmp_path / "definition_model_layered.pickle.gzip",
            term_obj=["term"],
            definition_obj=["definition"],
        )
        payload = script_mod.load_layered_definition_models(path)
        assert payload == {"term.pickle": ["term"], "definition.pickle": ["definition"]}


class TestReexportLayeredSkops:
    def test_writes_skops_zip_sibling(self, tmp_path: Path) -> None:
        path = _write_layered(tmp_path / "definition_model_layered.pickle.gzip")
        target = script_mod.reexport_layered_definition_models_skops(path)
        assert target == tmp_path / "definition_model_layered.skops.zip"
        assert target.exists()
        with ZipFile(target) as archive:
            assert set(archive.namelist()) == {"term.skops", "definition.skops"}
            assert archive.read("term.skops")
            assert archive.read("definition.skops")

    def test_removes_stale_part_file(self, tmp_path: Path) -> None:
        path = _write_layered(tmp_path / "definition_model_layered.pickle.gzip")
        target = tmp_path / "definition_model_layered.skops.zip"
        stale = target.with_name(target.name + ".part")
        stale.write_bytes(b"stale")
        written = script_mod.reexport_layered_definition_models_skops(path)
        assert written == target
        assert not stale.exists()

    def test_cleans_part_file_when_dump_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_layered(tmp_path / "definition_model_layered.pickle.gzip")
        target = tmp_path / "definition_model_layered.skops.zip"
        part = target.with_name(target.name + ".part")

        def boom(_obj, _buffer):
            raise RuntimeError("skops dump failed")

        monkeypatch.setattr("skops.io.dump", boom)
        with pytest.raises(RuntimeError, match="skops dump failed"):
            script_mod.reexport_layered_definition_models_skops(path)
        assert not part.exists()
        assert not target.exists()


class TestLegacyWarningCount:
    def test_counts_matching_warnings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "model.pickle"

        def noisy(_path: Path) -> dict[str, str]:
            warnings.warn("Trying to unpickle estimator LogisticRegression from version 1.2.2")
            warnings.warn("unrelated warning")
            warnings.warn("Trying to unpickle estimator Vectorizer from version 1.2.2")
            return {"ok": "yes"}

        monkeypatch.setattr(script_mod, "_load_for_reexport", noisy)
        assert script_mod.legacy_warning_count_for_load(path) == 2

    def test_load_failure_counts_as_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "broken.pickle"

        def boom(_path: Path) -> None:
            raise RuntimeError("cannot load")

        monkeypatch.setattr(script_mod, "_load_for_reexport", boom)
        assert script_mod.legacy_warning_count_for_load(path) == 0


class TestReexportSinglePickle:
    def test_addresses_clf_is_plain_pickle(self, tmp_path: Path) -> None:
        path = tmp_path / "addresses_clf.pickle"
        payload = {"addr": True}
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        written = script_mod._reexport_single_pickle(path, compress=0)
        assert written == path
        with path.open("rb") as handle:
            assert pickle.load(handle) == payload

    def test_other_models_are_joblib_dumped(self, tmp_path: Path) -> None:
        path = tmp_path / "date_model.pickle"
        clf = _tiny_estimator()
        joblib.dump(clf, path, compress=0)
        written = script_mod._reexport_single_pickle(path, compress=0)
        assert written == path
        loaded = joblib.load(path)
        assert isinstance(loaded, LogisticRegression)
        assert loaded.predict(np.array([[0.0]]))[0] == 0


class TestSanitizePandasIndices:
    def test_rewrites_index_in_dict_and_list(self) -> None:
        payload = {
            "columns": pd.Index(["a", "b"]),
            "nested": [pd.Index(["c"]), {"inner": pd.Index(["d"])}],
        }
        result = script_mod._sanitize_pandas_indices(payload)
        assert result is payload
        assert result["columns"] == ["a", "b"]
        assert result["nested"][0] == ["c"]
        assert result["nested"][1]["inner"] == ["d"]

    def test_rewrites_object_attributes(self) -> None:
        class Holder:
            def __init__(self) -> None:
                self.columns = pd.Index(["year", "month"])
                self.child = {"idx": pd.Index(["z"])}

        holder = Holder()
        result = script_mod._sanitize_pandas_indices(holder)
        assert result is holder
        assert holder.columns == ["year", "month"]
        assert holder.child["idx"] == ["z"]

    def test_tuple_recurses_into_mutable_members(self) -> None:
        """A tuple cannot be rewritten, but a dict inside it can be."""
        inner = {"idx": pd.Index(["keep"])}

        result = script_mod._sanitize_pandas_indices((inner, "literal"))

        assert result == (inner, "literal")
        assert inner["idx"] == ["keep"]
        assert not isinstance(inner["idx"], pd.Index)

    def test_set_recurses_into_mutable_members(self) -> None:
        """A set's members must be hashable, so recurse through an object."""

        class Holder:
            def __init__(self) -> None:
                self.payload = {"idx": pd.Index(["kept"])}

        holder = Holder()

        result = script_mod._sanitize_pandas_indices({holder})

        assert result == {holder}
        assert holder.payload["idx"] == ["kept"]
        assert not isinstance(holder.payload["idx"], pd.Index)

    def test_set_of_objects_is_sanitized(self) -> None:
        class Node:
            def __init__(self) -> None:
                self.columns = pd.Index(["s"])

        node = Node()
        result = script_mod._sanitize_pandas_indices({node})
        assert result == {node}
        assert node.columns == ["s"]

    def test_cycle_does_not_recurse_forever(self) -> None:
        cyclic: dict[str, object] = {}
        cyclic["self"] = cyclic
        cyclic["columns"] = pd.Index(["loop"])
        result = script_mod._sanitize_pandas_indices(cyclic)
        assert result["columns"] == ["loop"]
        assert result["self"] is cyclic

    def test_types_and_scalars_are_returned(self) -> None:
        assert script_mod._sanitize_pandas_indices(int) is int
        assert script_mod._sanitize_pandas_indices(7) == 7


class TestReexportSingleSkops:
    def test_writes_skops_sibling_and_sanitizes(self, tmp_path: Path) -> None:
        path = tmp_path / "date_model.pickle"
        clf = _tiny_estimator()
        clf.columns = pd.Index(["feat"])  # type: ignore[attr-defined]
        joblib.dump(clf, path)
        target = script_mod._reexport_single_skops(path)
        assert target == path.with_suffix(".skops")
        assert target.exists()
        assert path.exists()
        assert clf.columns == ["feat"]


class TestMain:
    def test_missing_path_returns_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        missing = tmp_path / "nope.pickle"
        rc = script_mod.main(["--paths", str(missing), "--format", "pickle"])
        captured = capsys.readouterr()
        assert rc == 1
        assert f"reexport: ERROR missing: {missing}" in captured.out

    def test_pickle_format_reexports_estimator(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "section_segmenter.pickle"
        joblib.dump(_tiny_estimator(), path)
        rc = script_mod.main(["--paths", str(path), "--format", "pickle", "--compress", "0"])
        captured = capsys.readouterr()
        assert rc == 0
        assert f"reexport: {path} legacy_warnings" in captured.out
        loaded = joblib.load(path)
        assert isinstance(loaded, LogisticRegression)

    def test_pickle_format_reexports_layered_gzip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_layered(
            tmp_path / "definition_model_layered.pickle.gzip",
            term_obj={"term": "v1"},
            definition_obj={"definition": "v1"},
        )
        rc = script_mod.main(["--paths", str(path), "--format", "pickle"])
        captured = capsys.readouterr()
        assert rc == 0
        assert f"reexport: {path} legacy_warnings" in captured.out
        reloaded = script_mod.load_layered_definition_models(path)
        assert reloaded["term.pickle"] == {"term": "v1"}

    def test_skops_format_writes_sibling(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "title_locator.pickle"
        joblib.dump(_tiny_estimator(), path)
        rc = script_mod.main(["--paths", str(path), "--format", "skops"])
        captured = capsys.readouterr()
        assert rc == 0
        target = path.with_suffix(".skops")
        assert target.exists()
        assert f"reexport: {path} -> {target}" in captured.out
        assert "legacy_warnings before=" in captured.out
        assert "after=0" in captured.out

    def test_skops_format_layered_gzip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_layered(tmp_path / "definition_model_layered.pickle.gzip")
        rc = script_mod.main(["--paths", str(path), "--format", "skops"])
        captured = capsys.readouterr()
        assert rc == 0
        target = tmp_path / "definition_model_layered.skops.zip"
        assert target.exists()
        assert f"reexport: {path} -> {target}" in captured.out

    def test_remove_legacy_deletes_pickle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "page_segmenter.pickle"
        joblib.dump(_tiny_estimator(), path)
        rc = script_mod.main(["--paths", str(path), "--format", "skops", "--remove-legacy"])
        captured = capsys.readouterr()
        assert rc == 0
        assert not path.exists()
        assert path.with_suffix(".skops").exists()
        assert "removed-legacy=yes" in captured.out

    def test_remove_legacy_skipped_when_target_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "page_segmenter.pickle"
        joblib.dump(_tiny_estimator(), path)
        missing_target = tmp_path / "missing.skops"

        def fake_skops(_path: Path) -> Path:
            return missing_target

        monkeypatch.setattr(script_mod, "_reexport_single_skops", fake_skops)
        rc = script_mod.main(["--paths", str(path), "--format", "skops", "--remove-legacy"])
        captured = capsys.readouterr()
        assert rc == 0
        assert path.exists()
        assert "removed-legacy=SKIPPED (target missing)" in captured.out

    def test_mixed_missing_and_valid_still_errors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        valid = tmp_path / "ok.pickle"
        joblib.dump(_tiny_estimator(), valid)
        missing = tmp_path / "gone.pickle"
        rc = script_mod.main(["--paths", str(missing), str(valid), "--format", "pickle"])
        captured = capsys.readouterr()
        assert rc == 1
        assert f"ERROR missing: {missing}" in captured.out
        assert f"reexport: {valid} legacy_warnings" in captured.out
