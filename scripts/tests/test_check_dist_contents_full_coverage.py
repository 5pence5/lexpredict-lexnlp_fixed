"""Full-statement-coverage tests for ci.check_dist_contents."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

import ci.check_dist_contents as mod
from ci.check_dist_contents import (
    MIN_NONEMPTY_LINE_COUNTS,
    find_duplicate_members,
    find_duplicate_resources,
    find_resource_failures,
    find_violations,
    installed_resources,
    is_forbidden_wheel_member,
    is_runtime_resource,
    iter_tar_names,
    iter_zip_names,
    main,
    nonempty_line_count,
    normalise_package_name,
    parse_args,
    read_directory_resources,
    read_tar_resources,
    read_zip_resources,
    source_resources,
    validate_artifacts,
    validate_installed,
    validate_source_resources,
)


def _good_resources() -> dict[str, bytes]:
    resources = {"lexnlp/data/a.json": b'{"ok": true}\n'}
    for name, minimum in MIN_NONEMPTY_LINE_COUNTS.items():
        resources[name] = b"entry\n" * minimum
    return resources


def _write_source_root(root: Path, resources: dict[str, bytes]) -> Path:
    for name, payload in resources.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


def _write_package_dir(root: Path, resources: dict[str, bytes]) -> Path:
    package_dir = root / "lexnlp"
    for name, payload in resources.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return package_dir


def _write_wheel(
    path: Path,
    members: dict[str, bytes],
    extra_writes: list[tuple[str, bytes]] | None = None,
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        for name, payload in extra_writes or []:
            archive.writestr(name, payload)
    return path


def _write_sdist(path: Path, members: dict[str, bytes], dirs: list[str] | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for dirname in dirs or []:
            info = tarfile.TarInfo(dirname)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _prefixed(members: dict[str, bytes], prefix: str = "lexnlp-2.3.0") -> dict[str, bytes]:
    return {f"{prefix}/{name}": payload for name, payload in members.items()}


def _happy_tree(tmp_path: Path) -> tuple[Path, Path]:
    resources = _good_resources()
    src_root = _write_source_root(tmp_path / "src", resources)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_members = dict(resources)
    wheel_members["lexnlp-2.3.0.dist-info/METADATA"] = b"Metadata-Version: 2.1\nName: lexnlp\n"
    _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", wheel_members)
    sdist_members = _prefixed(resources)
    sdist_members["lexnlp-2.3.0/PKG-INFO"] = b"Metadata-Version: 2.1\n"
    _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", sdist_members)
    return dist_dir, src_root


class TestParseArgs:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.dist_dir == Path("dist")
        assert args.installed is False
        assert args.source_root == Path(mod.__file__).resolve().parents[1]

    def test_explicit_values(self, tmp_path: Path) -> None:
        args = parse_args([str(tmp_path), "--source-root", str(tmp_path), "--installed"])
        assert args.dist_dir == tmp_path
        assert args.source_root == tmp_path
        assert args.installed is True


class TestRuntimeAndNames:
    def test_runtime_suffixes(self) -> None:
        assert is_runtime_resource("lexnlp/data/a.json")
        assert is_runtime_resource("lexnlp/data/a.csv")
        assert is_runtime_resource("lexnlp/data/a.pickle")
        assert is_runtime_resource("lexnlp/data/a.pickle.gzip")
        assert is_runtime_resource("lexnlp/data/a.skops")
        assert is_runtime_resource("lexnlp/data/a.skops.zip")
        assert is_runtime_resource("lexnlp/data/a.txt")
        assert is_runtime_resource("lexnlp/data/a.xml")
        assert not is_runtime_resource("lexnlp/mod.py")

    def test_normalise_variants(self) -> None:
        assert normalise_package_name("lexnlp/data/a.json") == "lexnlp/data/a.json"
        assert normalise_package_name("lexnlp-2.3.0/lexnlp/data/a.json") == "lexnlp/data/a.json"
        assert normalise_package_name("lexnlp\\data\\a.json") == "lexnlp/data/a.json"
        assert normalise_package_name("other/lexnlp/data/a.json") is None
        assert normalise_package_name("other/data/a.json") is None
        assert normalise_package_name("lexnlp/mod.py") is None

    def test_forbidden_wheel_members(self) -> None:
        assert is_forbidden_wheel_member("scripts/evil.py")
        assert is_forbidden_wheel_member("ci/check.py")
        assert is_forbidden_wheel_member("documentation/index.rst")
        assert is_forbidden_wheel_member("libs/stanford_nlp/model.bin")
        assert is_forbidden_wheel_member("notebooks/demo.ipynb")
        assert is_forbidden_wheel_member("lexnlp/tests/test_x.py")
        assert is_forbidden_wheel_member("lexnlp/data/tests/fixture.json")
        assert is_forbidden_wheel_member("scripts\\evil.py")
        assert not is_forbidden_wheel_member("lexnlp/data/a.json")
        assert not is_forbidden_wheel_member("lexnlp-2.3.0.dist-info/METADATA")


class TestArchiveIteration:
    def test_tar_lists_only_files(self, tmp_path: Path) -> None:
        path = tmp_path / "a.tar"
        with tarfile.open(path, "w") as archive:
            dir_info = tarfile.TarInfo("mydir")
            dir_info.type = tarfile.DIRTYPE
            archive.addfile(dir_info)
            payload = b"hello"
            file_info = tarfile.TarInfo("mydir/a.json")
            file_info.size = len(payload)
            archive.addfile(file_info, io.BytesIO(payload))
        assert list(iter_tar_names(path)) == ["mydir/a.json"]

    def test_zip_lists_only_files(self, tmp_path: Path) -> None:
        path = tmp_path / "a.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mydir/", b"")
            archive.writestr("mydir/a.json", b"{}")
        assert list(iter_zip_names(path)) == ["mydir/a.json"]


class TestReadTarResources:
    def test_reads_runtime_resources_only(self, tmp_path: Path) -> None:
        path = tmp_path / "pkg.tar.gz"
        payload_a = b'{"a": 1}'
        payload_b = b'{"b": 2}'
        members = {
            "lexnlp-2.3.0/lexnlp/data/a.json": payload_a,
            "lexnlp/data/b.json": payload_b,
            "lexnlp/mod.py": b"print(1)",
            "other/data/c.json": b"{}",
        }
        _write_sdist(path, members, dirs=["lexnlp-2.3.0/lexnlp/data"])
        assert read_tar_resources(path) == {
            "lexnlp/data/a.json": payload_a,
            "lexnlp/data/b.json": payload_b,
        }

    def test_unreadable_member_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        archive_path = tmp_path / "a.tar"
        archive_path.write_bytes(b"not-a-tar")
        member = types.SimpleNamespace(name="lexnlp/data/a.json", isfile=lambda: True)

        class _FakeArchive:
            def __enter__(self) -> _FakeArchive:
                return self

            def __exit__(self, *exc_info: object) -> bool:
                return False

            def getmembers(self) -> list[object]:
                return [member]

            def extractfile(self, _member: object) -> None:
                return None

        monkeypatch.setattr(tarfile, "open", lambda *args: _FakeArchive())
        with pytest.raises(ValueError, match="could not read"):
            read_tar_resources(archive_path)


class TestReadZipResources:
    def test_reads_runtime_resources_only(self, tmp_path: Path) -> None:
        path = tmp_path / "pkg.whl"
        payload_a = b'{"a": 1}'
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("emptydir/", b"")
            archive.writestr("lexnlp/data/a.json", payload_a)
            archive.writestr("lexnlp/mod.py", b"print(1)")
            archive.writestr("other/data/c.json", b"{}")
        assert read_zip_resources(path) == {"lexnlp/data/a.json": payload_a}


class TestReadDirectoryResources:
    def test_reads_runtime_files_only(self, tmp_path: Path) -> None:
        package_dir = tmp_path / "root" / "lexnlp"
        (package_dir / "sub").mkdir(parents=True)
        (package_dir / "data.json").write_bytes(b"{}")
        (package_dir / "mod.py").write_bytes(b"x = 1")
        (package_dir / "sub" / "nested.txt").write_bytes(b"hi")
        assert read_directory_resources(package_dir) == {
            "lexnlp/data.json": b"{}",
            "lexnlp/sub/nested.txt": b"hi",
        }


class TestSourceResources:
    def test_missing_package_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="missing source package directory"):
            source_resources(tmp_path / "empty-root")

    def test_no_resources_raises(self, tmp_path: Path) -> None:
        package_dir = tmp_path / "lexnlp"
        package_dir.mkdir()
        (package_dir / "mod.py").write_bytes(b"x = 1")
        with pytest.raises(ValueError, match="no runtime resources found"):
            source_resources(tmp_path)

    def test_happy_path(self, tmp_path: Path) -> None:
        package_dir = tmp_path / "lexnlp"
        package_dir.mkdir()
        (package_dir / "data.json").write_bytes(b"{}")
        assert source_resources(tmp_path) == {"lexnlp/data.json": b"{}"}


class TestInstalledResources:
    def test_missing_spec_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ValueError, match="not installed"):
            installed_resources()

    def test_spec_without_locations_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: types.SimpleNamespace(submodule_search_locations=[]),
        )
        with pytest.raises(ValueError, match="not installed"):
            installed_resources()

    def test_happy_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        package_dir = _write_package_dir(tmp_path / "inst", {"lexnlp/data/a.json": b"{}"})
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: types.SimpleNamespace(submodule_search_locations=[str(package_dir)]),
        )
        found_dir, resources = installed_resources()
        assert found_dir == package_dir.resolve()
        assert resources == {"lexnlp/data/a.json": b"{}"}


class TestFindViolations:
    def test_all_violation_kinds(self) -> None:
        names = [
            "lexnlp-2.3.0/documentation/docs/build/html/index.html",
            "some/dir/Pipfile",
            "some/dir/nested.pyo",
            "lexnlp/data/a.json",
        ]
        assert find_violations(names) == [
            "lexnlp-2.3.0/documentation/docs/build/html/index.html",
            "some/dir/Pipfile",
            "some/dir/nested.pyo",
        ]

    def test_backslash_and_pyc_forms(self) -> None:
        assert find_violations(["some\\dir\\python-requirements.txt"]) == ["some/dir/python-requirements.txt"]
        assert find_violations(["pkg/mod.pyc"]) == ["pkg/mod.pyc"]
        assert find_violations(["libs/stanford_nlp/model.bin"]) == ["libs/stanford_nlp/model.bin"]
        assert find_violations(["scripts/__pycache__/x.py"]) == ["scripts/__pycache__/x.py"]
        assert find_violations(["lexnlp/data/a.json"]) == []


class TestDuplicates:
    def test_duplicate_members(self) -> None:
        assert find_duplicate_members(["a.json", "b.json", "a.json", "a.json", "b.json"]) == ["a.json", "b.json"]
        assert find_duplicate_members(["a.json", "b.json"]) == []

    def test_duplicate_resources(self) -> None:
        names = [
            "lexnlp/data/a.json",
            "lexnlp-2.3.0/lexnlp/data/a.json",
            "lexnlp/mod.py",
            "other/data/c.json",
        ]
        assert find_duplicate_resources(names) == ["lexnlp/data/a.json"]
        assert find_duplicate_resources(["lexnlp/data/a.json", "lexnlp/mod.py"]) == []


class TestNonemptyLineCount:
    def test_counts_only_non_blank_lines(self) -> None:
        assert nonempty_line_count(b"a\n\n  \nb\n\t\n") == 2
        assert nonempty_line_count(b"") == 0


class TestFindResourceFailures:
    def test_missing_unexpected_and_differing(self) -> None:
        expected = {
            "lexnlp/data/keep.json": b"same",
            "lexnlp/data/gone.json": b"gone",
            "lexnlp/data/changed.json": b"old",
        }
        actual = {
            "lexnlp/data/keep.json": b"same",
            "lexnlp/data/changed.json": b"new",
            "lexnlp/data/extra.json": b"extra",
        }
        assert find_resource_failures(expected, actual, "wheel") == [
            "wheel: missing runtime resource: lexnlp/data/gone.json",
            "wheel: unexpected runtime resource: lexnlp/data/extra.json",
            "wheel: resource differs from source: lexnlp/data/changed.json",
        ]

    def test_short_language_dictionary_is_reported(self) -> None:
        name = next(iter(MIN_NONEMPTY_LINE_COUNTS))
        minimum = MIN_NONEMPTY_LINE_COUNTS[name]
        expected = {name: b"entry\n" * minimum}
        actual = {name: b"entry\n" * (minimum - 1)}
        assert find_resource_failures(expected, actual, "sdist") == [
            f"sdist: resource differs from source: {name}",
            f"sdist: {name} has {minimum - 1} non-empty lines; expected at least {minimum}",
        ]

    def test_full_language_dictionaries_pass(self) -> None:
        resources = {name: b"entry\n" * minimum for name, minimum in MIN_NONEMPTY_LINE_COUNTS.items()}
        assert find_resource_failures(resources, dict(resources), "sdist") == []


class TestValidateSourceResources:
    def test_missing_dictionaries(self) -> None:
        failures = validate_source_resources({})
        assert len(failures) == len(MIN_NONEMPTY_LINE_COUNTS)
        assert all("missing required language dictionary" in failure for failure in failures)

    def test_short_dictionary(self) -> None:
        name = next(iter(MIN_NONEMPTY_LINE_COUNTS))
        minimum = MIN_NONEMPTY_LINE_COUNTS[name]
        resources = {key: b"entry\n" * count for key, count in MIN_NONEMPTY_LINE_COUNTS.items()}
        resources[name] = b"entry\n" * (minimum - 1)
        assert validate_source_resources(resources) == [
            f"source: {name} has {minimum - 1} non-empty lines; expected at least {minimum}"
        ]

    def test_good_dictionaries(self) -> None:
        resources = {name: b"entry\n" * minimum for name, minimum in MIN_NONEMPTY_LINE_COUNTS.items()}
        assert validate_source_resources(resources) == []


class TestValidateInstalled:
    def test_source_error_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert validate_installed(tmp_path / "missing-root") == 1
        assert "dist-check:" in capsys.readouterr().err

    def test_differing_installed_package_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        changed = dict(resources)
        changed["lexnlp/data/a.json"] = b'{"ok": false}\n'
        package_dir = _write_package_dir(tmp_path / "inst", changed)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: types.SimpleNamespace(submodule_search_locations=[str(package_dir)]),
        )
        assert validate_installed(src_root) == 1
        assert "installed-package resource parity failed" in capsys.readouterr().err

    def test_matching_installed_package_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        package_dir = _write_package_dir(tmp_path / "inst", resources)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: types.SimpleNamespace(submodule_search_locations=[str(package_dir)]),
        )
        assert validate_installed(src_root) == 0
        assert "resource parity OK" in capsys.readouterr().out


class TestValidateArtifacts:
    def test_missing_dist_dir_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        src_root = _write_source_root(tmp_path / "src", _good_resources())
        assert validate_artifacts(tmp_path / "no-dist", src_root) == 1
        assert "missing dist directory" in capsys.readouterr().err

    def test_empty_dist_dir_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        src_root = _write_source_root(tmp_path / "src", _good_resources())
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "expected at least one wheel" in capsys.readouterr().err

    def test_bad_source_root_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dist_dir, _ = _happy_tree(tmp_path)
        assert validate_artifacts(dist_dir, tmp_path / "no-src") == 1
        assert "dist-check:" in capsys.readouterr().err

    def test_matching_artifacts_return_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dist_dir, src_root = _happy_tree(tmp_path)
        assert validate_artifacts(dist_dir, src_root) == 0
        assert "resource parity OK" in capsys.readouterr().out

    def test_forbidden_wheel_member_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        wheel_members = dict(resources)
        wheel_members["scripts/evil.py"] = b"print(1)"
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", wheel_members)
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", _prefixed(resources))
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "non-package file in wheel" in capsys.readouterr().err

    def test_forbidden_file_in_wheel_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        wheel_members = dict(resources)
        wheel_members["Pipfile"] = b"[[source]]\n"
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", wheel_members)
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", _prefixed(resources))
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "forbidden file" in capsys.readouterr().err

    def test_forbidden_file_in_sdist_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", dict(resources))
        sdist_members = _prefixed(resources)
        sdist_members["lexnlp-2.3.0/libs/stanford_nlp/model.bin"] = b"binary"
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", sdist_members)
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "forbidden file" in capsys.readouterr().err

    def test_duplicate_archive_member_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _write_wheel(
            dist_dir / "lexnlp-2.3.0-py3-none-any.whl",
            dict(resources),
            extra_writes=[("lexnlp/data/a.json", resources["lexnlp/data/a.json"])],
        )
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", _prefixed(resources))
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "duplicate archive member" in capsys.readouterr().err

    def test_duplicate_runtime_resource_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        wheel_members = dict(resources)
        wheel_members["lexnlp-9.9/lexnlp/data/a.json"] = resources["lexnlp/data/a.json"]
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", wheel_members)
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", _prefixed(resources))
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "duplicate runtime resource" in capsys.readouterr().err

    def test_missing_resource_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        wheel_members = dict(resources)
        del wheel_members["lexnlp/data/a.json"]
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", wheel_members)
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", _prefixed(resources))
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "missing runtime resource" in capsys.readouterr().err

    def test_missing_segmentation_member_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(mod, "REQUIRED_SEGMENTATION_SDIST_MEMBERS", ("segmentation/gate.py",))
        dist_dir, src_root = _happy_tree(tmp_path)
        assert validate_artifacts(dist_dir, src_root) == 1
        assert "missing segmentation gate source member" in capsys.readouterr().err

    def test_present_segmentation_member_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(mod, "REQUIRED_SEGMENTATION_SDIST_MEMBERS", ("segmentation/gate.py",))
        resources = _good_resources()
        src_root = _write_source_root(tmp_path / "src", resources)
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        _write_wheel(dist_dir / "lexnlp-2.3.0-py3-none-any.whl", dict(resources))
        sdist_members = _prefixed(resources)
        sdist_members["lexnlp-2.3.0/segmentation/gate.py"] = b'"""gate"""\n'
        _write_sdist(dist_dir / "lexnlp-2.3.0.tar.gz", sdist_members)
        assert validate_artifacts(dist_dir, src_root) == 0
        assert "resource parity OK" in capsys.readouterr().out


class TestMain:
    def test_main_installed_failure(self, tmp_path: Path) -> None:
        assert main(["--installed", "--source-root", str(tmp_path / "no-src")]) == 1

    def test_main_artifacts_success(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dist_dir, src_root = _happy_tree(tmp_path)
        assert main([str(dist_dir), "--source-root", str(src_root)]) == 0
        assert "resource parity OK" in capsys.readouterr().out

    def test_main_artifacts_failure(self, tmp_path: Path) -> None:
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()
        src_root = _write_source_root(tmp_path / "src", _good_resources())
        assert main([str(dist_dir), "--source-root", str(src_root)]) == 1
