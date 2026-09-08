"""Coverage tests for scripts/unify_py_file_structure.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import unify_py_file_structure as unify


AUTHOR_TEMPLATE = unify.author.strip()


def _author_block(version: str) -> str:
    return AUTHOR_TEMPLATE.replace("0.0.0", version)


@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the script at a throwaway tree so the real package is never rewritten."""
    original_author = unify.author
    original_exclude = list(unify.exclude_paths)
    original_paths = list(unify.parse_paths)
    original_file = unify.__file__

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    fake_script = scripts_dir / "unify_py_file_structure.py"
    fake_script.write_text("# placeholder\n", encoding="utf-8")
    (tmp_path / "lexnlp").mkdir()
    (tmp_path / "lexnlpprivate").mkdir()

    monkeypatch.setattr(unify, "__file__", str(fake_script))
    monkeypatch.setattr(unify, "parse_paths", ["lexnlp", "lexnlpprivate"])
    monkeypatch.setattr(unify, "exclude_paths", [])
    yield tmp_path

    unify.author = original_author
    unify.exclude_paths = original_exclude
    unify.parse_paths = original_paths
    unify.__file__ = original_file


def _write_py(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestRegexAndAuthorPattern:
    def test_compiled_patterns_match_versioned_author_block(self) -> None:
        block = _author_block("2.3.0")
        assert unify.release_version_re.fullmatch("2.3.0")
        assert unify.release_version_re.fullmatch("10.11.12")
        assert unify.release_version_re.search(block)
        assert unify.py_file_struc_re.fullmatch(block) is not None

    def test_fullmatch_accepts_shebang_docstring_author_imports_and_code(self) -> None:
        source = (
            "#!/usr/bin/env python\n"
            "# -*- coding: utf-8 -*-\n"
            "\n"
            '"""Unify helper."""\n'
            "\n"
            f"{_author_block('2.3.0')}\n"
            "\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "VALUE = 1\n"
        )
        match = unify.py_file_struc_re.fullmatch(source)
        assert match is not None
        groups = match.groupdict()
        assert groups["service"].startswith("#!/usr/bin/env python")
        assert "Unify helper" in groups["docstr"]
        assert "__author__" in groups["author"]
        assert "import os" in groups["imports"]
        assert "VALUE = 1" in groups["code"]


class TestUnifyFileStructure:
    def test_rewrites_simple_module_with_release_version(
        self, isolated_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = _write_py(isolated_project / "lexnlp" / "sample.py", "VALUE = 7\n")

        unify.unify_file_structure("9.8.7")

        rewritten = target.read_text(encoding="utf-8")
        assert '__version__ = "9.8.7"' in rewritten
        assert "blob/9.8.7/LICENSE" in rewritten
        assert "VALUE = 7" in rewritten
        assert rewritten.count("__author__") == 1
        assert rewritten.endswith("\n")
        captured = capsys.readouterr()
        assert f"Done: {target}" in captured.out

    def test_preserves_shebang_docstring_imports_and_code(
        self, isolated_project: Path
    ) -> None:
        source = (
            "#!/usr/bin/env python\n"
            "# coding: utf-8\n"
            "\n"
            '"""Module docstring."""\n'
            "\n"
            f"{_author_block('1.2.3')}\n"
            "\n"
            "import os\n"
            "from collections.abc import Callable\n"
            "\n"
            "def ping() -> str:\n"
            '    return "pong"\n'
        )
        target = _write_py(isolated_project / "lexnlp" / "pkg" / "mod.py", source)

        unify.unify_file_structure("4.5.6")

        rewritten = target.read_text(encoding="utf-8")
        assert rewritten.startswith("#!/usr/bin/env python")
        assert '"""Module docstring."""' in rewritten
        assert '__version__ = "4.5.6"' in rewritten
        assert "import os" in rewritten
        assert "from collections.abc import Callable" in rewritten
        assert "def ping() -> str:" in rewritten
        assert 'return "pong"' in rewritten

    def test_warns_when_author_block_would_be_duplicated(
        self, isolated_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = _write_py(
            isolated_project / "lexnlp" / "dup.py",
            "x = 1\n__author__ = \"already here\"\n",
        )

        unify.unify_file_structure("0.1.0")

        rewritten = target.read_text(encoding="utf-8")
        assert rewritten.count("__author__") > 1
        captured = capsys.readouterr()
        assert "WARN!!! Duplicated author block" in captured.out
        assert str(target) in captured.out

    def test_exclude_paths_appends_root_py_files_except_excluded(
        self, isolated_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kept = _write_py(isolated_project / "keep_me.py", "KEEP = True\n")
        ignored = _write_py(isolated_project / "ignore_me.py", "IGNORE = True\n")
        nested = _write_py(isolated_project / "lexnlp" / "nested.py", "NESTED = True\n")
        monkeypatch.setattr(unify, "exclude_paths", ["ignore_me.py"])

        unify.unify_file_structure("3.2.1")

        kept_text = kept.read_text(encoding="utf-8")
        ignored_text = ignored.read_text(encoding="utf-8")
        nested_text = nested.read_text(encoding="utf-8")
        assert '__version__ = "3.2.1"' in kept_text
        assert "KEEP = True" in kept_text
        assert ignored_text == "IGNORE = True\n"
        assert "NESTED = True" in nested_text
        assert '__version__ = "3.2.1"' in nested_text

    def test_walks_lexnlpprivate_as_well(self, isolated_project: Path) -> None:
        private = _write_py(
            isolated_project / "lexnlpprivate" / "secret.py",
            "SECRET = 42\n",
        )

        unify.unify_file_structure("8.8.8")

        rewritten = private.read_text(encoding="utf-8")
        assert "SECRET = 42" in rewritten
        assert '__version__ = "8.8.8"' in rewritten

    def test_empty_package_dirs_are_a_no_op_walk(
        self, isolated_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        unify.unify_file_structure("1.0.0")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "1.0.0" in unify.author


class TestMain:
    def test_missing_release_number_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        script = _SCRIPTS_DIR / "unify_py_file_structure.py"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["unify_py_file_structure.py"])
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path(str(script), run_name="__main__")
        assert exc_info.value.code == 1
        assert 'Provide release number in format "1.2.3"' in capsys.readouterr().out

    def test_release_number_argument_runs_unifier(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = _SCRIPTS_DIR / "unify_py_file_structure.py"
        walked: list[str] = []

        def fake_walk(path: str):
            walked.append(path)
            return iter(())

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", ["unify_py_file_structure.py", "7.7.7"])
            mp.setattr("os.walk", fake_walk)
            mp.setattr("os.listdir", lambda _path: [])
            runpy.run_path(str(script), run_name="__main__")
        assert walked, "unify_file_structure should walk parse_paths"
        assert any(path.endswith("lexnlp") for path in walked)
        assert capsys.readouterr().out == ""
