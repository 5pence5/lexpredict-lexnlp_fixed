"""Coverage tests for ci/skip_audit.py."""

from __future__ import annotations

import ast
import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

from ci import skip_audit

MARKERS_SOURCE = """\
import pytest

VALUE = 1

alias = pytest.mark.skip


@pytest.mark.skip(reason="legacy")
def test_a():
    pass


@pytest.mark.skipif(
    True,
    reason="flaky",
)
def test_b():
    pass


@pytest.mark.xfail(strict=True)
def test_c():
    pass


pytest.mark.skip("direct call")
notify(pytest.mark.xfail)
pytest.skip("not a marker")
pytest.mark.other("unknown marker")
check_something()
value = obj.attr
"""

ANNOTATED_SOURCE = """\
import pytest


# skip-audit: issue=PROJ-123 expires=2999-01-01
@pytest.mark.skip(reason="temporary")
def test_a():
    pass
"""

EXPIRED_SOURCE = """\
import pytest


# skip-audit: issue=PROJ-123 expires=2000-01-01
@pytest.mark.skip(reason="temporary")
def test_a():
    pass
"""

BAD_DATE_SOURCE = """\
import pytest


# skip-audit: issue=PROJ-123 expires=2026-13-99
@pytest.mark.skip(reason="temporary")
def test_a():
    pass
"""

MISSING_SOURCE = """\
import pytest


@pytest.mark.skip(reason="no ticket yet")
def test_a():
    pass
"""

FLAGGED_SOURCE = """\
import pytest


@pytest.mark.skip(reason="legacy")
def test_old():
    pass
"""


def _write(repo: Path, relative: str, content: str | bytes) -> Path:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _hide_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Force the rglob fallback by making the `git` executable unfindable."""
    empty = tmp_path / "empty_bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))


def _install_fake_git(bin_dir: Path, stdout: str, exit_code: int = 0) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    git = bin_dir / "git"
    git.write_text(
        "#!" + sys.executable + f"\nimport sys\nsys.stdout.write({stdout!r})\nsys.exit({exit_code})\n",
        encoding="utf-8",
    )
    git.chmod(git.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _prepend_path(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ.get("PATH", ""))


def _call_func(source: str) -> ast.AST:
    return ast.parse(source).body[0].value.func  # type: ignore[attr-defined]


def test_list_python_files_uses_git_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _install_fake_git(tmp_path / "bin", "b.py\n\n  \n a.py \n")
    _prepend_path(monkeypatch, tmp_path / "bin")
    assert skip_audit.list_python_files(repo) == [repo / "a.py", repo / "b.py"]


def test_list_python_files_falls_back_when_git_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _write(repo, ".git/ignored.py", "VALUE = 1\n")
    _write(repo, "b.py", "VALUE = 1\n")
    _write(repo, "a.py", "VALUE = 1\n")
    _install_fake_git(tmp_path / "bin", "", exit_code=1)
    _prepend_path(monkeypatch, tmp_path / "bin")
    assert skip_audit.list_python_files(repo) == [repo / "a.py", repo / "b.py"]


def test_list_python_files_falls_back_when_git_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _write(repo, "only.py", "VALUE = 1\n")
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.list_python_files(repo) == [repo / "only.py"]


def test_marker_kind_rejects_non_attribute_nodes() -> None:
    call = ast.parse('skip("x")').body[0].value
    assert isinstance(call, ast.Call)
    assert skip_audit.marker_kind(call) is None
    name = ast.parse("skip").body[0].value
    assert skip_audit.marker_kind(name) is None


def test_marker_kind_rejects_unknown_attributes() -> None:
    assert skip_audit.marker_kind(_call_func('pytest.mark.other("x")')) is None


def test_marker_kind_accepts_pytest_mark_kinds() -> None:
    assert skip_audit.marker_kind(_call_func('pytest.mark.skip("x")')) == "skip"
    assert skip_audit.marker_kind(_call_func("pytest.mark.skipif(True)")) == "skipif"
    assert skip_audit.marker_kind(_call_func("pytest.mark.xfail()")) == "xfail"


def test_marker_kind_rejects_non_pytest_mark_chains() -> None:
    assert skip_audit.marker_kind(_call_func('pytest.skip("x")')) is None
    assert skip_audit.marker_kind(_call_func('pytest.other.skip("x")')) is None


def test_marker_kind_rejects_wrong_mark_owner() -> None:
    assert skip_audit.marker_kind(_call_func('other.mark.skip("x")')) is None
    assert skip_audit.marker_kind(_call_func('a.b.mark.skip("x")')) is None


def test_collect_markers_finds_call_and_attribute_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _write(repo, "markers.py", MARKERS_SOURCE)
    _write(repo, "clean.py", "VALUE = 1\n")
    _hide_git(monkeypatch, tmp_path)
    markers, errors = skip_audit.collect_markers(repo)
    assert errors == []
    assert [(marker.line, marker.kind) for marker in markers] == [
        (5, "skip"),
        (8, "skip"),
        (13, "skipif"),
        (21, "xfail"),
        (26, "skip"),
        (27, "xfail"),
    ]
    assert all(marker.path == Path("markers.py") for marker in markers)
    assert [marker.col for marker in markers] == [8, 1, 1, 1, 0, 7]
    by_line = {marker.line: marker for marker in markers}
    assert by_line[5].expression == "pytest.mark.skip"
    assert "\n" not in by_line[13].expression
    assert "  " not in by_line[13].expression
    assert by_line[13].expression.startswith("pytest.mark.skipif(")


def test_collect_markers_reports_unreadable_and_broken_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    _write(repo, "ok.py", "VALUE = 1\n")
    _write(repo, "bad_encoding.py", b"\xff\xfe\x00bad\n")
    _write(repo, "broken.py", "def broken(:\n")
    _hide_git(monkeypatch, tmp_path)
    markers, errors = skip_audit.collect_markers(repo)
    assert markers == []
    assert len(errors) == 2
    assert any("bad_encoding.py" in error and "UTF-8" in error for error in errors)
    assert any("broken.py" in error and "syntax error" in error for error in errors)


def test_load_allowlist_missing_file_returns_empty(tmp_path: Path) -> None:
    assert skip_audit.load_allowlist(tmp_path / "nope.txt") == set()


def test_load_allowlist_skips_comments_and_blanks(tmp_path: Path) -> None:
    allow = tmp_path / "allow.txt"
    allow.write_text("# comment\n\n  \na.py:1:skip\n b.py:skip:sha256=abc123 \n# tail\n", encoding="utf-8")
    assert skip_audit.load_allowlist(allow) == {"a.py:1:skip", "b.py:skip:sha256=abc123"}


def test_find_annotation_scans_marker_line_and_two_lines_above() -> None:
    lines = [
        "# skip-audit: issue=OLD-1 expires=2000-01-01",
        "",
        "x = 1",
        "y = 2",
    ]
    assert skip_audit.find_annotation(lines, 1) is not None
    match = skip_audit.find_annotation(lines, 3)
    assert match is not None
    assert match.group("issue") == "OLD-1"
    assert skip_audit.find_annotation(lines, 4) is None


def test_find_annotation_matches_on_marker_line_itself() -> None:
    lines = ["value = 1  # skip-audit: issue=A-9 expires=2999-12-31"]
    match = skip_audit.find_annotation(lines, 1)
    assert match is not None
    assert match.group("expires") == "2999-12-31"


def test_parse_args_defaults_point_at_repo() -> None:
    args = skip_audit.parse_args([])
    assert args.allowlist == "ci/skip_audit_allowlist.txt"
    assert args.print_markers is False
    expected = Path(skip_audit.__file__).resolve().parent.parent
    assert Path(args.repo_root) == expected


def test_parse_args_accepts_overrides(tmp_path: Path) -> None:
    args = skip_audit.parse_args(["--repo-root", str(tmp_path), "--allowlist", "custom.txt", "--print-markers"])
    assert args.repo_root == str(tmp_path)
    assert args.allowlist == "custom.txt"
    assert args.print_markers is True


def test_marker_keys_use_legacy_and_stable_forms() -> None:
    marker = skip_audit.Marker(path=Path("pkg/test_mod.py"), line=10, col=4, kind="skip", expression="pytest.mark.skip")
    assert marker.key == "pkg/test_mod.py:10:skip"
    digest = hashlib.sha256(b"pytest.mark.skip").hexdigest()[:12]
    assert marker.stable_key == f"pkg/test_mod.py:skip:sha256={digest}"


def test_main_ok_when_repo_has_no_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "clean.py", "VALUE = 1\n")
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 0
    out, err = capsys.readouterr()
    assert "skip-audit: OK" in out
    assert "(markers=0, allowlisted=0, annotated_new=0)" in out
    assert err == ""


def test_main_ok_with_valid_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "annotated.py", ANNOTATED_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 0
    out, _ = capsys.readouterr()
    assert "(markers=1, allowlisted=0, annotated_new=1)" in out


def test_main_reports_missing_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", MISSING_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 1
    _, err = capsys.readouterr()
    assert "skip-audit: policy violations found" in err
    assert "flagged.py:4:skip missing annotation" in err
    assert "ci/skip_audit_allowlist.txt" in err


def test_main_reports_expired_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", EXPIRED_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 1
    _, err = capsys.readouterr()
    assert "flagged.py:5:skip has expired annotation" in err
    assert "expires=2000-01-01" in err


def test_main_reports_invalid_expires_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", BAD_DATE_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 1
    _, err = capsys.readouterr()
    assert "flagged.py:5:skip has invalid expires date: 2026-13-99" in err


def test_main_accepts_legacy_allowlist_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", FLAGGED_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    (markers, _) = skip_audit.collect_markers(repo)
    assert len(markers) == 1
    _write(repo, "ci/skip_audit_allowlist.txt", f"# legacy\n\n{markers[0].key}\n")
    assert skip_audit.main(["--repo-root", str(repo)]) == 0
    out, _ = capsys.readouterr()
    assert "(markers=1, allowlisted=1, annotated_new=0)" in out


def test_main_accepts_stable_allowlist_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", FLAGGED_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    (markers, _) = skip_audit.collect_markers(repo)
    assert len(markers) == 1
    _write(repo, "allow.txt", f"{markers[0].stable_key}\n")
    assert skip_audit.main(["--repo-root", str(repo), "--allowlist", "allow.txt"]) == 0
    out, _ = capsys.readouterr()
    assert "(markers=1, allowlisted=1, annotated_new=0)" in out


def test_main_fails_on_parse_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "broken.py", "def broken(:\n")
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo)]) == 1
    _, err = capsys.readouterr()
    assert "skip-audit: parse errors detected" in err
    assert "broken.py" in err


def test_main_print_markers_lists_detected_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "flagged.py", FLAGGED_SOURCE)
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo), "--print-markers"]) == 0
    out, _ = capsys.readouterr()
    assert "flagged.py:4:skip" in out
    assert "stable=" in out
    assert "expr=pytest.mark.skip" in out


def test_main_accepts_absolute_allowlist_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    _write(repo, "annotated.py", ANNOTATED_SOURCE)
    allow = tmp_path / "allow.txt"
    allow.write_text("", encoding="utf-8")
    _hide_git(monkeypatch, tmp_path)
    assert skip_audit.main(["--repo-root", str(repo), "--allowlist", str(allow)]) == 0
    out, _ = capsys.readouterr()
    assert "skip-audit: OK" in out
