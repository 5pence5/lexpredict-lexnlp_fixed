"""Focused, isolated tests for the local release-branch helper."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = REPOSITORY_ROOT / "scripts" / "create_release_branch.sh"


class CreateReleaseBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.repo = self.root / "fixture"
        self.repo.mkdir()
        (self.repo / "lexnlp").mkdir()
        (self.repo / "pyproject.toml").write_text(
            "[project]\nname = \"lexnlp\"\nversion = \"2.3.0\"\n", encoding="utf-8"
        )
        (self.repo / "lexnlp" / "__init__.py").write_text(
            '__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"\n'
            '__version__ = "2.3.0"\n',
            encoding="utf-8",
        )
        (self.repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        self._git("init")
        self._git("config", "user.email", "tests@example.invalid")
        self._git("config", "user.name", "Release helper tests")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")

        self.bin_directory = self.root / "bin"
        self.bin_directory.mkdir()
        self.uv_log = self.root / "uv.log"
        (self.bin_directory / "uv").write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$UV_LOG\"\n", encoding="utf-8"
        )
        (self.bin_directory / "uv").chmod(0o755)

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=self.repo, check=True, text=True, capture_output=True)

    def _run_helper(self, *args: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ | {
            "PATH": f"{self.bin_directory}{os.pathsep}{os.environ['PATH']}",
            "UV_LOG": str(self.uv_log),
        }
        return subprocess.run(
            [str(RELEASE_SCRIPT), *args], cwd=self.repo, text=True, capture_output=True, env=environment
        )

    def test_creates_local_branch_and_updates_canonical_sources(self) -> None:
        result = self._run_helper("2.3.1")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._git("branch", "--show-current").stdout.strip(), "release/2.3.1")
        self.assertIn('version = "2.3.1"', (self.repo / "pyproject.toml").read_text(encoding="utf-8"))
        init_text = (self.repo / "lexnlp" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('__version__ = "2.3.1"', init_text)
        self.assertIn("/blob/2.3.1/LICENSE", init_text)
        self.assertEqual(self.uv_log.read_text(encoding="utf-8").splitlines(), ["lock", "lock --locked"])

    def test_rejects_a_dirty_worktree_before_creating_a_branch(self) -> None:
        (self.repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        result = self._run_helper("2.3.1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("working tree is not clean", result.stderr)
        self.assertEqual(self._git("branch", "--list", "release/2.3.1").stdout, "")

    def test_rejects_inconsistent_canonical_versions_before_creating_a_branch(self) -> None:
        init_path = self.repo / "lexnlp" / "__init__.py"
        init_path.write_text(
            init_path.read_text(encoding="utf-8").replace('"2.3.0"', '"2.2.9"'), encoding="utf-8"
        )
        self._git("add", "lexnlp/__init__.py")
        self._git("commit", "-m", "make runtime version inconsistent")

        result = self._run_helper("2.3.1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disagree", result.stderr)
        self.assertEqual(self._git("branch", "--list", "release/2.3.1").stdout, "")

    def test_rejects_an_existing_remote_tracking_branch(self) -> None:
        self._git("update-ref", "refs/remotes/origin/release/2.3.1", "HEAD")
        result = self._run_helper("2.3.1")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("remote-tracking branch already exists", result.stderr)
        self.assertEqual(self._git("branch", "--list", "release/2.3.1").stdout, "")


if __name__ == "__main__":
    unittest.main()
