"""Coverage tests for lexnlp.extract.common.language_dictionary_reader missing lines."""

from __future__ import annotations

import codecs
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from lexnlp.extract.common.language_dictionary_reader import LanguageDictionaryReader


class TestLanguageDictionaryReaderCoverage(TestCase):
    def test_read_str_set_skips_empty_readline_entry(self) -> None:
        # Real file objects never yield "" from readlines(), so line 26
        # (``if not line: continue``) is only reachable with a mocked reader
        # whose readlines() list contains an empty string. It must be skipped.
        fake_file = MagicMock()
        fake_file.__enter__.return_value = fake_file
        fake_file.__exit__.return_value = False
        fake_file.readlines.return_value = ["", "alpha\n", "\n", "  beta  \n"]
        with patch.object(codecs, "open", return_value=fake_file) as mock_open:
            words = LanguageDictionaryReader.read_str_set("dummy/path.txt")
        mock_open.assert_called_once_with("dummy/path.txt", encoding="utf8", mode="r")
        self.assertEqual({"alpha", "beta"}, words)

    def test_read_str_set_real_file_strips_and_skips_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text("alpha\n\n   \n  beta  \ngamma\n", encoding="utf8")
            words = LanguageDictionaryReader.read_str_set(str(path))
        # Truly empty lines ("\n") are skipped, but a whitespace-only line
        # ("   \n") survives the "\n" strip and becomes "" after stripping.
        self.assertEqual({"alpha", "", "beta", "gamma"}, words)

    def test_read_str_set_no_strip_symbols_keeps_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text("  beta  \n", encoding="utf8")
            words = LanguageDictionaryReader.read_str_set(str(path), strip_symbols="")
        self.assertEqual({"  beta  "}, words)
