"""Coverage tests for :mod:`lexnlp.extract.common.universal_court_parser`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import pandas

from lexnlp.extract.common.universal_court_parser import (
    MatchFound,
    ParserInitParams,
    UniversalCourtsParser,
)
from lexnlp.utils.lines_processing.line_processor import LineOrPhrase
from lexnlp.utils.lines_processing.phrase_finder import PhraseFinder

_CSV_HEADER = "Court Type,Court Name,Jurisdiction,Alias"
_CSV_ROWS = [
    "Supreme Court,Supreme Court of Testland,Testland,SCT",
    "District Court,District Court of Foobar,Foobar,DCF",
]


def _make_parser() -> tuple[UniversalCourtsParser, tempfile.TemporaryDirectory[str]]:
    tmpdir = tempfile.TemporaryDirectory()
    csv_path = Path(tmpdir.name) / "courts.csv"
    csv_path.write_text("\n".join([_CSV_HEADER, *_CSV_ROWS]) + "\n", encoding="utf-8")
    ptrs = ParserInitParams()
    ptrs.dataframe_paths = [str(csv_path)]
    parser = UniversalCourtsParser(ptrs)
    return parser, tmpdir


class TestMatchFoundRepr(TestCase):
    def test_repr_with_fields(self) -> None:
        parser, tmpdir = _make_parser()
        try:
            subset = parser.courts.iloc[0:1]
            match = MatchFound(subset, 0, 5, "hello")
            match.court_name = "Supreme Court of Testland"
            match.court_type = "Supreme Court"
            match.jurisdiction = "Testland"
            text = repr(match)
            self.assertIn("Exact: True", text)
            self.assertIn("[1]", text)
            self.assertIn("court name: Supreme Court of Testland", text)
            self.assertIn("court type: Supreme Court", text)
            self.assertIn("jurisdiction: Testland", text)
        finally:
            tmpdir.cleanup()

    def test_repr_with_nil_subset(self) -> None:
        parser, tmpdir = _make_parser()
        try:
            subset = parser.courts.iloc[0:1]
            match = MatchFound(subset, 0, 5, "hello")
            match.subset = None
            text = repr(match)
            self.assertIn("[nil]", text)
            self.assertIn("Exact: True", text)
            self.assertNotIn("court name:", text)
            match.court_name = "X"
            match.court_type = "Y"
            match.jurisdiction = "Z"
            text_with_fields = repr(match)
            self.assertIn("court name: X", text_with_fields)
            self.assertIn("court type: Y", text_with_fields)
            self.assertIn("jurisdiction: Z", text_with_fields)
        finally:
            tmpdir.cleanup()


class TestLoadCourtsEmpty(TestCase):
    def test_load_courts_with_no_paths_returns_empty_frame(self) -> None:
        parser, tmpdir = _make_parser()
        try:
            frame = parser.load_courts([])
            self.assertIsInstance(frame, pandas.DataFrame)
            self.assertTrue(frame.empty)
            self.assertEqual(0, len(frame))
        finally:
            tmpdir.cleanup()


class TestFindCourtByKeyColumn(TestCase):
    def test_no_matching_row_returns_none(self) -> None:
        parser, tmpdir = _make_parser()
        try:
            finder = PhraseFinder(["Imaginary Court of Nowhere"])
            phrase = LineOrPhrase(" Imaginary Court of Nowhere ", 10)
            found = finder.find_word(phrase.text, True)
            self.assertEqual(1, len(found))
            self.assertEqual("Imaginary Court of Nowhere", found[0][0])
            self.assertIsNone(parser.find_court_by_key_column(phrase, finder, parser.court_name_column))
        finally:
            tmpdir.cleanup()

    def test_matching_row_returns_match(self) -> None:
        parser, tmpdir = _make_parser()
        try:
            phrase = LineOrPhrase(" Supreme Court of Testland ", 7)
            result = parser.find_court_by_key_column(phrase, parser.finder_court_name, parser.court_name_column)
            self.assertIsNotNone(result)
            assert result is not None
            match, found = result
            self.assertEqual("Supreme Court of Testland", found[0][0])
            self.assertEqual(1, len(match.subset))
            self.assertEqual(7 + found[0][1], match.entry_start)
            self.assertEqual(7 + found[0][2], match.entry_end)
        finally:
            tmpdir.cleanup()
