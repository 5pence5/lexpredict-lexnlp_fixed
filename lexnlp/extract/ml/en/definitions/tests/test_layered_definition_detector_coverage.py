"""Coverage tests for LayeredDefinitionDetector load, distance, train, and merge paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from zipfile import ZipFile

import pandas
from pandas.testing import assert_frame_equal

from lexnlp.extract.ml.en.definitions.layered_definition_detector import LayeredDefinitionDetector


class _FixedSpanDetector:
    def __init__(self, spans: list[tuple[int, int]]) -> None:
        self.spans = spans
        self.load_paths: list[str] = []
        self.train_calls: list[tuple[object, pandas.DataFrame, str, bool]] = []

    def load(self, file_path: str) -> None:
        self.load_paths.append(file_path)

    def predict_text(self, sentence: str, join_settings=None, feature_mask=None):
        return list(self.spans)

    def train_and_save_on_dataframe(self, settings, frame, path, compress=False) -> None:
        self.train_calls.append((settings, frame.copy(), path, compress))
        Path(path).write_bytes(b"trained")


class TestLoadCompressed(TestCase):
    def test_empty_archive_raises_when_no_model_members(self) -> None:
        folder = Path(self._tmp())
        archive = folder / "empty.zip"
        with ZipFile(archive, "w") as packed:
            packed.writestr("readme.txt", "no models here")
        detector = LayeredDefinitionDetector()
        with self.assertRaises(RuntimeError) as caught:
            detector.load_compressed(str(archive))
        self.assertIn("No model members found", str(caught.exception))
        self.assertIn(str(archive), str(caught.exception))
        self.assertFalse(detector.initialized)
        self.assertFalse((folder / "unpack_def_model_temp").exists())

    def test_unknown_member_name_raises(self) -> None:
        folder = Path(self._tmp())
        archive = folder / "unknown.zip"
        with ZipFile(archive, "w") as packed:
            packed.writestr("other.pickle", b"nope")
        detector = LayeredDefinitionDetector()
        with self.assertRaises(RuntimeError) as caught:
            detector.load_compressed(str(archive))
        self.assertIn('Found unknown file "other.pickle"', str(caught.exception))
        self.assertFalse(detector.initialized)

    def test_definition_and_term_members_are_dispatched_by_stem(self) -> None:
        folder = Path(self._tmp())
        archive = folder / "models.zip"
        with ZipFile(archive, "w") as packed:
            packed.writestr("definition.skops", b"def-model")
            packed.writestr("term.pickle", b"term-model")
        detector = LayeredDefinitionDetector()
        definition = _FixedSpanDetector([])
        term = _FixedSpanDetector([])
        detector.model_definition = definition
        detector.model_term = term
        detector.load_compressed(str(archive))
        self.assertTrue(detector.initialized)
        self.assertEqual([Path(path).name for path in definition.load_paths], ["definition.skops"])
        self.assertEqual([Path(path).name for path in term.load_paths], ["term.pickle"])
        self.assertFalse((folder / "unpack_def_model_temp").exists())

    def _tmp(self) -> str:
        import tempfile

        return tempfile.mkdtemp()


class TestGetAnnotations(TestCase):
    def test_no_terms_returns_empty_list(self) -> None:
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([])
        detector.model_definition = _FixedSpanDetector([(0, 4)])
        self.assertEqual(detector.get_annotations("some defined term"), [])

    def test_definition_entirely_before_term_uses_gap_distance(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(6, 9)])
        detector.model_definition = _FixedSpanDetector([(0, 3)])
        ants = detector.get_annotations(sentence)
        self.assertEqual(len(ants), 1)
        self.assertEqual(ants[0].name, "ghi")
        self.assertEqual(ants[0].locale, "en")
        # defs[0] is (definition_index, distance); coords reuse that pair.
        self.assertEqual(ants[0].coords, (0, 9))
        self.assertEqual(ants[0].text, sentence[0:9])

    def test_definition_entirely_after_term_uses_gap_distance(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(0, 3)])
        detector.model_definition = _FixedSpanDetector([(7, 10)])
        ants = detector.get_annotations(sentence)
        self.assertEqual(len(ants), 1)
        self.assertEqual(ants[0].name, "abc")
        # index 0, distance 4; min(0, 0)=0, max(4, 3)=4
        self.assertEqual(ants[0].coords, (0, 4))
        self.assertEqual(ants[0].text, sentence[0:4])

    def test_definition_starting_inside_term_uses_negative_overlap(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(0, 6)])
        detector.model_definition = _FixedSpanDetector([(2, 8)])
        ants = detector.get_annotations(sentence)
        self.assertEqual(ants[0].name, "abcdef")
        # distance = -(6 - 2 + 1) = -5; min(0, 0)=0, max(-5, 6)=6
        self.assertEqual(ants[0].coords, (0, 6))

    def test_definition_ending_inside_term_uses_negative_overlap(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(2, 8)])
        detector.model_definition = _FixedSpanDetector([(0, 5)])
        ants = detector.get_annotations(sentence)
        self.assertEqual(ants[0].name, "cdefgh")
        # distance = -(5 - 2 + 1) = -4; min(0, 2)=0, max(-4, 8)=8
        self.assertEqual(ants[0].coords, (0, 8))

    def test_definition_containing_term_uses_term_length_distance(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(3, 6)])
        detector.model_definition = _FixedSpanDetector([(1, 9)])
        ants = detector.get_annotations(sentence)
        self.assertEqual(ants[0].name, "def")
        # distance = -(6 - 3 + 1) = -4; min(0, 3)=0, max(-4, 6)=6
        self.assertEqual(ants[0].coords, (0, 6))

    def test_term_without_definitions_falls_back_to_term_span(self) -> None:
        sentence = "abcdefghij"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([(2, 5)])
        detector.model_definition = _FixedSpanDetector([])
        ants = detector.get_annotations(sentence)
        self.assertEqual(len(ants), 1)
        self.assertEqual(ants[0].name, "cde")
        self.assertEqual(ants[0].coords, (2, 5))
        self.assertEqual(ants[0].text, "cde")


class TestJoinAdjacentDefinitionsLabels(TestCase):
    def test_empty_definitions_return_zero_mask(self) -> None:
        text = "abcdef"
        mask, labels = LayeredDefinitionDetector.join_adjacent_definitions_labels([], [(1, 3)], text)
        self.assertEqual(mask, [0] * len(text))
        self.assertEqual(labels, [])

    def test_adjacent_definitions_merge_and_absorb_following_term(self) -> None:
        text = "x" * 30
        mask, labels = LayeredDefinitionDetector.join_adjacent_definitions_labels(
            [(0, 10), (11, 20)],
            [(20, 25)],
            text,
        )
        self.assertEqual(labels, [(0, 26)])
        self.assertEqual(mask[19], 0)
        self.assertEqual(mask[20:25], [1, 1, 1, 1, 1])
        self.assertEqual(mask[25], 0)

    def test_separated_definitions_stay_distinct(self) -> None:
        text = "x" * 20
        mask, labels = LayeredDefinitionDetector.join_adjacent_definitions_labels(
            [(0, 4), (8, 12)],
            [(15, 18)],
            text,
        )
        self.assertEqual(labels, [(0, 4), (8, 12)])
        self.assertEqual(mask, [0] * 20)

    def test_unsorted_definitions_are_ordered_before_merge(self) -> None:
        text = "x" * 16
        mask, labels = LayeredDefinitionDetector.join_adjacent_definitions_labels(
            [(9, 12), (0, 8)],
            [],
            text,
        )
        self.assertEqual(labels, [(0, 12)])
        self.assertEqual(mask, [0] * 16)


class TestTrainOnDoccanoAndFormattedData(TestCase):
    def test_train_on_doccano_jsonl_splits_labels_and_zips_models(self) -> None:
        import tempfile

        folder = Path(tempfile.mkdtemp())
        jsonl_path = folder / "doccano.jsonl"
        save_path = folder / "layered.zip"
        row_text = "Member assets to any Person (a \"Merger Transaction\") unless:"
        # definition frames the term the way Doccano exports do
        labels = [
            [0, 27, "definition"],
            [27, 46, "term"],
            [46, 48, "definition"],
            [50, 56, "ignored"],
        ]
        jsonl_path.write_text(
            json.dumps(
                {
                    "text": row_text,
                    "labels": labels,
                    "annotation_approver": "alice",
                    "id": 7,
                    "meta": {"source": "fixture"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        detector = LayeredDefinitionDetector()
        term = _FixedSpanDetector([])
        definition = _FixedSpanDetector([])
        detector.model_term = term
        detector.model_definition = definition
        detector.train_on_doccano_jsonl(str(save_path), str(jsonl_path))

        self.assertEqual(len(term.train_calls), 1)
        self.assertEqual(len(definition.train_calls), 1)
        _term_settings, term_frame, term_path, term_compress = term.train_calls[0]
        _def_settings, def_frame, def_path, def_compress = definition.train_calls[0]
        self.assertFalse(term_compress)
        self.assertFalse(def_compress)
        self.assertEqual(list(term_frame.columns), ["sentence", "labels"])
        self.assertEqual(list(def_frame.columns), ["sentence", "labels", "feature_mask"])
        self.assertEqual(term_frame.iloc[0]["sentence"], row_text)
        self.assertEqual(term_frame.iloc[0]["labels"], [(27, 46)])
        self.assertEqual(Path(term_path).name, "terms")
        self.assertEqual(Path(def_path).name, "definitions")
        self.assertTrue(save_path.is_file())
        with ZipFile(save_path) as packed:
            self.assertEqual(set(packed.namelist()), {"term.pickle", "definition.pickle"})
            self.assertEqual(packed.read("term.pickle"), b"trained")
            self.assertEqual(packed.read("definition.pickle"), b"trained")
        self.assertFalse((folder / "def_model_temp").exists())

    def test_train_on_formatted_data_ignores_missing_temp_folder(self) -> None:
        import tempfile

        folder = Path(tempfile.mkdtemp())
        save_path = folder / "out.zip"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([])
        detector.model_definition = _FixedSpanDetector([])
        definition_frame = pandas.DataFrame(
            [["hello", [(0, 5)], [1, 1, 1, 1, 1]]],
            columns=["sentence", "labels", "feature_mask"],
        )
        term_frame = pandas.DataFrame([["hello", [(1, 4)]]], columns=["sentence", "labels"])
        detector.train_on_formatted_data(definition_frame, term_frame, str(save_path))
        self.assertTrue(save_path.is_file())
        with ZipFile(save_path) as packed:
            self.assertEqual(set(packed.namelist()), {"term.pickle", "definition.pickle"})
        _settings, captured_terms, _path, _compress = detector.model_term.train_calls[0]
        assert_frame_equal(captured_terms.reset_index(drop=True), term_frame.reset_index(drop=True))

    def test_zip_succeeds_even_if_temp_cleanup_raises(self) -> None:
        import shutil
        import tempfile
        from unittest.mock import patch

        folder = Path(tempfile.mkdtemp())
        save_path = folder / "out.zip"
        detector = LayeredDefinitionDetector()
        detector.model_term = _FixedSpanDetector([])
        detector.model_definition = _FixedSpanDetector([])
        definition_frame = pandas.DataFrame(
            [["hello", [(0, 5)], [1, 1, 1, 1, 1]]],
            columns=["sentence", "labels", "feature_mask"],
        )
        term_frame = pandas.DataFrame([["hello", [(1, 4)]]], columns=["sentence", "labels"])
        real_rmtree = shutil.rmtree
        calls = {"n": 0}

        def flaky_rmtree(path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("busy")
            return real_rmtree(path, *args, **kwargs)

        with patch(
            "lexnlp.extract.ml.en.definitions.layered_definition_detector.shutil.rmtree",
            flaky_rmtree,
        ):
            detector.train_on_formatted_data(definition_frame, term_frame, str(save_path))
        self.assertTrue(save_path.is_file())
        with ZipFile(save_path) as packed:
            self.assertEqual(set(packed.namelist()), {"term.pickle", "definition.pickle"})
        self.assertGreaterEqual(calls["n"], 2)
