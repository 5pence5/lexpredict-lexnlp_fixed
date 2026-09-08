"""Coverage tests for lexnlp.extract.ml.detector.sample_processor."""

from __future__ import annotations

import numpy
import pandas

from lexnlp.extract.ml.detector.sample_processor import (
    get_target_start_end_from_corgetes,
    get_target_start_end_from_text,
    process_sample,
)


class RecordingClassifier:
    """Minimal token-sequence classifier with deterministic spans and features."""

    def __init__(self) -> None:
        self.feature_list = ["position", "length"]
        self.seen_masks: list[object] = []

    def get_feature_data(
        self, text: str, feature_mask: list[int] | None = None
    ) -> tuple[numpy.ndarray, list[tuple[int, int]]]:
        self.seen_masks.append(feature_mask)
        parts = text.split()
        n_tokens = len(parts)
        feature_data = numpy.arange(n_tokens * len(self.feature_list), dtype=numpy.int8).reshape(
            n_tokens, len(self.feature_list)
        )
        spans: list[tuple[int, int]] = []
        cursor = 0
        for part in parts:
            start = text.find(part, cursor)
            end = start + len(part)
            spans.append((start, end))
            cursor = end
        return feature_data, spans


class TestGetTargetStartEndFromText:
    def test_returns_span_of_formatted_phrase(self) -> None:
        text = "The cat sat on the mat"
        row = {"quantity_formatted": "cat sat"}
        assert get_target_start_end_from_text(text, "quantity_formatted", row) == [(4, 11)]

    def test_missing_phrase_uses_find_sentinel(self) -> None:
        text = "no match here"
        row = {"quantity_formatted": "absent"}
        start_end = get_target_start_end_from_text(text, "quantity_formatted", row)
        assert start_end == [(-1, 5)]


class TestGetTargetStartEndFromCorgetes:
    def test_returns_row_value_unchanged(self) -> None:
        coords = [(1, 4), (8, 12)]
        row = {"noun_phrase_formatted": coords}
        assert get_target_start_end_from_corgetes("ignored", "noun_phrase_formatted", row) is coords


class TestProcessSample:
    def test_empty_dataframe_returns_zero_row_feature_and_target(self) -> None:
        classifier = RecordingClassifier()
        sample_df = pandas.DataFrame(columns=["sentence", "quantity_formatted"])
        features, targets = process_sample(sample_df, classifier, pre_alloc_multiple=4)
        assert features.shape == (0, 2)
        assert targets.shape == (0,)
        assert features.dtype == numpy.int8

    def test_assigns_start_inner_end_and_outer_classes(self) -> None:
        classifier = RecordingClassifier()
        text = "aa bb cc dd"
        sample_df = pandas.DataFrame([{"sentence": text, "quantity_formatted": "bb cc dd"}])
        features, targets = process_sample(
            sample_df,
            classifier,
            pre_alloc_multiple=10,
            outer_class=0,
            start_class=1,
            inner_class=2,
            end_class=3,
        )
        assert features.shape[0] == 4
        assert list(targets) == [0, 1, 2, 3]

    def test_single_token_phrase_is_start_class(self) -> None:
        classifier = RecordingClassifier()
        sample_df = pandas.DataFrame([{"sentence": "aa bb cc", "quantity_formatted": "bb"}])
        _features, targets = process_sample(sample_df, classifier, pre_alloc_multiple=10)
        assert list(targets) == [0, 1, 0]

    def test_uses_corgetes_coordinates_when_supplied(self) -> None:
        classifier = RecordingClassifier()
        text = "aa bb cc"
        sample_df = pandas.DataFrame([{"sentence": text, "quantity_formatted": [(3, 5)]}])
        _features, targets = process_sample(
            sample_df,
            classifier,
            pre_alloc_multiple=10,
            get_target_start_end=get_target_start_end_from_corgetes,
        )
        assert list(targets) == [0, 1, 0]

    def test_feature_mask_column_is_forwarded(self) -> None:
        classifier = RecordingClassifier()
        mask = [0, 1, 1, 0]
        sample_df = pandas.DataFrame([{"sentence": "aa bb", "quantity_formatted": "aa", "mask": mask}])
        process_sample(
            sample_df,
            classifier,
            pre_alloc_multiple=10,
            feature_mask_column="mask",
        )
        assert classifier.seen_masks == [mask]

    def test_without_target_data_returns_features_only(self) -> None:
        classifier = RecordingClassifier()
        sample_df = pandas.DataFrame([{"sentence": "aa bb cc"}])
        result = process_sample(
            sample_df,
            classifier,
            build_target_data=False,
            pre_alloc_multiple=10,
        )
        assert isinstance(result, numpy.ndarray)
        assert result.shape == (3, 2)
        assert list(result[0]) == [0, 1]

    def test_resizes_feature_and_target_when_preallocation_is_tight(self) -> None:
        classifier = RecordingClassifier()
        sample_df = pandas.DataFrame(
            [
                {"sentence": "Hi", "quantity_formatted": "Hi"},
                {"sentence": "one two three", "quantity_formatted": "two"},
            ]
        )
        features, targets = process_sample(sample_df, classifier, pre_alloc_multiple=1)
        assert features.shape == (4, 2)
        assert targets.shape == (4,)
        assert list(targets) == [1, 0, 1, 0]

    def test_resizes_features_only_when_targets_are_disabled(self) -> None:
        classifier = RecordingClassifier()
        sample_df = pandas.DataFrame(
            [
                {"sentence": "Hi"},
                {"sentence": "one two three"},
            ]
        )
        features = process_sample(
            sample_df,
            classifier,
            build_target_data=False,
            pre_alloc_multiple=1,
        )
        assert features.shape == (4, 2)
        assert features[0, 0] == 0
        assert int(features[3, 1]) == 5

    def test_multiple_entity_spans_on_one_row(self) -> None:
        classifier = RecordingClassifier()
        text = "aa bb cc dd"
        sample_df = pandas.DataFrame([{"sentence": text, "spans": [(0, 2), (9, 11)]}])
        _features, targets = process_sample(
            sample_df,
            classifier,
            pre_alloc_multiple=10,
            column_name_formatted="spans",
            get_target_start_end=get_target_start_end_from_corgetes,
        )
        assert list(targets) == [1, 0, 0, 1]
