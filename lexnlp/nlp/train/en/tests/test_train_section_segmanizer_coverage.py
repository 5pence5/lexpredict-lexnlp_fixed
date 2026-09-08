"""Coverage tests for SectionSegmentizerTrainManager.

Targets ``lexnlp/nlp/train/en/train_section_segmanizer.py``: feature building,
the sklearn training helpers, model dumping and the document-distribution /
section-break feature helpers. Heavy I/O (sample repository downloads) is
mocked; all feature math and sklearn fitting run for real.
"""

from __future__ import annotations

import pandas
import pytest

from lexnlp.nlp.train.en import train_section_segmanizer as seg_module
from lexnlp.nlp.train.en.train_section_segmanizer import SectionSegmentizerTrainManager


def _tiny_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {"f0": 0.0, "f1": 1.0},
            {"f0": 1.0, "f1": 0.0},
            {"f0": 0.2, "f1": 0.8},
            {"f0": 0.9, "f1": 0.1},
        ]
    )


class TestInit:
    def test_defaults(self) -> None:
        manager = SectionSegmentizerTrainManager()
        assert manager.line_window_pre == 3
        assert manager.line_window_post == 3
        assert manager.feature_data == []
        assert manager.target_data == []
        assert manager.feature_df is None


class TestBuildFeatures:
    def test_builds_rows_targets_and_frame(self, tmp_path, monkeypatch) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("Section 1\nbody\nSection 2\ntail\nmore\nend\n", encoding="utf-8")

        def fake_ensure(positions, target_path, mapping):
            assert target_path == "target"
            return {str(doc): [0, 2]}

        monkeypatch.setattr(seg_module, "ensure_documents_in_folder", fake_ensure)
        manager = SectionSegmentizerTrainManager()
        manager.build_features("repo", "target")
        assert len(manager.feature_data) == 6
        assert manager.target_data == [1, 0, 1, 0, 0, 0]
        assert manager.feature_df is not None
        assert manager.feature_df.shape[0] == 6
        assert manager.feature_df.shape[1] == len(manager.feature_data[0])
        assert "section" in manager.feature_data[0]

    def test_short_document_leaves_window_loop_empty(self, tmp_path, monkeypatch) -> None:
        doc = tmp_path / "tiny.txt"
        doc.write_text("only\nthree\nlines\n", encoding="utf-8")
        monkeypatch.setattr(
            seg_module,
            "ensure_documents_in_folder",
            lambda positions, target_path, mapping: {str(doc): []},
        )
        manager = SectionSegmentizerTrainManager()
        with pytest.raises(UnboundLocalError):
            manager.build_features("repo", "target")


class TestTrainHelpers:
    def test_train_logistic_regression_rejects_l1_with_lbfgs(self) -> None:
        manager = SectionSegmentizerTrainManager()
        manager.feature_df = _tiny_frame()
        manager.target_data = [0, 1, 0, 1]
        with pytest.raises(ValueError, match="lbfgs"):
            manager.train_logistic_regression()

    def test_train_extra_trees_classifier(self) -> None:
        manager = SectionSegmentizerTrainManager()
        manager.feature_df = _tiny_frame()
        manager.target_data = [0, 1, 0, 1]
        model = manager.train_extra_trees_classifier()
        assert model.n_estimators == 50
        assert list(model.predict(manager.feature_df)) == [0, 1, 0, 1]

    def test_train_decision_tree(self) -> None:
        manager = SectionSegmentizerTrainManager()
        manager.feature_df = _tiny_frame()
        manager.target_data = [0, 1, 0, 1]
        model = manager.train_decision_tree()
        assert model.max_leaf_nodes == 256
        assert list(model.predict(manager.feature_df)) == [0, 1, 0, 1]


class TestDumpModel:
    def test_dump_writes_project_level_pickle(self, monkeypatch, capsys) -> None:
        calls: list[tuple] = []

        def fake_dump(model, target_path) -> None:
            calls.append((model, target_path))

        monkeypatch.setattr(seg_module.joblib, "dump", fake_dump)
        monkeypatch.setattr(seg_module.os.path, "getsize", lambda path: 2 * 1024 * 1024)
        sentinel = object()
        SectionSegmentizerTrainManager.dump_model_on_project_level(sentinel)
        assert len(calls) == 1
        model, target_path = calls[0]
        assert model is sentinel
        assert target_path.endswith("section_segmenter.pickle")
        out = capsys.readouterr().out
        assert "is updated" in out
        assert "2.0" in out


class TestBuildDocumentDistribution:
    def test_raw_counts_without_norm(self) -> None:
        manager = SectionSegmentizerTrainManager()
        dist = manager._build_document_distribution("ab\na", norm=False)
        assert dist["doc_char_a"] == 2
        assert dist["doc_char_b"] == 1
        assert dist["doc_startchar_a"] == 2
        assert dist["doc_startchar_b"] == 0

    def test_non_printable_start_char_counts_as_other(self) -> None:
        manager = SectionSegmentizerTrainManager()
        dist = manager._build_document_distribution("abc\n\u2603 snowman", norm=False)
        assert dist["doc_startchar_other"] == 1
        assert dist["doc_startchar_a"] == 1

    def test_blank_lines_are_skipped(self) -> None:
        manager = SectionSegmentizerTrainManager()
        dist = manager._build_document_distribution("   \nabc", norm=False)
        assert dist["doc_startchar_a"] == 1
        assert sum(v for k, v in dist.items() if k.startswith("doc_startchar")) == 1

    def test_norm_scales_to_one(self) -> None:
        manager = SectionSegmentizerTrainManager()
        dist = manager._build_document_distribution("ab\na", norm=True)
        char_total = sum(v for k, v in dist.items() if k.startswith("doc_char"))
        start_total = sum(v for k, v in dist.items() if k.startswith("doc_startchar"))
        assert char_total == pytest.approx(1.0)
        assert start_total == pytest.approx(1.0)


class TestBuildSectionBreakFeatures:
    def test_start_line_shrinks_pre_window(self) -> None:
        # NOTE: the section/article keyword flags below the window loop read
        # the *last* window line (lines[3] here), not lines[line_id].
        manager = SectionSegmentizerTrainManager()
        lines = ["Section 1", "body", "tail", "more", "end"]
        features = manager._build_section_break_features(lines, 0)
        assert manager.line_window_pre == 0
        assert "line_len_0" in features
        assert "line_lenstrip_0" in features
        assert features["line_len_0"] == len("Section 1")
        assert features["section"] == 0
        assert features["Section"] == 0
        assert features["sw_section"] == 0

    def test_end_line_shrinks_post_window_and_skips_out_of_range(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["l0", "l1", "l2", "l3", "l4"]
        features = manager._build_section_break_features(lines, 4)
        assert manager.line_window_post == 1
        assert "line_len_1" not in features
        assert "line_len_0" in features
        assert "line_len_-3" in features

    def test_keyword_flags(self) -> None:
        # Keyword flags are read from the last window line, so the keyword
        # line is placed at index 3 (window for line_id=0 covers 0..3).
        manager = SectionSegmentizerTrainManager()
        lines = ["x", "y", "z", "section SECTION Section article ARTICLE Article", "w", "v"]
        features = manager._build_section_break_features(lines, 0)
        assert features["section"] == 1
        assert features["SECTION"] == 1
        assert features["Section"] == 1
        assert features["article"] == 1
        assert features["ARTICLE"] == 1
        assert features["Article"] == 1
        assert features["sw_section"] == 1
        assert features["sw_article"] == 0
        assert features["first_char_punct"] is False
        assert features["last_char_punct"] is False
        assert features["first_char_number"] is False
        assert features["last_char_number"] is False

    def test_article_prefix_and_punct_digit_edges(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["x", "y", "z", "Article 1. terms:", "w", "v"]
        features = manager._build_section_break_features(lines, 0)
        assert features["sw_article"] == 1
        assert features["sw_section"] == 0
        assert features["first_char_number"] is False
        assert features["last_char_punct"] is True

    def test_digit_first_and_last(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["x", "y", "z", "1st section 2", "w", "v"]
        features = manager._build_section_break_features(lines, 0)
        assert features["first_char_number"] is True
        assert features["last_char_number"] is True
        assert features["first_char_punct"] is False

    def test_empty_line_flags_are_false(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["x", "y", "z", "", "w", "v"]
        features = manager._build_section_break_features(lines, 0)
        assert features["section"] == 0
        assert features["sw_section"] == 0
        assert features["first_char_punct"] is False
        assert features["last_char_punct"] is False
        assert features["first_char_number"] is False
        assert features["last_char_number"] is False

    def test_char_counts_and_doc_distribution(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["aab", "x", "y", "z", "w"]
        features = manager._build_section_break_features(lines, 0)
        assert features["char_a"] == 2
        assert features["char_b"] == 1
        assert "doc_char_a" not in features
        with_doc = manager._build_section_break_features(lines, 1, include_doc={"doc_char_a": 0.5})
        assert with_doc["doc_char_a"] == 0.5

    def test_line_statistics(self) -> None:
        manager = SectionSegmentizerTrainManager()
        lines = ["  Hello World  ", "x", "y", "z", "w"]
        features = manager._build_section_break_features(lines, 0)
        assert features["line_len_0"] == len("  Hello World  ")
        assert features["line_lenstrip_0"] == len("Hello World")
        assert features["line_title_case_0"] == ("  Hello World  ".title() == "  Hello World  ")
        assert features["line_n_alpha_0"] > 0
        assert features["line_n_whitespace_0"] > 0
