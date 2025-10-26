import pathlib
import sys

import pandas

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lexnlp.nlp.train.en.train_section_segmanizer import SectionSegmentizerTrainManager


def test_keyword_features_use_target_line_only():
    trainer = SectionSegmentizerTrainManager()
    lines = ["Intro", "plain text", "Section Ten"]

    features = trainer._build_section_break_features(lines, 1)

    assert features["section"] == 0
    assert features["Section"] == 0
    assert features["sw_section"] == 0


def test_context_window_defaults_remain_unchanged():
    trainer = SectionSegmentizerTrainManager()
    lines = ["Section 1", "Body", "End"]

    trainer._build_section_break_features(lines, 0)

    assert trainer.line_window_pre == 3
    assert trainer.line_window_post == 3


def test_logistic_regression_uses_supported_solver():
    trainer = SectionSegmentizerTrainManager()
    trainer.feature_df = pandas.DataFrame(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        columns=["f1", "f2"],
    )
    trainer.target_data = [0, 1, 0, 1]

    model = trainer.train_logistic_regression()

    assert model.solver == "liblinear"
