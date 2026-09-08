__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest.mock import patch

import numpy
import pandas

from lexnlp.extract.ml.classifier.base_token_sequence_classifier_model import (
    BaseTokenSequenceClassifierModel,
)
from lexnlp.extract.ml.detector.detecting_settings import DetectingSettings
from lexnlp.extract.ml.en.definitions.definition_phrase_detector import DefinitionPhraseDetector

TEXT_ONE = "The Employment Period shall mean foo"
TEXT_TWO = "This Agreement means something else here"


def _make_detector_with_model() -> DefinitionPhraseDetector:
    detector = DefinitionPhraseDetector()
    detector.model = BaseTokenSequenceClassifierModel.get_classifier(
        use_spacy=False,
        match_tokens=["shall", "mean"],
        letter_set="abc",
        digit_set="123",
        punc_set=':"()',
    )
    return detector


def _make_phrase_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {
                "sentence": TEXT_ONE,
                "labels": [(0, len(TEXT_ONE))],
                "feature_mask": [0] * len(TEXT_ONE),
            },
            {
                "sentence": TEXT_TWO,
                "labels": [(0, len(TEXT_TWO))],
                "feature_mask": [0] * len(TEXT_TWO),
            },
        ]
    )


class TestProcessSample:
    def test_returns_feature_and_target_arrays(self) -> None:
        detector = _make_detector_with_model()
        frame = pandas.DataFrame(
            [
                {
                    "sentence": TEXT_ONE,
                    "labels": [(0, len(TEXT_ONE))],
                    "feature_mask": [0] * len(TEXT_ONE),
                }
            ]
        )
        result = detector.process_sample(frame)
        assert isinstance(result, tuple)
        assert len(result) == 2
        feature_data, target_data = result
        assert isinstance(feature_data, numpy.ndarray)
        assert isinstance(target_data, numpy.ndarray)
        assert feature_data.shape[0] == target_data.shape[0]
        assert feature_data.shape[0] == 6
        assert feature_data.shape[1] == len(detector.model.feature_list)
        # whole-sentence label marks every token as start/inner/end
        assert set(target_data.tolist()) == {1.0, 2.0, 3.0}
        assert target_data[0] == 1.0

    def test_feature_mask_is_used(self) -> None:
        detector = _make_detector_with_model()
        masked = [0] * len(TEXT_ONE)
        for i in range(4, 21):
            masked[i] = 1
        frame = pandas.DataFrame([{"sentence": TEXT_ONE, "labels": [(4, 21)], "feature_mask": masked}])
        feature_data, _ = detector.process_sample(frame)
        mask_index = detector.model.feature_list.index("mask")
        assert feature_data[:, mask_index].max() == 1


class TestTrainAndSaveOnDataframe:
    def test_trains_and_saves_model(self, tmp_path) -> None:
        detector = DefinitionPhraseDetector()
        frame = _make_phrase_frame()
        settings = DetectingSettings(use_spacy=False, pre_window=0, post_window=0)
        save_path = str(tmp_path / "phrase_model.pickle")
        detector.train_and_save_on_dataframe(settings, frame, save_path, compress=False)
        assert detector.model is not None
        assert detector.model.model is not None
        assert detector.model.punc_set == ':"()'
        assert any("shall" in feature for feature in detector.model.feature_list)
        assert any("mean" in feature for feature in detector.model.feature_list)
        import os

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
        predicted, targets = detector.predict(frame)
        assert predicted.shape[0] == targets.shape[0]
        assert predicted.shape[0] > 0


class TestTrainAndSave:
    def test_reads_csv_and_delegates(self, tmp_path) -> None:
        csv_path = tmp_path / "train.csv"
        frame = _make_phrase_frame()
        frame.to_csv(csv_path, index=False)
        settings = DetectingSettings(use_spacy=False)
        detector = DefinitionPhraseDetector()
        save_path = str(tmp_path / "out.pickle")
        with patch.object(detector, "train_and_save_on_dataframe", autospec=True) as mocked_save:
            detector.train_and_save(settings, str(csv_path), train_size=-1, save_path=save_path)
        assert mocked_save.call_count == 1
        _, kwargs = mocked_save.call_args
        delegated_frame = mocked_save.call_args[0][1]
        assert len(delegated_frame) == 2
        assert list(delegated_frame["sentence"]) == [TEXT_ONE, TEXT_TWO]
        assert mocked_save.call_args[0][0] is settings
        assert mocked_save.call_args[0][2] == save_path
        assert kwargs.get("compress", False) is False

    def test_train_size_limits_rows(self, tmp_path) -> None:
        csv_path = tmp_path / "train.csv"
        _make_phrase_frame().to_csv(csv_path, index=False)
        detector = DefinitionPhraseDetector()
        with patch.object(detector, "train_and_save_on_dataframe", autospec=True) as mocked_save:
            detector.train_and_save(
                DetectingSettings(use_spacy=False),
                str(csv_path),
                train_size=1,
                save_path=str(tmp_path / "out.pickle"),
            )
        delegated_frame = mocked_save.call_args[0][1]
        assert len(delegated_frame) == 1
        assert delegated_frame.iloc[0]["sentence"] == TEXT_ONE
