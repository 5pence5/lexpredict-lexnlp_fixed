__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.ml.detector.detecting_settings import DetectingSettings


class TestDetectingSettingsDefaults:
    def test_default_values(self) -> None:
        settings = DetectingSettings()
        assert settings.use_spacy is False
        assert settings.pre_window == 0
        assert settings.post_window == 0
        assert settings.model_type == "random_forest"

    def test_explicit_values_stored(self) -> None:
        settings = DetectingSettings(use_spacy=True, pre_window=1, post_window=2, model_type="extra_trees")
        assert settings.use_spacy is True
        assert settings.pre_window == 1
        assert settings.post_window == 2
        assert settings.model_type == "extra_trees"

    def test_repr_defaults(self) -> None:
        settings = DetectingSettings()
        text = repr(settings)
        assert text == "use_spacy=False, pre_window=0, post_window=0, model_type=random_forest"

    def test_repr_explicit(self) -> None:
        settings = DetectingSettings(use_spacy=True, pre_window=1, post_window=2, model_type="extra_trees")
        text = repr(settings)
        assert text == "use_spacy=True, pre_window=1, post_window=2, model_type=extra_trees"
        assert "use_spacy=True" in text
        assert "pre_window=1" in text
        assert "post_window=2" in text
        assert "model_type=extra_trees" in text
