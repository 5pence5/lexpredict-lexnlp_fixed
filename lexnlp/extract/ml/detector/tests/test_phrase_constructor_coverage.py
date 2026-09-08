__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.ml.detector.phrase_constructor import (
    PhraseConstructorMethod,
    PhraseConstructorSettings,
)


class TestPhraseConstructorSettingsRepr(TestCase):
    def test_repr_by_class_default(self):
        settings = PhraseConstructorSettings()
        self.assertEqual(PhraseConstructorMethod.by_class, settings.method)
        self.assertEqual("by class, strict=False", repr(settings))

    def test_repr_by_class_strict(self):
        settings = PhraseConstructorSettings(method=PhraseConstructorMethod.by_class, strict=True)
        self.assertEqual("by class, strict=True", repr(settings))

    def test_repr_by_score_defaults(self):
        settings = PhraseConstructorSettings(method=PhraseConstructorMethod.by_score)
        self.assertEqual("by score, min_score=2, max_zeros=2", repr(settings))

    def test_repr_by_score_custom(self):
        settings = PhraseConstructorSettings(
            method=PhraseConstructorMethod.by_score, max_zeros=5, min_token_score=7
        )
        self.assertEqual("by score, min_score=7, max_zeros=5", repr(settings))
