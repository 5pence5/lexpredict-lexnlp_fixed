"""Coverage tests for uncovered lines in lexnlp.ml.normalizers."""

from collections.abc import Generator
from unittest import TestCase

from lexnlp.extract.common.annotations.text_annotation import TextAnnotation
from lexnlp.ml.normalizers import Normalizer


def _hello_extractor(text: str) -> Generator[TextAnnotation]:
    yield TextAnnotation(name="t", locale="en", coords=(0, 5), text="hello")


def _empty_extractor(text: str) -> Generator[TextAnnotation]:
    if False:
        yield  # pragma: no cover - makes this a generator


def _boom_extractor(text: str) -> Generator[TextAnnotation]:
    raise RuntimeError("boom")
    yield  # pragma: no cover - makes this a generator


class TestNormalizerCoverage(TestCase):
    def test_init_stores_normalizations(self):
        norms = [(_hello_extractor, "__X__")]
        normalizer = Normalizer(normalizations=norms)
        self.assertIs(norms, normalizer.normalizations)

    def test_call_replaces_annotation(self):
        normalizer = Normalizer(normalizations=[(_hello_extractor, "__X__")])
        self.assertEqual(" __X__  world", normalizer("hello world"))
        # No annotations -> text passes through unchanged.
        normalizer_empty = Normalizer(normalizations=[(_empty_extractor, "__X__")])
        self.assertEqual("hello world", normalizer_empty("hello world"))

    def test_call_without_normalizations_returns_text(self):
        normalizer = Normalizer(normalizations=[])
        self.assertEqual("hello world", normalizer("hello world"))

    def test_exception_fallback_returns_original(self):
        normalizer = Normalizer(normalizations=[(_boom_extractor, "__X__")])
        self.assertEqual("hello world", normalizer("hello world"))
        self.assertEqual(
            ["hello world"],
            list(Normalizer._find_replace("hello world", _boom_extractor, " __X__ ")),
        )

    def test_strip_offsets_helper(self):
        self.assertEqual((2, 2), Normalizer._get_strip_offsets("  hi  "))
