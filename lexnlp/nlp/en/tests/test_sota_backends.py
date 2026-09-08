"""Tests for optional lossless sentence-boundary adapters."""

from unittest import TestCase

from lexnlp.nlp.en.segments.backends import SaTSentenceSegmenter, parts_to_spans


class _FakeSaT:
    def __init__(self, parts):
        self.parts = parts
        self.calls = []

    def split(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.parts


class TestSegmentationBackends(TestCase):
    def test_parts_to_spans_preserves_mixed_newlines_and_unicode(self):
        text = "Clause α.\r\nClause β."
        self.assertEqual(
            ((0, 11, "Clause α.\r\n"), (11, len(text), "Clause β.")),
            parts_to_spans(text, ["Clause α.\r\n", "Clause β."]),
        )

    def test_sat_adapter_is_injectable_exact_and_identified(self):
        text = "First.\nSecond."
        model = _FakeSaT(["First.\n", "Second."])
        adapter = SaTSentenceSegmenter(
            model,
            "segment-any-text/sat-3l-sm@revision-abc:threshold-0.42:v1",
            {"threshold": 0.42},
        )
        self.assertEqual(
            [(0, 7, "First.\n"), (7, len(text), "Second.")],
            list(adapter(text)),
        )
        self.assertEqual(
            [(text, {"split_on_input_newlines": False, "threshold": 0.42})],
            model.calls,
        )

    def test_adapter_fails_closed_on_identity_and_inexact_output(self):
        with self.assertRaisesRegex(ValueError, "backend_id"):
            SaTSentenceSegmenter(_FakeSaT(["Text."]), "")

        adapter = SaTSentenceSegmenter(
            _FakeSaT(["First.", "Second."]),
            "segment-any-text/fake@normalised:v1",
        )
        with self.assertRaisesRegex(ValueError, "preserve the input exactly"):
            list(adapter("First. Second."))

        nested = SaTSentenceSegmenter(
            _FakeSaT([["First."], ["Second."]]),
            "segment-any-text/fake@nested:v1",
        )
        with self.assertRaisesRegex(TypeError, "not a string"):
            list(nested("First.Second."))

    def test_empty_text_does_not_call_model(self):
        model = _FakeSaT([])
        self.assertEqual(
            [],
            list(SaTSentenceSegmenter(model, "segment-any-text/fake@empty:v1")("")),
        )
        self.assertEqual([], model.calls)
