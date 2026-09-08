__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from pathlib import Path
from unittest import TestCase

from sklearn.linear_model import LogisticRegression

from lexnlp.ml.model_card import (
    ModelCardMetadata,
    dump_model_with_card,
    write_model_card,
)


class TestWriteModelCard(TestCase):
    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = Path(self.tmp.name)

    def _fitted_estimator(self) -> LogisticRegression:
        clf = LogisticRegression(max_iter=50, solver="lbfgs")
        # tiny fixture so add_hyperparams works without errors
        clf.fit([[0.0], [1.0]], [0, 1])
        return clf

    def test_returns_path_with_md_extension(self) -> None:
        clf = self._fitted_estimator()
        md = ModelCardMetadata(
            description="LexNLP date classifier",
            license="AGPL-3.0-or-later",
            authors="ContraxSuite, LLC",
        )
        out = write_model_card(clf, self.tmpdir / "date_model", metadata=md)
        self.assertEqual(out.suffix, ".md")
        self.assertTrue(out.exists())

    def test_card_contains_metadata(self) -> None:
        clf = self._fitted_estimator()
        md = ModelCardMetadata(
            description="LexNLP date classifier",
            license="AGPL-3.0-or-later",
            authors="ContraxSuite, LLC",
        )
        out = write_model_card(clf, self.tmpdir / "date_model.md", metadata=md)
        content = out.read_text(encoding="utf-8")
        self.assertIn("LexNLP date classifier", content)

    def test_card_metrics_table_included(self) -> None:
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", license="", authors="")
        out = write_model_card(
            clf,
            self.tmpdir / "m.md",
            metadata=md,
            metrics={"accuracy": 0.91, "f1_macro": 0.87},
        )
        content = out.read_text(encoding="utf-8")
        self.assertIn("accuracy", content)
        self.assertIn("0.91", content)

    def test_card_tags_render_each_value(self) -> None:
        """Every tag in ``ModelCardMetadata.tags`` surfaces in the final
        rendered card, not just the first one."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(
            description="x",
            license="",
            authors="",
            tags=("legal", "classifier"),
        )
        out = write_model_card(clf, self.tmpdir / "tagged.md", metadata=md)
        content = out.read_text(encoding="utf-8")
        self.assertIn("legal", content)
        self.assertIn("classifier", content)

    def test_card_metrics_keep_numeric_precision(self) -> None:
        """Numeric metric values must reach ``Card.add_metrics`` unchanged,
        so the rendered card preserves the caller's precision."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", license="", authors="")
        out = write_model_card(
            clf,
            self.tmpdir / "metrics.md",
            metadata=md,
            metrics={"accuracy": 0.123456},
        )
        content = out.read_text(encoding="utf-8")
        self.assertIn("0.123456", content)

    def test_dump_model_with_card_writes_both_artifacts(self) -> None:
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", license="", authors="")
        paths = dump_model_with_card(
            clf,
            self.tmpdir / "contract_type",
            metadata=md,
            metrics={"accuracy": 0.95},
        )
        self.assertTrue(paths.model.exists())
        self.assertEqual(paths.model.suffix, ".skops")
        self.assertTrue(paths.card.exists())
        self.assertEqual(paths.card.suffix, ".md")

    def test_dump_model_with_card_returns_sibling_paths(self) -> None:
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", license="", authors="")
        paths = dump_model_with_card(
            clf,
            self.tmpdir / "x",
            metadata=md,
        )
        self.assertEqual(paths.model.stem, paths.card.stem)
        self.assertEqual(paths.model.parent, paths.card.parent)


class TestModelCardMetadataValidation(TestCase):
    def test_required_description(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            ModelCardMetadata()  # type: ignore[call-arg] -- intentionally omitting required ``description`` to assert runtime constructor fails

    def test_frozen(self) -> None:
        import dataclasses

        md = ModelCardMetadata(description="x", license="", authors="")
        # ``dataclasses.FrozenInstanceError`` is the precise exception raised
        # when assigning to a frozen dataclass attribute. Use ``setattr`` to
        # exercise the runtime path without needing a type-checker suppression.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            md.description = "changed"

    def test_tags_default_is_empty_tuple(self) -> None:
        md = ModelCardMetadata(description="x")
        self.assertEqual(md.tags, ())


# ---------------------------------------------------------------------------
# Additional tests for PR changes: tags rendering and metric precision
# ---------------------------------------------------------------------------


class TestWriteModelCardAdditional(TestCase):
    """Additional tests that extend coverage of the PR-specific changes to
    write_model_card (tags as comma-separated, metrics as numeric)."""

    def setUp(self) -> None:
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tmpdir = Path(self.tmp.name)

    def _fitted_estimator(self):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=50, solver="lbfgs")
        clf.fit([[0.0], [1.0]], [0, 1])
        return clf

    def test_single_tag_renders_without_comma(self) -> None:
        """A tuple with exactly one tag must render without a trailing comma."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", tags=("solo",))
        out = write_model_card(clf, self.tmpdir / "single_tag.md", metadata=md)
        content = out.read_text(encoding="utf-8")
        self.assertIn("solo", content)
        # The join of a single-element tuple produces no comma.
        # "solo," would indicate incorrect handling.
        self.assertNotIn("solo,", content)

    def test_three_tags_all_render(self) -> None:
        """All three tags in a 3-element tuple must appear in the card."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(
            description="x",
            tags=("legal", "nlp", "en"),
        )
        out = write_model_card(clf, self.tmpdir / "three_tags.md", metadata=md)
        content = out.read_text(encoding="utf-8")
        for tag in ("legal", "nlp", "en"):
            self.assertIn(tag, content)

    def test_empty_tags_no_tag_section(self) -> None:
        """When tags is empty the card is written without crashing."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x", tags=())
        out = write_model_card(clf, self.tmpdir / "no_tags.md", metadata=md)
        self.assertTrue(out.exists())

    def test_metrics_integer_value_renders(self) -> None:
        """Integer metric values must survive the no-string-coercion path."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x")
        out = write_model_card(
            clf,
            self.tmpdir / "int_metrics.md",
            metadata=md,
            metrics={"n_samples": 1000},
        )
        content = out.read_text(encoding="utf-8")
        self.assertIn("1000", content)

    def test_creates_nested_parent_directory(self) -> None:
        """write_model_card must create any missing parent directories."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x")
        deep_path = self.tmpdir / "a" / "b" / "c" / "card.md"
        out = write_model_card(clf, deep_path, metadata=md)
        self.assertTrue(out.exists())
        self.assertEqual(out.suffix, ".md")

    def test_suffix_normalised_to_md(self) -> None:
        """Passing a path without .md suffix must still produce a .md file."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x")
        out = write_model_card(clf, self.tmpdir / "card.txt", metadata=md)
        self.assertEqual(out.suffix, ".md")

    def test_multiple_metrics_all_render(self) -> None:
        """Every metric key and value must appear in the rendered card."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="x")
        metrics = {"precision": 0.9, "recall": 0.8, "f1": 0.85}
        out = write_model_card(clf, self.tmpdir / "multi.md", metadata=md, metrics=metrics)
        content = out.read_text(encoding="utf-8")
        for key, val in metrics.items():
            self.assertIn(key, content)
            self.assertIn(str(val), content)

    def test_no_metrics_does_not_crash(self) -> None:
        """write_model_card with metrics=None must succeed silently."""
        clf = self._fitted_estimator()
        md = ModelCardMetadata(description="bare")
        out = write_model_card(clf, self.tmpdir / "bare.md", metadata=md, metrics=None)
        self.assertTrue(out.exists())
