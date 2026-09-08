"""Coverage tests for lexnlp.nlp.en.segments.titles.build_model and trailing titles."""

from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy
import pandas
import sklearn.ensemble

import lexnlp.nlp.en.segments.titles as titles_mod
from lexnlp.nlp.en.segments.titles import build_model, get_titles

TEXT_F1 = "LEASE AGREEMENT\nThis is the body of the lease.\nMore body text here.\n"
TEXT_F2 = "AMENDMENT\n\nThird line body text here.\n"
TEXT_F3 = "SKIP ME\nline one here\nline two here\nline three here\n"


def _training_frame() -> pandas.DataFrame:
    return pandas.DataFrame(
        [
            {"File": "https://github.com/owner/repo/blob/main/f1.txt", "Line Number": "1"},
            {"File": "https://github.com/owner/repo/blob/main/f2.txt", "Line Number": "1-2"},
            # Three target lines are skipped by the builder.
            {"File": "https://github.com/owner/repo/blob/main/f3.txt", "Line Number": "1-3"},
            # Null line numbers are filtered out before downloading.
            {"File": "https://github.com/owner/repo/blob/main/f4.txt", "Line Number": numpy.nan},
        ]
    )


class TestBuildModel:
    def test_builds_and_dumps_extra_trees_model(self, monkeypatch) -> None:
        texts = {"f1.txt": TEXT_F1, "f2.txt": TEXT_F2, "f3.txt": TEXT_F3}
        seen_urls: list[str] = []
        dumped: dict[str, object] = {}

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        def fake_get(url: str, timeout: int = 60) -> FakeResponse:
            seen_urls.append(url)
            for name, text in texts.items():
                if url.endswith(name):
                    return FakeResponse(text)
            raise AssertionError(f"unexpected url {url}")

        def fake_read_csv(*args: object, **kwargs: object) -> pandas.DataFrame:
            return _training_frame()

        def fake_dump(model: object, path: str) -> None:
            dumped["model"] = model
            dumped["path"] = path

        monkeypatch.setattr(titles_mod.pandas, "read_csv", fake_read_csv)
        monkeypatch.setattr(titles_mod.requests, "get", fake_get)
        monkeypatch.setattr(titles_mod.joblib, "dump", fake_dump)

        build_model("unused-training.csv")

        # Only the three rows with line numbers are downloaded; the null row
        # is filtered and the three-line range row is downloaded then skipped.
        assert len(seen_urls) == 3
        assert seen_urls[0] == "https://raw.githubusercontent.com/owner/repo/main/f1.txt"
        model = dumped["model"]
        assert isinstance(model, sklearn.ensemble.ExtraTreesClassifier)
        assert model.n_estimators == 25
        assert hasattr(model, "classes_")
        assert dumped["path"] == "title_locator.pickle"

    def test_skipped_ranges_and_blank_lines_are_not_training_targets(self, monkeypatch) -> None:
        captured: dict[str, object] = {}
        real_concat = pandas.concat

        def fake_fit(self: object, features: pandas.DataFrame, target: pandas.Series) -> object:
            captured["target_sum"] = int(target.sum())
            captured["rows"] = int(target.shape[0])
            return real_fit(self, features, target)

        real_fit = sklearn.ensemble.ExtraTreesClassifier.fit
        monkeypatch.setattr(sklearn.ensemble.ExtraTreesClassifier, "fit", fake_fit)
        monkeypatch.setattr(titles_mod.pandas, "read_csv", lambda *a, **k: _training_frame())

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        texts = {"f1.txt": TEXT_F1, "f2.txt": TEXT_F2, "f3.txt": TEXT_F3}
        monkeypatch.setattr(
            titles_mod.requests,
            "get",
            lambda url, timeout=60: FakeResponse(texts[url.rsplit("/", 1)[-1]]),
        )
        dumped: dict[str, object] = {}
        monkeypatch.setattr(joblib, "dump", lambda model, path: dumped.setdefault("model", model))

        build_model("unused-training.csv")

        # f1 contributes one target line; f2 contributes line 1 only because
        # line 2 is blank; f3 is skipped entirely.
        assert captured["target_sum"] == 2
        assert dumped["model"] is not None


class TestTrailingTitle:
    def test_title_at_end_of_document_is_yielded(self, monkeypatch) -> None:
        text = "This body line is not a title at all.\nAGREEMENT"
        fake_model = SimpleNamespace(
            predict_proba=lambda _features: numpy.array([[0.9, 0.1], [0.05, 0.95]])
        )
        monkeypatch.setattr(titles_mod, "SECTION_SEGMENTER_MODEL", fake_model)
        assert list(get_titles(text)) == ["AGREEMENT"]
