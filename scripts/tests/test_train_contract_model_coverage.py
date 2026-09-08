"""Coverage tests for scripts/train_contract_model.py."""

from __future__ import annotations

import io
import json
import pickle
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_contract_model


class TestParseArgs:
    def test_defaults(self) -> None:
        args = train_contract_model.parse_args([])
        assert args.baseline_tag == "pipeline/is-contract/0.1"
        assert args.candidate_tag == "pipeline/is-contract/0.2"
        assert args.positive_tags == list(train_contract_model.DEFAULT_POSITIVE_TAGS)
        assert args.negative_tags == list(train_contract_model.DEFAULT_NEGATIVE_TAGS)
        assert args.max_docs_per_tag == 800
        assert args.head_character_n == 4000
        assert args.validation_size == pytest.approx(0.2)
        assert args.random_state == 7
        assert list(args.estimators) == ["gaussian_nb", "logistic_regression", "random_forest"]
        assert args.max_workers == 4
        assert args.min_probability == pytest.approx(0.3)
        assert args.skip_quality_gate is False
        assert args.keep_candidate_on_failure is False
        assert args.force is False

    def test_max_docs_per_tag_must_be_positive(self) -> None:
        with pytest.raises(SystemExit):
            train_contract_model.parse_args(["--max-docs-per-tag", "0"])

    def test_head_character_n_must_be_positive(self) -> None:
        with pytest.raises(SystemExit):
            train_contract_model.parse_args(["--head-character-n", "-5"])

    def test_validation_size_bounds(self) -> None:
        with pytest.raises(SystemExit):
            train_contract_model.parse_args(["--validation-size", "0.04"])
        with pytest.raises(SystemExit):
            train_contract_model.parse_args(["--validation-size", "0.51"])
        assert train_contract_model.parse_args(["--validation-size", "0.05"]).validation_size == pytest.approx(0.05)
        assert train_contract_model.parse_args(["--validation-size", "0.5"]).validation_size == pytest.approx(0.5)

    def test_max_workers_must_be_positive(self) -> None:
        with pytest.raises(SystemExit):
            train_contract_model.parse_args(["--max-workers", "0"])

    def test_empty_positive_tags_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(train_contract_model, "DEFAULT_POSITIVE_TAGS", ())
        with pytest.raises(SystemExit):
            train_contract_model.parse_args([])

    def test_empty_negative_tags_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(train_contract_model, "DEFAULT_NEGATIVE_TAGS", ())
        with pytest.raises(SystemExit):
            train_contract_model.parse_args([])

    def test_explicit_tags_accepted(self) -> None:
        args = train_contract_model.parse_args(
            ["--positive-tags", "corpus/a/0.1", "--negative-tags", "corpus/b/0.1", "--estimators", "gaussian_nb"]
        )
        assert args.positive_tags == ["corpus/a/0.1"]
        assert args.negative_tags == ["corpus/b/0.1"]
        assert list(args.estimators) == ["gaussian_nb"]


class TestEnsureTagDownloaded:
    def test_returns_cached_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        expected = tmp_path / "model.bin"
        expected.write_bytes(b"x")
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", lambda tag: expected)
        calls: list[str] = []
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: calls.append(tag),
        )
        assert train_contract_model.ensure_tag_downloaded("pipeline/x/1.0") == expected
        assert calls == []

    def test_downloads_on_cache_miss(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        expected = tmp_path / "model.bin"
        expected.write_bytes(b"x")
        attempts = {"n": 0}

        def fake_get(tag: str) -> Path:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise FileNotFoundError(tag)
            return expected

        downloaded: list[str] = []
        monkeypatch.setattr("lexnlp.ml.catalog.get_path_from_catalog", fake_get)
        monkeypatch.setattr(
            "lexnlp.ml.catalog.download.download_github_release",
            lambda tag, prompt_user=False: downloaded.append(tag),
        )
        assert train_contract_model.ensure_tag_downloaded("pipeline/x/1.0") == expected
        assert downloaded == ["pipeline/x/1.0"]


def _make_fitted_pipeline():
    import numpy as np
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(0)
    values = rng.rand(20, 4)
    labels = [0, 1] * 10
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", GaussianNB())])
    pipeline.fit(values, labels)
    return pipeline


class TestPatchLegacyEstimatorAttributes:
    def test_sigma_only_gains_var_aliases(self) -> None:
        import numpy as np

        pipeline = _make_fitted_pipeline()
        estimator = pipeline.steps[-1][1]
        estimator.sigma_ = np.asarray(estimator.var_)
        del estimator.var_
        if hasattr(estimator, "variance_"):
            del estimator.variance_
        train_contract_model.patch_legacy_estimator_attributes(pipeline)
        assert hasattr(estimator, "var_")
        assert hasattr(estimator, "variance_")
        assert estimator.variance_ is estimator.var_

    def test_existing_var_kept_and_variance_added(self) -> None:
        import numpy as np

        pipeline = _make_fitted_pipeline()
        estimator = pipeline.steps[-1][1]
        assert hasattr(estimator, "var_")
        estimator.sigma_ = np.asarray(estimator.var_)
        if hasattr(estimator, "variance_"):
            del estimator.variance_
        original_var = estimator.var_
        train_contract_model.patch_legacy_estimator_attributes(pipeline)
        assert estimator.var_ is original_var
        assert estimator.variance_ is original_var

    def test_modern_estimator_without_sigma_gains_no_aliases(self) -> None:
        pipeline = _make_fitted_pipeline()
        estimator = pipeline.steps[-1][1]
        assert not hasattr(estimator, "sigma_")
        train_contract_model.patch_legacy_estimator_attributes(pipeline)
        assert not hasattr(estimator, "sigma_")
        assert not hasattr(estimator, "variance_")

    def test_estimator_without_sigma_untouched(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200))])
        pipeline.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        estimator = pipeline.steps[-1][1]
        train_contract_model.patch_legacy_estimator_attributes(pipeline)
        assert not hasattr(estimator, "sigma_")

    def test_transform_clip_branches(self) -> None:
        from sklearn.pipeline import Pipeline

        class _Step:
            def __init__(self, **attrs) -> None:
                for key, value in attrs.items():
                    setattr(self, key, value)

            def fit(self, X, y=None):  # noqa: ANN001, ANN202
                return self

            def transform(self, X):  # noqa: ANN001, ANN202
                return X

        with_clip = _Step(clip=True)
        without_clip_value = _Step(clip=0)
        without_clip_attr = _Step()
        estimator = _Step()
        pipeline = Pipeline(
            [
                ("a", with_clip),
                ("b", None),
                ("c", "passthrough"),
                ("d", without_clip_value),
                ("e", without_clip_attr),
                ("clf", estimator),
            ]
        )
        train_contract_model.patch_legacy_estimator_attributes(pipeline)
        assert with_clip.clip is True
        assert without_clip_value.clip == 0
        assert without_clip_attr.clip is False


class TestLoadPipelineForTag:
    def test_loads_and_patches_pipeline(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import numpy as np
        from cloudpickle import dump

        pipeline = _make_fitted_pipeline()
        estimator = pipeline.steps[-1][1]
        estimator.sigma_ = np.asarray(estimator.var_)
        del estimator.var_
        model_path = tmp_path / "model.cloudpickle"
        with model_path.open("wb") as handle:
            dump(pipeline, handle)
        monkeypatch.setattr(train_contract_model, "ensure_tag_downloaded", lambda tag: model_path)
        path, loaded = train_contract_model.load_pipeline_for_tag("pipeline/is-contract/0.1")
        assert path == model_path
        assert hasattr(loaded.steps[-1][1], "var_")
        assert hasattr(loaded.steps[-1][1], "variance_")


def _write_tar(path: Path, members: dict[str, bytes], *, mode: str = "w") -> Path:
    with tarfile.open(path, mode=mode) as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


class TestIterTextsFromArchive:
    def test_yields_txt_only_truncated(self, tmp_path: Path) -> None:
        archive = _write_tar(
            tmp_path / "corpus.tar",
            {
                "doc1.txt": b"hello world",
                "notes.md": b"not a text doc",
                "empty.txt": b"   \n  ",
                "UPPER.TXT": b"upper extension",
            },
        )
        texts = list(train_contract_model.iter_texts_from_archive(archive, max_docs=10, head_character_n=5))
        assert texts == ["hello", "upper"]

    def test_respects_max_docs(self, tmp_path: Path) -> None:
        archive = _write_tar(
            tmp_path / "corpus.tar",
            {f"doc{i}.txt": f"document number {i}".encode() for i in range(5)},
        )
        texts = list(train_contract_model.iter_texts_from_archive(archive, max_docs=2, head_character_n=4000))
        assert len(texts) == 2
        assert texts[0].startswith("document number 0")

    def test_compressed_archive_supported(self, tmp_path: Path) -> None:
        archive = _write_tar(tmp_path / "corpus.tar.gz", {"doc.txt": b"compressed hello"}, mode="w:gz")
        texts = list(train_contract_model.iter_texts_from_archive(archive, max_docs=10, head_character_n=4000))
        assert texts == ["compressed hello"]

    def test_extractfile_none_is_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        archive = _write_tar(tmp_path / "corpus.tar", {"ghost.txt": b"unreadable", "real.txt": b"readable"})

        real_open = tarfile.open

        class _GhostArchive:
            def __init__(self, inner) -> None:
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *exc) -> bool:
                return self._inner.__exit__(*exc)

            def __iter__(self):
                return iter(self._inner.getmembers())

            def extractfile(self, member):
                if member.name == "ghost.txt":
                    return None
                return self._inner.extractfile(member)

        def fake_open(path, mode="r:*"):
            return _GhostArchive(real_open(path, mode=mode))

        monkeypatch.setattr(tarfile, "open", fake_open)
        texts = list(train_contract_model.iter_texts_from_archive(archive, max_docs=10, head_character_n=4000))
        assert texts == ["readable"]


class TestCollectCorpusSamples:
    def test_collects_labels_and_counts(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pos = _write_tar(tmp_path / "pos.tar", {"a.txt": b"contract one", "b.txt": b"contract two"})
        neg = _write_tar(tmp_path / "neg.tar", {"c.txt": b"patent one"})
        paths = {"corpus/pos/0.1": pos, "corpus/neg/0.1": neg}
        monkeypatch.setattr(train_contract_model, "ensure_tag_downloaded", lambda tag: paths[tag])
        texts, labels, counts = train_contract_model.collect_corpus_samples(
            ["corpus/pos/0.1"], label=True, max_docs_per_tag=10, head_character_n=4000
        )
        assert texts == ["contract one", "contract two"]
        assert labels == [True, True]
        assert counts == {"corpus/pos/0.1": 2}

        texts, labels, counts = train_contract_model.collect_corpus_samples(
            ["corpus/neg/0.1"], label=False, max_docs_per_tag=10, head_character_n=4000
        )
        assert labels == [False]
        assert counts == {"corpus/neg/0.1": 1}

    def test_empty_archive_raises_training_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        empty = _write_tar(tmp_path / "empty.tar", {"notes.md": b"no txt here"})
        monkeypatch.setattr(train_contract_model, "ensure_tag_downloaded", lambda tag: empty)
        with pytest.raises(train_contract_model.TrainingError, match="No text samples extracted"):
            train_contract_model.collect_corpus_samples(
                ["corpus/empty/0.1"], label=True, max_docs_per_tag=10, head_character_n=4000
            )


class TestMakeEstimator:
    def test_gaussian_nb(self) -> None:
        from sklearn.naive_bayes import GaussianNB

        estimator = train_contract_model.make_estimator("gaussian_nb", random_state=7, max_workers=2)
        assert isinstance(estimator, GaussianNB)

    def test_logistic_regression_params(self) -> None:
        from sklearn.linear_model import LogisticRegression

        estimator = train_contract_model.make_estimator("logistic_regression", random_state=7, max_workers=2)
        assert isinstance(estimator, LogisticRegression)
        assert estimator.class_weight == "balanced"
        assert estimator.max_iter == 600
        assert estimator.random_state == 7

    def test_random_forest_params(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        estimator = train_contract_model.make_estimator("random_forest", random_state=7, max_workers=3)
        assert isinstance(estimator, RandomForestClassifier)
        assert estimator.n_estimators == 300
        assert estimator.class_weight == "balanced_subsample"
        assert estimator.min_samples_leaf == 2
        assert estimator.random_state == 7
        assert estimator.n_jobs == 3

    def test_unsupported_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported estimator"):
            train_contract_model.make_estimator("svm", random_state=7, max_workers=1)


class TestBuildCandidatePipeline:
    def test_steps_deep_copied_and_estimator_appended(self) -> None:
        from sklearn.feature_extraction.text import CountVectorizer

        steps = [("vec", CountVectorizer())]
        pipeline = train_contract_model.build_candidate_pipeline(
            steps, "logistic_regression", random_state=7, max_workers=1
        )
        assert [name for name, _ in pipeline.steps] == ["vec", "logistic_regression"]
        assert pipeline.steps[0][1] is not steps[0][1]
        assert pipeline.steps[0][1].get_params() == steps[0][1].get_params()

    def test_built_pipeline_fits_and_predicts(self) -> None:
        from sklearn.feature_extraction.text import CountVectorizer

        pipeline = train_contract_model.build_candidate_pipeline(
            [("vec", CountVectorizer())], "logistic_regression", random_state=7, max_workers=1
        )
        pipeline.fit(["contract agreement", "patent widget", "contract sale", "weather sunny"], [1, 0, 1, 0])
        assert list(pipeline.predict(["contract deal", "rainy weather"])) is not None


class TestScorePipeline:
    def test_metrics_match_sklearn(self) -> None:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=500))])
        train_texts = ["contract agreement parties", "contract sale goods", "patent widget process", "weather sunny"]
        pipeline.fit(train_texts, [True, True, False, False])
        texts = ["contract obligations", "widget manufacturing", "agreement signed", "rain forecast"]
        labels = [True, False, True, False]
        result = train_contract_model.score_pipeline(pipeline, texts, labels, min_probability=0.3)
        probabilities = pipeline.predict_proba(texts)[:, 1]
        expected = probabilities >= 0.3
        assert result["accuracy"] == float(accuracy_score(labels, expected))
        assert result["f1"] == float(f1_score(labels, expected))
        assert result["precision"] == float(precision_score(labels, expected, zero_division=0))
        assert result["recall"] == float(recall_score(labels, expected, zero_division=0))


class TestChooseBest:
    def test_selects_highest_f1(self) -> None:
        scores = {
            "a": {"f1": 0.5, "accuracy": 0.9, "precision": 0.9, "recall": 0.9},
            "b": {"f1": 0.8, "accuracy": 0.6, "precision": 0.6, "recall": 0.6},
        }
        assert train_contract_model.choose_best(scores) == "b"

    def test_tie_broken_by_accuracy_then_precision_then_recall(self) -> None:
        scores = {
            "a": {"f1": 0.7, "accuracy": 0.6, "precision": 0.9, "recall": 0.9},
            "b": {"f1": 0.7, "accuracy": 0.8, "precision": 0.1, "recall": 0.1},
        }
        assert train_contract_model.choose_best(scores) == "b"
        scores = {
            "a": {"f1": 0.7, "accuracy": 0.8, "precision": 0.4, "recall": 0.9},
            "b": {"f1": 0.7, "accuracy": 0.8, "precision": 0.6, "recall": 0.1},
        }
        assert train_contract_model.choose_best(scores) == "b"
        scores = {
            "a": {"f1": 0.7, "accuracy": 0.8, "precision": 0.6, "recall": 0.2},
            "b": {"f1": 0.7, "accuracy": 0.8, "precision": 0.6, "recall": 0.9},
        }
        assert train_contract_model.choose_best(scores) == "b"


def _make_small_text_pipeline():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("vec", CountVectorizer()), ("clf", LogisticRegression(max_iter=500))])
    pipeline.fit(
        ["contract agreement parties", "contract sale goods", "patent widget process", "weather sunny"],
        [1, 0, 1, 0],
    )
    return pipeline


class TestWriteCandidateToCatalog:
    def test_writes_pickled_pipeline(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import lexnlp.ml.catalog as catalog

        monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog")
        pipeline = _make_small_text_pipeline()
        destination = train_contract_model.write_candidate_to_catalog(
            baseline_model_path=Path("models/model.cloudpickle"),
            candidate_tag="pipeline/cand/0.2",
            pipeline=pipeline,
            force=False,
        )
        assert destination == tmp_path / "catalog" / "pipeline/cand/0.2" / "model.cloudpickle"
        with destination.open("rb") as handle:
            loaded = pickle.load(handle)
        assert list(loaded.predict(["contract deal"])) == list(pipeline.predict(["contract deal"]))

    def test_exists_without_force_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import lexnlp.ml.catalog as catalog

        monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog")
        pipeline = _make_small_text_pipeline()
        kwargs = {
            "baseline_model_path": Path("models/model.cloudpickle"),
            "candidate_tag": "pipeline/cand/0.2",
            "pipeline": pipeline,
        }
        train_contract_model.write_candidate_to_catalog(force=False, **kwargs)
        with pytest.raises(FileExistsError, match="already exists"):
            train_contract_model.write_candidate_to_catalog(force=False, **kwargs)

    def test_force_overwrites_existing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import lexnlp.ml.catalog as catalog

        monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog")
        destination_dir = tmp_path / "catalog" / "pipeline/cand/0.2"
        destination_dir.mkdir(parents=True)
        existing = destination_dir / "model.cloudpickle"
        existing.write_bytes(b"stale")
        pipeline = _make_small_text_pipeline()
        destination = train_contract_model.write_candidate_to_catalog(
            baseline_model_path=Path("models/model.cloudpickle"),
            candidate_tag="pipeline/cand/0.2",
            pipeline=pipeline,
            force=True,
        )
        assert destination == existing
        assert existing.read_bytes() != b"stale"


class TestRunQualityGate:
    def _call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_metrics_file: bool) -> dict[str, object]:
        seen: dict[str, object] = {}

        def fake_run(cmd, *, check, capture_output, text):
            seen["cmd"] = cmd
            seen["check"] = check
            seen["capture_output"] = capture_output
            seen["text"] = text
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        fixture = tmp_path / "fix.csv"
        fixture.write_text("Text,Is_Contract\n", encoding="utf-8")
        metrics_path = tmp_path / "baseline.json"
        if with_metrics_file:
            metrics_path.write_text("{}", encoding="utf-8")
        train_contract_model.run_quality_gate(
            baseline_tag="pipeline/base/0.1",
            candidate_tag="pipeline/cand/0.2",
            fixture=fixture,
            baseline_metrics_json=metrics_path,
            min_probability=0.3,
            max_accuracy_regression=0.0,
            max_f1_regression=0.1,
        )
        return seen

    def test_cmd_without_metrics_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._call(tmp_path, monkeypatch, with_metrics_file=False)
        cmd = seen["cmd"]
        assert isinstance(cmd, list)
        assert cmd[0] == sys.executable
        assert cmd[1].endswith("model_quality_gate.py")
        assert "--baseline-tag" in cmd and "pipeline/base/0.1" in cmd
        assert "--candidate-tag" in cmd and "pipeline/cand/0.2" in cmd
        assert "--baseline-metrics-json" not in cmd
        assert seen["check"] is True
        assert seen["capture_output"] is True
        assert seen["text"] is True

    def test_cmd_with_metrics_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._call(tmp_path, monkeypatch, with_metrics_file=True)
        cmd = seen["cmd"]
        assert isinstance(cmd, list)
        flag_index = cmd.index("--baseline-metrics-json")
        assert cmd[flag_index + 1].endswith("baseline.json")


def _to_dense(matrix):
    return matrix.toarray()


def _make_dense_baseline_pipeline():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    pipeline = Pipeline(
        [
            ("vec", CountVectorizer()),
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipeline.fit(
        ["contract agreement parties", "contract sale goods", "patent widget process", "weather sunny"],
        [1, 0, 1, 0],
    )
    return pipeline


def _install_main_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import lexnlp.ml.catalog as catalog

    baseline = _make_dense_baseline_pipeline()
    monkeypatch.setattr(
        train_contract_model,
        "load_pipeline_for_tag",
        lambda tag: (tmp_path / "baseline" / "model.cloudpickle", baseline),
    )
    positive = [f"contract agreement sale goods number {i}" for i in range(6)]
    negative = [f"patent widget manufacturing process number {i}" for i in range(6)]

    def fake_collect(tags, *, label, max_docs_per_tag, head_character_n):
        if label:
            return list(positive), [True] * len(positive), {tag: len(positive) for tag in tags}
        return list(negative), [False] * len(negative), {tag: len(negative) for tag in tags}

    monkeypatch.setattr(train_contract_model, "collect_corpus_samples", fake_collect)
    monkeypatch.setattr(catalog, "CATALOG", tmp_path / "catalog")
    return baseline


class TestMain:
    def _argv(self, tmp_path: Path, extra: list[str] | None = None) -> list[str]:
        argv = [
            "--positive-tags",
            "corpus/pos/0.1",
            "--negative-tags",
            "corpus/neg/0.1",
            "--output-json",
            str(tmp_path / "report.json"),
        ]
        if extra:
            argv.extend(extra)
        return argv

    def test_skip_quality_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_main_mocks(monkeypatch, tmp_path)
        rc = train_contract_model.main(self._argv(tmp_path, ["--skip-quality-gate"]))
        assert rc == 0
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["baseline_tag"] == "pipeline/is-contract/0.1"
        assert report["candidate_tag"] == "pipeline/is-contract/0.2"
        assert report["selected_estimator"] in ("gaussian_nb", "logistic_regression", "random_forest")
        assert set(report["estimators"]) == {"gaussian_nb", "logistic_regression", "random_forest"}
        for scores in report["estimators"].values():
            assert set(scores) == {"accuracy", "f1", "precision", "recall"}
        assert report["dataset"]["total_samples"] == 12
        assert report["dataset"]["train_samples"] == 9
        assert report["dataset"]["validation_samples"] == 3
        assert report["dataset"]["positive_counts"] == {"corpus/pos/0.1": 6}
        assert report["dataset"]["negative_counts"] == {"corpus/neg/0.1": 6}
        assert report["quality_gate"] == {"skipped": True, "status": "skipped"}
        assert set(report["validation_baseline"]) == {"accuracy", "f1", "precision", "recall"}
        assert set(report["validation_candidate_selected"]) == {"accuracy", "f1", "precision", "recall"}
        candidate = Path(report["candidate_model_path"])
        assert candidate.exists()

    def test_quality_gate_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_main_mocks(monkeypatch, tmp_path)
        seen: dict[str, object] = {}
        monkeypatch.setattr(train_contract_model, "run_quality_gate", lambda **kwargs: seen.update(kwargs) or None)
        rc = train_contract_model.main(self._argv(tmp_path))
        assert rc == 0
        assert seen["baseline_tag"] == "pipeline/is-contract/0.1"
        assert seen["candidate_tag"] == "pipeline/is-contract/0.2"
        assert seen["min_probability"] == pytest.approx(0.3)
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["quality_gate"] == {"skipped": False, "status": "passed"}
        printed = json.loads(capsys.readouterr().out)
        assert printed["quality_gate"]["status"] == "passed"

    def test_quality_gate_failed_removes_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_main_mocks(monkeypatch, tmp_path)
        error = subprocess.CalledProcessError(2, ["gate"], output="gate out", stderr="gate err")

        def fake_gate(**kwargs) -> None:
            raise error

        monkeypatch.setattr(train_contract_model, "run_quality_gate", fake_gate)
        rc = train_contract_model.main(self._argv(tmp_path))
        assert rc == 1
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["quality_gate"]["status"] == "failed"
        assert report["quality_gate"]["returncode"] == 2
        assert report["quality_gate"]["stdout"] == "gate out"
        assert report["quality_gate"]["stderr"] == "gate err"
        assert report["quality_gate"]["candidate_removed"] is True
        assert not Path(report["candidate_model_path"]).exists()
        assert "gate err" in capsys.readouterr().err

    def test_quality_gate_failed_bytes_output_kept_on_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_main_mocks(monkeypatch, tmp_path)
        error = subprocess.CalledProcessError(4, ["gate"], output=b"raw out", stderr=b"raw err")

        def fake_gate(**kwargs) -> None:
            raise error

        monkeypatch.setattr(train_contract_model, "run_quality_gate", fake_gate)
        rc = train_contract_model.main(self._argv(tmp_path, ["--keep-candidate-on-failure"]))
        assert rc == 1
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["quality_gate"]["stdout"] == "raw out"
        assert report["quality_gate"]["stderr"] == "raw err"
        assert "candidate_removed" not in report["quality_gate"]
        assert Path(report["candidate_model_path"]).exists()

    def test_quality_gate_failed_without_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _install_main_mocks(monkeypatch, tmp_path)
        error = subprocess.CalledProcessError(3, ["gate"])

        def fake_gate(**kwargs) -> None:
            raise error

        monkeypatch.setattr(train_contract_model, "run_quality_gate", fake_gate)
        rc = train_contract_model.main(self._argv(tmp_path))
        assert rc == 1
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["quality_gate"]["status"] == "failed"
        assert "stdout" not in report["quality_gate"]
        assert "stderr" not in report["quality_gate"]
        assert capsys.readouterr().err == ""

    def test_no_feature_steps_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        single = Pipeline([("clf", LogisticRegression(max_iter=50))])
        single.fit([[0.0], [1.0]], [0, 1])
        monkeypatch.setattr(
            train_contract_model,
            "load_pipeline_for_tag",
            lambda tag: (tmp_path / "model.cloudpickle", single),
        )
        with pytest.raises(train_contract_model.TrainingError, match="no feature steps"):
            train_contract_model.main(self._argv(tmp_path))


class TestMainGuard:
    def test_run_as_main_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import runpy

        monkeypatch.setattr(sys, "argv", ["train_contract_model.py", "--max-docs-per-tag", "0"])
        with pytest.raises(SystemExit):
            runpy.run_path(str(_SCRIPTS_DIR / "train_contract_model.py"), run_name="__main__")
