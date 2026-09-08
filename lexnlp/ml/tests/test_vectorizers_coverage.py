"""Coverage tests for ml vectorizers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
from gensim.models.doc2vec import Doc2Vec
from numpy import ndarray

from lexnlp.ml.vectorizers import Vectorizer, VectorizerDoc2Vec, VectorizerKeywordSearch


class _PassThrough(Vectorizer):
    def vectorize(self, tokens) -> ndarray:
        return super().vectorize(tokens)


def test_base_vectorize_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        _PassThrough().vectorize(["a"])


def test_keyword_search_init_stores_keywords_tuple() -> None:
    keywords = [("d", 1.0, 0.0), ("f", 2.0, -1.0)]
    vec = VectorizerKeywordSearch(keywords)
    assert vec.keywords == (("d", 1.0, 0.0), ("f", 2.0, -1.0))


def test_keyword_search_vectorize_marks_hits_and_misses() -> None:
    vec = VectorizerKeywordSearch([("d", 1.0, 0.0), ("f", 1.0, 0.0)])
    result = vec.vectorize(["a", "b", "c", "e", "f"])
    assert isinstance(result, ndarray)
    assert result.tolist() == [0.0, 1.0]


def test_doc2vec_init_with_instance_keeps_model() -> None:
    model = Doc2Vec(vector_size=5, min_count=1)
    vec = VectorizerDoc2Vec(model)
    assert vec.doc2vec is model


def test_load_doc2vec_with_instance_returns_same_object() -> None:
    model = Doc2Vec(vector_size=5, min_count=1)
    assert VectorizerDoc2Vec._load_doc2vec(model) is model


def test_load_doc2vec_with_pathlike_calls_doc2vec_load() -> None:
    sentinel = object()
    with mock.patch.object(Doc2Vec, "load", return_value=sentinel) as mock_load:
        result = VectorizerDoc2Vec._load_doc2vec(Path("some-model.bin"))
    mock_load.assert_called_once()
    assert result is sentinel


def test_load_doc2vec_with_invalid_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        VectorizerDoc2Vec._load_doc2vec("not-a-model")


def test_doc2vec_init_with_invalid_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        VectorizerDoc2Vec(12345)  # type: ignore[arg-type]
