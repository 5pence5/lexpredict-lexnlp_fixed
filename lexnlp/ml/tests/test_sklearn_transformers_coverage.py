__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import numpy
import pytest
from scipy.sparse import csr_matrix

from lexnlp.ml.normalizers import Normalizer
from lexnlp.ml.sklearn_transformers import (
    TransformerPreprocessor,
    TransformerVectorizer,
    _predict,
    parallel_estimator,
)
from lexnlp.ml.vectorizers import VectorizerKeywordSearch


class _DenseEstimator:
    def predict(self, chunk):
        return numpy.asarray(chunk) * 2


class _SparseEstimator:
    def predict(self, chunk):
        rows = len(chunk)
        return csr_matrix(numpy.ones((rows, 2)))


class TestPredict:
    def test_predict_calls_named_method_on_slice(self) -> None:
        estimator = _DenseEstimator()
        result = _predict(estimator, [1, 2, 3, 4], "predict", 1, 3)
        assert list(result) == [4, 6]


class TestParallelEstimator:
    def test_dense_results_are_concatenated(self) -> None:
        data = numpy.arange(10)
        result = parallel_estimator(_DenseEstimator(), data, "predict", n_jobs=2)
        assert isinstance(result, numpy.ndarray)
        assert list(result) == list(data * 2)

    def test_sparse_results_are_vstacked(self) -> None:
        data = numpy.arange(10)
        result = parallel_estimator(_SparseEstimator(), data, "predict", n_jobs=2)
        assert result.shape == (10, 2)
        assert result.sum() == 20

    def test_single_job_matches_direct_call(self) -> None:
        data = numpy.arange(5)
        result = parallel_estimator(_DenseEstimator(), data, "predict", n_jobs=1)
        assert list(result) == [0, 2, 4, 6, 8]


class TestTransformerVectorizer:
    def test_init_stores_vectorizers_as_tuple(self) -> None:
        vectorizers = [VectorizerKeywordSearch([("cat", 1.0, 0.0)])]
        transformer = TransformerVectorizer(vectorizers)
        assert isinstance(transformer.vectorizers, tuple)
        assert len(transformer.vectorizers) == 1

    def test_fit_returns_self(self) -> None:
        transformer = TransformerVectorizer([VectorizerKeywordSearch([("cat", 1.0, 0.0)])])
        assert transformer.fit(["some text"]) is transformer

    def test_transform_concatenates_vectorizer_outputs(self) -> None:
        transformer = TransformerVectorizer(
            [
                VectorizerKeywordSearch([("cat", 1.0, 0.0)]),
                VectorizerKeywordSearch([("dog", 1.0, 0.0)]),
            ]
        )
        vectors = transformer.transform(["cat sat", "dog ran"])
        assert len(vectors) == 2
        assert list(vectors[0]) == [1.0, 0.0]
        assert list(vectors[1]) == [0.0, 1.0]


class TestTransformerPreprocessorInit:
    def test_init_stores_normalizer_and_head(self) -> None:
        normalizer = Normalizer(normalizations=[])
        preprocessor = TransformerPreprocessor(normalizer, head_character_n=50)
        assert preprocessor.normalizer is normalizer
        assert preprocessor.head_character_n == 50

    def test_init_defaults_head_to_zero(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]))
        assert preprocessor.head_character_n == 0

    def test_negative_head_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="head_character_n"):
            TransformerPreprocessor(Normalizer(normalizations=[]), head_character_n=-1)

    def test_fit_returns_self(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]))
        assert preprocessor.fit(["some text"]) is preprocessor


class TestTransformerPreprocessorText:
    def test_handle_block_text_without_head_yields_all_sentences(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]))
        sentences = list(preprocessor._handle_block_text("Hello world. Second sentence here."))
        assert sentences == ["Hello world.", "Second sentence here."]

    def test_preprocess_block_text_without_head(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]))
        result = preprocessor.preprocess("The cat sat.")
        assert "cat" in result
        assert "the" not in result.split()

    def test_preprocess_iterable_with_head_limits_sentences(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]), head_character_n=10)
        result = preprocessor.preprocess(["The cat sat quietly here.", "Dogs bark loudly outside."])
        assert "cat" in result
        assert "dog" not in result and "bark" not in result

    def test_preprocess_block_text_with_head_limits_sentences(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]), head_character_n=10)
        result = preprocessor.preprocess("The cat sat quietly here. Dogs bark loudly outside.")
        assert "cat" in result
        assert "bark" not in result

    def test_transform_applies_preprocess_to_each_document(self) -> None:
        preprocessor = TransformerPreprocessor(Normalizer(normalizations=[]))
        result = preprocessor.transform(["The cat sat.", "Dogs bark."])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert "cat" in result[0]
        assert "dog" in result[1]
