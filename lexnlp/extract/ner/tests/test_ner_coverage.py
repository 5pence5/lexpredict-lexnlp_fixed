__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nltk.tree import Tree

from lexnlp.extract.ner import (
    HybridNERMatch,
    _nltk_extract,
    _overlap_ratio,
    _resolve_spacy_model_name,
    _spacy_extract,
    extract_entities,
    spacy_is_available,
)


def _fake_spacy_doc():
    return SimpleNamespace(
        ents=[
            SimpleNamespace(start_char=0, end_char=8, text="John Doe", label_="PERSON"),
            SimpleNamespace(start_char=20, end_char=29, text="Acme Corp", label_="ORG"),
        ]
    )


class TestSpacyIsAvailable:
    def test_true_when_spacy_importable(self) -> None:
        with patch.dict("sys.modules", {"spacy": MagicMock()}):
            assert spacy_is_available() is True


class TestResolveSpacyModelName:
    def test_default_model_name(self, monkeypatch) -> None:
        monkeypatch.delenv("LEXNLP_SPACY_MODEL", raising=False)
        assert _resolve_spacy_model_name() == "en_core_web_sm"

    def test_env_var_overrides_default(self, monkeypatch) -> None:
        monkeypatch.setenv("LEXNLP_SPACY_MODEL", "en_core_web_md")
        assert _resolve_spacy_model_name() == "en_core_web_md"


class TestSpacyExtract:
    def test_returns_spacy_backend_matches(self) -> None:
        doc = _fake_spacy_doc()
        with patch(
            "lexnlp.extract.ml.classifier.spacy_token_sequence_model._load_spacy_pipeline",
            return_value=lambda text: doc,
        ):
            matches = _spacy_extract("John Doe works at Acme Corp")
        assert len(matches) == 2
        first, second = matches
        assert first == HybridNERMatch(start=0, end=8, text="John Doe", label="PERSON", backend="spacy")
        assert second == HybridNERMatch(start=20, end=29, text="Acme Corp", label="ORG", backend="spacy")
        assert all(m.backend == "spacy" for m in matches)

    def test_no_entities_returns_empty_list(self) -> None:
        doc = SimpleNamespace(ents=[])
        with patch(
            "lexnlp.extract.ml.classifier.spacy_token_sequence_model._load_spacy_pipeline",
            return_value=lambda text: doc,
        ):
            assert _spacy_extract("nothing to see here") == []


class TestNltkExtractEdgeCases:
    def test_empty_text_returns_empty_list(self) -> None:
        assert _nltk_extract("") == []

    def test_empty_chunk_is_skipped(self) -> None:
        with (
            patch("nltk.ne_chunk", return_value=[Tree("PERSON", [])]),
        ):
            assert _nltk_extract("hello") == []

    def test_oversized_chunk_is_skipped(self) -> None:
        big = Tree("PERSON", [(f"w{i}", "NN") for i in range(5)])
        with patch("nltk.ne_chunk", return_value=[big]):
            assert _nltk_extract("hello") == []


class TestExtractEntitiesSpacyBranch:
    def test_prefer_spacy_returns_spacy_matches(self) -> None:
        sentinel = [HybridNERMatch(start=0, end=4, text="John", label="PERSON", backend="spacy")]
        with (
            patch("lexnlp.extract.ner.spacy_is_available", return_value=True),
            patch("lexnlp.extract.ner._spacy_extract", return_value=sentinel) as mock_extract,
        ):
            result = extract_entities("John went home.", prefer_spacy=True)
        assert result is sentinel
        mock_extract.assert_called_once_with("John went home.")

    def test_missing_spacy_model_falls_back_to_nltk(self) -> None:
        sentinel = [HybridNERMatch(start=0, end=4, text="John", label="PERSON", backend="nltk")]
        with (
            patch("lexnlp.extract.ner.spacy_is_available", return_value=True),
            patch("lexnlp.extract.ner._spacy_extract", side_effect=OSError("Can't find model")),
            patch("lexnlp.extract.ner._nltk_extract", return_value=sentinel) as mock_nltk,
        ):
            result = extract_entities("John went home.", prefer_spacy=True)
        assert result is sentinel
        mock_nltk.assert_called_once_with("John went home.")


class TestOverlapRatio:
    def test_zero_length_span_returns_zero(self) -> None:
        assert _overlap_ratio((5, 5), (0, 10)) == 0.0

    def test_both_zero_length_returns_zero(self) -> None:
        assert _overlap_ratio((3, 3), (3, 3)) == 0.0

    def test_half_overlap_returns_half(self) -> None:
        assert _overlap_ratio((0, 10), (5, 15)) == 0.5
