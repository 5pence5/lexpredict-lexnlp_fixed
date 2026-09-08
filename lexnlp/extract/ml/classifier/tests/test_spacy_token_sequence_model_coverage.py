"""Coverage tests for lexnlp/extract/ml/classifier/spacy_token_sequence_model.py.

``spacy`` is an optional dependency (``lexnlp[ner]`` extra) and is not
installed in this environment, so the tests install a minimal fake ``spacy``
module (recording ``load`` calls) and a fake parsed ``Doc``. Only the model
download / ``spacy`` import is mocked; all feature-list and feature-data
logic runs for real.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Iterator

import pytest

from lexnlp.extract.ml.classifier import spacy_token_sequence_model as spacy_model
from lexnlp.extract.ml.classifier.spacy_token_sequence_model import SpacyTokenSequenceClassifierModel


class FakeToken:
    def __init__(
        self,
        text: str,
        idx: int,
        *,
        pos: str = "NOUN",
        tag: str = "NN",
        dep: str = "nsubj",
        is_title: bool = False,
        is_lower: bool = False,
        is_upper: bool = False,
    ) -> None:
        self._text = text
        self.idx = idx
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.is_title = is_title
        self.is_lower = is_lower
        self.is_upper = is_upper

    def __str__(self) -> str:
        return self._text

    def __len__(self) -> int:
        return len(self._text)


class FakeDoc:
    def __init__(self, tokens: list[FakeToken]) -> None:
        self._tokens = tokens

    def __len__(self) -> int:
        return len(self._tokens)

    def __iter__(self) -> Iterator[FakeToken]:
        return iter(self._tokens)


def _install_fake_spacy(monkeypatch: pytest.MonkeyPatch, *, fail_load: BaseException | None = None):
    """Install a fake ``spacy`` module; return (module, calls, pipeline)."""
    calls: list[str] = []

    def fake_load(name: str):
        calls.append(name)
        if fail_load is not None:
            raise fail_load
        return pipeline

    def pipeline(text: str) -> FakeDoc:
        raise AssertionError("pipeline must be replaced by the test")

    module = types.ModuleType("spacy")
    module.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", module)
    spacy_model._load_spacy_pipeline.cache_clear()
    return module, calls, pipeline


@pytest.fixture
def clear_spacy_cache():
    spacy_model._load_spacy_pipeline.cache_clear()
    yield
    spacy_model._load_spacy_pipeline.cache_clear()


class TestModuleConstants:
    def test_default_model_name(self) -> None:
        assert spacy_model.DEFAULT_SPACY_MODEL == "en_core_web_sm"

    def test_module_path_points_at_package(self) -> None:
        assert os.path.isdir(spacy_model.MODULE_PATH)
        assert spacy_model.MODULE_PATH.endswith("classifier")

    def test_pos_tag_dep_lists(self) -> None:
        assert "NOUN" in spacy_model.SPACY_POS_LIST
        assert "VERB" in spacy_model.SPACY_POS_LIST
        assert "NNP" in spacy_model.SPACY_TAG_LIST
        assert "VBZ" in spacy_model.SPACY_TAG_LIST
        assert "punct" in spacy_model.SPACY_DEP_LIST
        assert "nsubj" in spacy_model.SPACY_DEP_LIST


class TestResolveSpacyModelName:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LEXNLP_SPACY_MODEL", raising=False)
        assert spacy_model._resolve_spacy_model_name() == "en_core_web_sm"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LEXNLP_SPACY_MODEL", "en_core_web_md")
        assert spacy_model._resolve_spacy_model_name() == "en_core_web_md"


class TestLoadSpacyPipeline:
    def test_loads_and_caches(self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None) -> None:
        _, calls, _ = _install_fake_spacy(monkeypatch)
        first = spacy_model._load_spacy_pipeline("en_core_web_sm")
        second = spacy_model._load_spacy_pipeline("en_core_web_sm")
        assert first is second
        assert calls == ["en_core_web_sm"]

    def test_default_name_from_env(self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None) -> None:
        _, calls, _ = _install_fake_spacy(monkeypatch)
        monkeypatch.setenv("LEXNLP_SPACY_MODEL", "en_core_web_md")
        spacy_model._load_spacy_pipeline()
        assert calls == ["en_core_web_md"]

    def test_missing_spacy_raises_actionable_import_error(
        self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None
    ) -> None:
        monkeypatch.setitem(sys.modules, "spacy", None)
        spacy_model._load_spacy_pipeline.cache_clear()
        with pytest.raises(ImportError, match="optional dependency"):
            spacy_model._load_spacy_pipeline("en_core_web_sm")

    def test_load_failure_propagates(self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None) -> None:
        _install_fake_spacy(monkeypatch, fail_load=OSError("model files were never downloaded"))
        with pytest.raises(OSError, match="never downloaded"):
            spacy_model._load_spacy_pipeline("en_core_web_sm")


class TestNlpEnAccessors:
    def test_nlp_en_returns_pipeline(self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None) -> None:
        sentinel = object()
        calls: list[str] = []
        module = types.ModuleType("spacy")
        module.load = lambda name: calls.append(name) or sentinel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "spacy", module)
        spacy_model._load_spacy_pipeline.cache_clear()
        assert spacy_model._NLP_EN() is sentinel
        assert calls == ["en_core_web_sm"]

    def test_nlp_en_lambda_calls_pipeline(self, monkeypatch: pytest.MonkeyPatch, clear_spacy_cache: None) -> None:
        seen: list[str] = []

        def fake_pipeline(text: str) -> str:
            seen.append(text)
            return f"doc:{text}"

        module, _, _ = _install_fake_spacy(monkeypatch)
        module.load = lambda name: fake_pipeline  # type: ignore[attr-defined]
        assert spacy_model.NLP_EN("hello") == "doc:hello"
        assert seen == ["hello"]


def _make_model(**kwargs) -> SpacyTokenSequenceClassifierModel:
    return SpacyTokenSequenceClassifierModel(
        letter_set=["A", "a", "b"],
        digit_set=["7"],
        punc_set=["."],
        symbol_set=["$"],
        match_tokens=["Hi"],
        **kwargs,
    )


class TestInit:
    def test_defaults_stored(self) -> None:
        model = SpacyTokenSequenceClassifierModel()
        assert model.letter_set == []
        assert model.digit_set == []
        assert model.pre_window == 0
        assert model.post_window == 0
        assert model.calculate_sum is False
        assert model.string_checks is False
        assert len(model.feature_list) > 0
        assert model._feature_index_map["position"] == model.feature_list.index("position")

    def test_explicit_args_stored(self) -> None:
        model = _make_model(pre_window=1, post_window=1, string_checks=True, calculate_sum=True)
        assert model.pre_window == 1
        assert model.post_window == 1
        assert model.string_checks is True
        assert model.calculate_sum is True
        assert model.match_tokens == ["Hi"]


class TestGetFeatureList:
    def test_core_and_spacy_features(self) -> None:
        features = _make_model().get_feature_list()
        assert features[:3] == ["position", "length", "mask"]
        assert "is_pos_NOUN" in features
        assert "is_pos_other" in features
        assert "is_tag_NN" in features
        assert "is_tag_other" in features
        assert "is_dep_nsubj" in features
        assert "is_dep_other" in features

    def test_window_zero_offsets_without_string_checks(self) -> None:
        features = _make_model(string_checks=False).get_feature_list()
        assert "0_is_start" in features
        assert "0_is_end" in features
        assert "0_is_title" not in features
        assert "0_is_lower" not in features
        assert "0_is_upper" not in features
        assert "-1_is_start" not in features
        assert "1_is_start" not in features

    def test_string_checks_add_title_case_features(self) -> None:
        features = _make_model(string_checks=True).get_feature_list()
        assert "0_is_title" in features
        assert "0_is_lower" in features
        assert "0_is_upper" in features

    def test_window_offsets(self) -> None:
        features = _make_model(pre_window=1, post_window=2).get_feature_list()
        assert "-1_is_start" in features
        assert "0_is_start" in features
        assert "1_is_start" in features
        assert "2_is_end" in features
        assert "-2_is_start" not in features
        assert "3_is_start" not in features

    def test_character_features(self) -> None:
        features = _make_model().get_feature_list()
        for name in (
            "0_char_A",
            "0_lchar_a",
            "0_first_char_A",
            "0_first_lchar_a",
            "0_last_char_A",
            "0_last_lchar_a",
            "0_digit_7",
            "0_first_digit_7",
            "0_last_digit_7",
            "0_punc_.",
            "0_first_punc_.",
            "0_last_punc_.",
            "0_symbol_$",
            "0_first_symbol_$",
            "0_last_symbol_$",
            "0_char_other",
            "0_first_char_other",
            "0_last_char_other",
        ):
            assert name in features, name
        assert "0_token_Hi" in features

    def test_unicode_category_features_cover_model_sets(self) -> None:
        model = _make_model()
        features = model.get_feature_list()
        for category in model.unicode_category_set:
            assert f"0_cat_{category}" in features
            assert f"0_first_cat_{category}" in features
            assert f"0_last_cat_{category}" in features
        for top in model.unicode_top_category_set:
            assert f"0_tcat_{top}" in features
            assert f"0_first_tcat_{top}" in features
            assert f"0_last_tcat_{top}" in features

    def test_explicit_overrides_used(self) -> None:
        model = _make_model()
        features = model.get_feature_list(letter_set=["z"], digit_set=[], punc_set=[], symbol_set=[])
        assert "0_char_z" in features
        assert "0_char_A" not in features

    def test_sum_features(self) -> None:
        summed = _make_model(calculate_sum=True).get_feature_list()
        assert "sum_char_other" in summed
        assert "sum_char_A" in summed
        assert "sum_digit_7" in summed
        assert "sum_punc_." in summed
        assert "sum_symbol_$" in summed
        plain = _make_model(calculate_sum=False).get_feature_list()
        assert "sum_char_other" not in plain
        assert not any(name.startswith("sum_") for name in plain)


TEXT = "Hi 7. é $"


def _make_doc() -> FakeDoc:
    return FakeDoc(
        [
            FakeToken("Hi", 0, pos="NOUN", tag="NN", dep="nsubj", is_title=True),
            FakeToken("7", 3, pos="AUX", tag="CD", dep="zzz-dep"),
            FakeToken(".", 4, pos="PUNCT", tag=".", dep="punct"),
            FakeToken("é", 6, pos="X", tag="ZZZ", dep="dep", is_lower=True),
            FakeToken("$", 8, pos="SYM", tag="$", dep="dep", is_lower=False),
        ]
    )


def _make_data_model(**kwargs) -> SpacyTokenSequenceClassifierModel:
    import string

    return SpacyTokenSequenceClassifierModel(
        letter_set=list(string.ascii_letters),
        digit_set=list(string.digits),
        punc_set=["."],
        symbol_set=["$"],
        match_tokens=["Hi"],
        string_checks=True,
        **kwargs,
    )


def _feature(model: SpacyTokenSequenceClassifierModel, data, row: int, name: str) -> int:
    return int(data[row, model._feature_index_map[name]])


class TestGetFeatureData:
    def test_tokens_offsets_shape_and_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import numpy

        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, tokens = model.get_feature_data(TEXT)
        assert tokens == [(0, 2), (3, 4), (4, 5), (6, 7), (8, 9)]
        assert data.shape == (5, len(model.feature_list))
        assert data.dtype == numpy.int8
        assert _feature(model, data, 0, "position") == 0
        assert _feature(model, data, 3, "position") == 3
        assert _feature(model, data, 0, "length") == 2
        assert _feature(model, data, 1, "length") == 1

    def test_mask_defaults_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert all(_feature(model, data, i, "mask") == 0 for i in range(5))
        data_none, _ = model.get_feature_data(TEXT, feature_mask=None)
        assert all(_feature(model, data_none, i, "mask") == 0 for i in range(5))

    def test_mask_takes_token_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT, feature_mask=[0, 0, 9, 5, 0, 8, 2, 0, 4])
        assert _feature(model, data, 0, "mask") == 0
        assert _feature(model, data, 1, "mask") == 5
        assert _feature(model, data, 2, "mask") == 0
        assert _feature(model, data, 3, "mask") == 2
        assert _feature(model, data, 4, "mask") == 4

    def test_pos_tag_dep_known_and_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert _feature(model, data, 0, "is_pos_NOUN") == 1
        assert _feature(model, data, 1, "is_pos_other") == 1  # AUX not in list
        assert _feature(model, data, 0, "is_tag_NN") == 1
        assert _feature(model, data, 3, "is_tag_other") == 1  # ZZZ not in list
        assert _feature(model, data, 0, "is_dep_nsubj") == 1
        assert _feature(model, data, 1, "is_dep_other") == 1  # zzz-dep not in list
        assert _feature(model, data, 2, "is_dep_punct") == 1

    def test_string_checks_and_match_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert _feature(model, data, 0, "0_is_title") == 1
        assert _feature(model, data, 1, "0_is_title") == 0
        assert _feature(model, data, 3, "0_is_lower") == 1
        assert _feature(model, data, 0, "0_token_Hi") == 1
        assert _feature(model, data, 1, "0_token_Hi") == 0

    def test_letter_digit_punc_symbol_other_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert _feature(model, data, 0, "0_char_H") == 1
        assert _feature(model, data, 0, "0_lchar_h") == 1
        assert _feature(model, data, 0, "0_first_char_H") == 1
        assert _feature(model, data, 0, "0_first_lchar_h") == 1
        assert _feature(model, data, 0, "0_char_i") == 1
        assert _feature(model, data, 0, "0_last_char_i") == 1
        assert _feature(model, data, 0, "0_last_lchar_i") == 1
        assert _feature(model, data, 1, "0_digit_7") == 1
        assert _feature(model, data, 1, "0_first_digit_7") == 1
        assert _feature(model, data, 1, "0_last_digit_7") == 1
        assert _feature(model, data, 2, "0_punc_.") == 1
        assert _feature(model, data, 2, "0_first_punc_.") == 1
        assert _feature(model, data, 2, "0_last_punc_.") == 1
        assert _feature(model, data, 4, "0_symbol_$") == 1
        assert _feature(model, data, 4, "0_first_symbol_$") == 1
        assert _feature(model, data, 4, "0_last_symbol_$") == 1
        assert _feature(model, data, 3, "0_char_other") == 1
        assert _feature(model, data, 3, "0_first_char_other") == 1
        assert _feature(model, data, 3, "0_last_char_other") == 1
        assert _feature(model, data, 0, "0_char_other") == 0

    def test_unicode_category_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert _feature(model, data, 0, "0_cat_Lu") == 1
        assert _feature(model, data, 0, "0_cat_Ll") == 1
        assert _feature(model, data, 0, "0_first_cat_Lu") == 1
        assert _feature(model, data, 0, "0_last_cat_Ll") == 1
        assert _feature(model, data, 0, "0_tcat_L") == 2
        assert _feature(model, data, 0, "0_first_tcat_L") == 1
        assert _feature(model, data, 0, "0_last_tcat_L") == 1
        assert _feature(model, data, 1, "0_cat_Nd") == 1
        assert _feature(model, data, 1, "0_tcat_N") == 1

    def test_unknown_characters_fall_back_to_c_and_cc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model()
        model.unicode_character_top_category_mapping = {}
        model.unicode_character_category_mapping = {}
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: FakeDoc([FakeToken("Hi", 0)]))
        data, tokens = model.get_feature_data("Hi")
        assert tokens == [(0, 2)]
        assert _feature(model, data, 0, "0_cat_Cc") == 2
        assert _feature(model, data, 0, "0_first_cat_Cc") == 1
        assert _feature(model, data, 0, "0_last_cat_Cc") == 1
        assert _feature(model, data, 0, "0_tcat_C") == 2
        assert _feature(model, data, 0, "0_first_tcat_C") == 1
        assert _feature(model, data, 0, "0_last_tcat_C") == 1

    def test_window_propagates_neighbor_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model(pre_window=1, post_window=1)
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        assert "-1_char_H" in model._feature_index_map
        assert "1_digit_7" in model._feature_index_map
        # Pre-window side: token 1 copies token 0's base features under "-1_".
        assert _feature(model, data, 1, "-1_char_H") == _feature(model, data, 0, "0_char_H") == 1
        # Post-window side near the tail: token 3 copies token 4's base features under "1_".
        assert _feature(model, data, 3, "1_symbol_$") == _feature(model, data, 4, "0_symbol_$") == 1
        # Tail token copies its predecessor under "-1_".
        assert _feature(model, data, 4, "-1_char_other") == _feature(model, data, 3, "0_char_other") == 1

    def test_window_range_excludes_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _make_data_model(pre_window=1, post_window=1)
        monkeypatch.setattr(spacy_model, "NLP_EN", lambda text: _make_doc())
        data, _ = model.get_feature_data(TEXT)
        # Token 0's range only covers itself, so no neighbor features are copied into it.
        assert _feature(model, data, 0, "1_digit_7") == 0
        assert _feature(model, data, 1, "0_digit_7") == 1
