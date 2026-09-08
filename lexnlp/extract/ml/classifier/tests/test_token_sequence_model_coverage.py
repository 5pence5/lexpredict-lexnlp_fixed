"""Coverage tests for TokenSequenceClassifierModel.

Targets the character-tokenizer feature list / feature data paths in
``lexnlp/extract/ml/classifier/token_sequence_model.py``.
"""

from __future__ import annotations

import numpy

from lexnlp.extract.ml.classifier.token_sequence_model import TokenSequenceClassifierModel


def _make_model(**kwargs) -> TokenSequenceClassifierModel:
    return TokenSequenceClassifierModel(
        letter_set=["a", "b", "c", "d", "e", "f"],
        digit_set=["1", "7"],
        punc_set=["."],
        symbol_set=["$"],
        match_tokens=["Hi"],
        **kwargs,
    )


class TestInit:
    def test_defaults_stored(self) -> None:
        model = TokenSequenceClassifierModel()
        assert model.letter_set == []
        assert model.digit_set == []
        assert model.punc_set == []
        assert model.symbol_set == []
        assert model.match_tokens == []
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
    def test_defaults_come_from_instance(self) -> None:
        model = _make_model(pre_window=1, post_window=1)
        assert model.get_feature_list() == model.get_feature_list(
            model.letter_set,
            model.digit_set,
            model.punc_set,
            model.symbol_set,
            model.pre_window,
            model.post_window,
        )

    def test_explicit_overrides_replace_defaults(self) -> None:
        model = _make_model()
        features = model.get_feature_list(
            letter_set=["z"],
            digit_set=[],
            punc_set=[],
            symbol_set=[],
            pre_window=0,
            post_window=0,
        )
        assert "0_char_z" in features
        assert "0_lchar_z" in features
        assert "0_char_a" not in features
        assert "0_digit_1" not in features
        assert "-1_is_start" not in features
        assert "1_is_start" not in features

    def test_core_features_without_string_checks(self) -> None:
        features = _make_model(string_checks=False).get_feature_list()
        assert features[:3] == ["position", "length", "mask"]
        assert "0_is_start" in features
        assert "0_is_end" in features
        assert "0_is_title" not in features
        assert "0_is_lower" not in features
        assert "0_is_upper" not in features

    def test_string_checks_add_case_features(self) -> None:
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

    def test_character_digit_punc_symbol_features(self) -> None:
        features = _make_model().get_feature_list()
        for name in (
            "0_char_a",
            "0_lchar_a",
            "0_first_char_a",
            "0_first_lchar_a",
            "0_last_char_a",
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
            "0_token_Hi",
        ):
            assert name in features

    def test_unicode_category_features_present(self) -> None:
        model = _make_model()
        features = model.get_feature_list()
        assert "0_cat_Ll" in features
        assert "0_first_cat_Ll" in features
        assert "0_last_cat_Ll" in features
        assert "0_tcat_L" in features
        assert "0_first_tcat_L" in features
        assert "0_last_tcat_L" in features

    def test_calculate_sum_appends_sum_features(self) -> None:
        features = _make_model(calculate_sum=True).get_feature_list()
        assert "sum_char_other" in features
        assert "sum_char_a" in features
        assert "sum_digit_7" in features
        assert "sum_punc_." in features
        assert "sum_symbol_$" in features

    def test_no_sum_features_by_default(self) -> None:
        features = _make_model().get_feature_list()
        assert not any(name.startswith("sum_") for name in features)


class TestGetFeatureData:
    def test_token_offsets_and_position_length(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("ab cd")
        assert tokens == [(0, 2), (3, 5)]
        assert feature_data.shape == (2, len(model.feature_list))
        assert feature_data.dtype == numpy.int8
        pos = model._feature_index_map["position"]
        length = model._feature_index_map["length"]
        assert feature_data[0, pos] == 0
        assert feature_data[1, pos] == 1
        assert feature_data[0, length] == 2
        assert feature_data[1, length] == 2

    def test_start_end_flags(self) -> None:
        model = _make_model()
        feature_data, _ = model.get_feature_data("ab cd")
        idx = model._feature_index_map
        assert feature_data[0, idx["0_is_start"]] == 1
        assert feature_data[0, idx["0_is_end"]] == 0
        assert feature_data[1, idx["0_is_start"]] == 0
        assert feature_data[1, idx["0_is_end"]] == 1

    def test_letter_first_last_counts(self) -> None:
        model = _make_model()
        feature_data, _ = model.get_feature_data("ab")
        idx = model._feature_index_map
        assert feature_data[0, idx["0_char_a"]] == 1
        assert feature_data[0, idx["0_lchar_a"]] == 1
        assert feature_data[0, idx["0_first_char_a"]] == 1
        assert feature_data[0, idx["0_first_lchar_a"]] == 1
        assert feature_data[0, idx["0_last_char_b"]] == 1
        assert feature_data[0, idx["0_last_lchar_b"]] == 1
        assert feature_data[0, idx["0_first_char_b"]] == 0
        assert feature_data[0, idx["0_last_char_a"]] == 0

    def test_digit_first_last_counts(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("a17")
        assert tokens == [(0, 3)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_digit_1"]] == 1
        assert feature_data[0, idx["0_digit_7"]] == 1
        assert feature_data[0, idx["0_first_digit_1"]] == 0
        assert feature_data[0, idx["0_last_digit_1"]] == 0
        assert feature_data[0, idx["0_first_digit_7"]] == 0
        assert feature_data[0, idx["0_last_digit_7"]] == 1

    def test_punc_first_last_counts(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("a.c")
        assert tokens == [(0, 3)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_punc_."]] == 1
        assert feature_data[0, idx["0_first_punc_."]] == 0
        assert feature_data[0, idx["0_last_punc_."]] == 0

    def test_symbol_first_last_counts(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("$ab")
        assert tokens == [(0, 3)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_symbol_$"]] == 1
        assert feature_data[0, idx["0_first_symbol_$"]] == 1
        assert feature_data[0, idx["0_last_symbol_$"]] == 0

    def test_other_char_counts(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("a~")
        assert tokens == [(0, 2)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_char_other"]] == 1
        assert feature_data[0, idx["0_first_char_other"]] == 0
        assert feature_data[0, idx["0_last_char_other"]] == 1

    def test_match_token_flag(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("Hi there")
        assert tokens[0] == (0, 2)
        idx = model._feature_index_map
        assert feature_data[0, idx["0_token_Hi"]] == 1
        assert feature_data[1, idx["0_token_Hi"]] == 0

    def test_string_checks_flags(self) -> None:
        model = _make_model(string_checks=True)
        feature_data, _ = model.get_feature_data("Hello hello HELLO")
        idx = model._feature_index_map
        assert feature_data[0, idx["0_is_title"]] == 1
        assert feature_data[0, idx["0_is_lower"]] == 0
        assert feature_data[0, idx["0_is_upper"]] == 0
        assert feature_data[1, idx["0_is_title"]] == 0
        assert feature_data[1, idx["0_is_lower"]] == 1
        assert feature_data[1, idx["0_is_upper"]] == 0
        assert feature_data[2, idx["0_is_title"]] == 0
        assert feature_data[2, idx["0_is_lower"]] == 0
        assert feature_data[2, idx["0_is_upper"]] == 1

    def test_feature_mask_takes_token_max(self) -> None:
        model = _make_model()
        text = "ab cd"
        feature_data, _ = model.get_feature_data(text, feature_mask=[0, 3, 0, 1, 0])
        idx = model._feature_index_map
        assert feature_data[0, idx["mask"]] == 3
        assert feature_data[1, idx["mask"]] == 1

    def test_mask_defaults_to_zero(self) -> None:
        model = _make_model()
        feature_data, _ = model.get_feature_data("ab cd")
        idx = model._feature_index_map
        assert feature_data[0, idx["mask"]] == 0
        assert feature_data[1, idx["mask"]] == 0

    def test_single_char_digit_token_is_first_and_last(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("7")
        assert tokens == [(0, 1)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_digit_7"]] == 1
        assert feature_data[0, idx["0_first_digit_7"]] == 1
        assert feature_data[0, idx["0_last_digit_7"]] == 1

    def test_single_char_punc_token_is_first_and_last(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data(".")
        assert tokens == [(0, 1)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_punc_."]] == 1
        assert feature_data[0, idx["0_first_punc_."]] == 1
        assert feature_data[0, idx["0_last_punc_."]] == 1

    def test_single_char_symbol_token_is_first_and_last(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("$")
        assert tokens == [(0, 1)]
        idx = model._feature_index_map
        assert feature_data[0, idx["0_symbol_$"]] == 1
        assert feature_data[0, idx["0_first_symbol_$"]] == 1
        assert feature_data[0, idx["0_last_symbol_$"]] == 1

    def test_consecutive_separators_advance_token_start(self) -> None:
        model = _make_model()
        feature_data, tokens = model.get_feature_data("ab  cd")
        assert tokens == [(0, 2), (4, 6)]
        assert feature_data.shape[0] == 2

    def test_unicode_category_counts(self) -> None:
        model = _make_model()
        feature_data, _ = model.get_feature_data("ab")
        idx = model._feature_index_map
        assert feature_data[0, idx["0_cat_Ll"]] == 2
        assert feature_data[0, idx["0_tcat_L"]] == 2
        assert feature_data[0, idx["0_first_cat_Ll"]] == 1
        assert feature_data[0, idx["0_first_tcat_L"]] == 1
        assert feature_data[0, idx["0_last_cat_Ll"]] == 1
        assert feature_data[0, idx["0_last_tcat_L"]] == 1

    def test_window_copies_neighbor_base_features(self) -> None:
        model = _make_model(pre_window=1, post_window=1)
        feature_data, tokens = model.get_feature_data("ab cd ef")
        assert len(tokens) == 3
        idx = model._feature_index_map
        base = list(model._base_feature_list)
        assert base, "expected window-0 base features"
        for name in base:
            assert feature_data[1, idx["-1_" + name]] == feature_data[0, idx["0_" + name]]
        assert feature_data[1, idx["-1_char_a"]] == 1
        assert feature_data[1, idx["-1_char_c"]] == 0

    def test_first_token_has_no_window_copies(self) -> None:
        model = _make_model(pre_window=1, post_window=1)
        feature_data, tokens = model.get_feature_data("ab cd ef gh ij")
        assert len(tokens) == 5
        idx = model._feature_index_map
        for name in model._base_feature_list:
            assert feature_data[0, idx["-1_" + name]] == 0
            assert feature_data[0, idx["1_" + name]] == 0
