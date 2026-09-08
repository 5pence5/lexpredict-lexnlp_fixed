"""Coverage tests for lexnlp/utils/amount_delimiting.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lexnlp.utils.amount_delimiting import (
    DelimitedBlock,
    check_block_grouping,
    get_delimited_blocks,
    infer_delimiters,
)


class _FakeLocaleContextManager:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _mock_conventions(decimal: str, group: str, grouping: list[int]):
    conventions = {
        "decimal_point": decimal,
        "thousands_sep": group,
        "grouping": grouping,
    }
    return (
        patch(
            "lexnlp.utils.amount_delimiting.LocaleContextManager",
            _FakeLocaleContextManager,
        ),
        patch(
            "lexnlp.utils.amount_delimiting.locale.localeconv",
            return_value=conventions,
        ),
    )


def _infer(text: str, locale: str, decimal: str, group: str, grouping: list[int]):
    ctx, conv = _mock_conventions(decimal, group, grouping)
    with ctx, conv:
        return infer_delimiters(text, locale)


class TestGetDelimitedBlocks:
    def test_empty_string_returns_none(self):
        assert get_delimited_blocks("") is None

    def test_no_delimiters_single_block(self):
        delimiters, blocks = get_delimited_blocks("123")
        assert delimiters == set()
        assert blocks == [DelimitedBlock(length=2, delimiter=None)]

    def test_single_comma_block(self):
        delimiters, blocks = get_delimited_blocks("10,000")
        assert delimiters == {","}
        assert blocks == [DelimitedBlock(length=3, delimiter=",")]

    def test_two_comma_blocks(self):
        delimiters, blocks = get_delimited_blocks("10,000,000")
        assert delimiters == {","}
        assert blocks == [
            DelimitedBlock(length=3, delimiter=","),
            DelimitedBlock(length=3, delimiter=","),
        ]


class TestCheckBlockGrouping:
    def test_matching_grouping_passes(self):
        blocks = [
            DelimitedBlock(length=3, delimiter=","),
            DelimitedBlock(length=3, delimiter=","),
        ]
        assert check_block_grouping(blocks, ".", [3, 3, 0]) is True

    def test_mismatched_non_decimal_delimiter_fails(self):
        blocks = [DelimitedBlock(length=2, delimiter=",")]
        assert check_block_grouping(blocks, ".", [3, 3, 0]) is False

    def test_mismatched_decimal_delimiter_at_index_zero_then_repeat_fails(self):
        blocks = [
            DelimitedBlock(length=3, delimiter="."),
            DelimitedBlock(length=2, delimiter="."),
        ]
        assert check_block_grouping(blocks, ".", [3, 3, 0]) is False

    def test_mismatched_decimal_delimiter_beyond_index_zero_fails(self):
        blocks = [
            DelimitedBlock(length=2, delimiter="."),
            DelimitedBlock(length=3, delimiter=","),
        ]
        assert check_block_grouping(blocks, ".", [3, 3, 0]) is False

    def test_decimal_delimiter_at_index_zero_records_and_passes(self):
        blocks = [DelimitedBlock(length=2, delimiter=".")]
        assert check_block_grouping(blocks, ".", [3, 3, 0]) is True


class TestInferDelimitersInvalidText:
    def test_alpha_text_returns_none(self):
        assert _infer("abc", "en_US", ".", ",", [3, 3, 0]) is None

    def test_alphanumeric_text_returns_none(self):
        assert _infer("12a34", "en_US", ".", ",", [3, 3, 0]) is None


class TestInferDelimitersLocaleBranches:
    def test_bare_integer_returns_locale_defaults(self):
        result = _infer("12345", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_de_de_wrong_conventions_are_corrected(self):
        result = _infer("1.000,50", "de_DE", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ",", "group_delimiter": "."}

    def test_en_us_wrong_conventions_are_corrected(self):
        result = _infer("1,000.50", "en_US", ",", ".", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_en_us_wrong_grouping_is_corrected(self):
        result = _infer("1,000.50", "en_US", ".", ",", [3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_generic_empty_grouping_fallback(self):
        result = _infer("12345", "fr_FR", ".", ",", [])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_generic_empty_grouping_single_delimiter(self):
        result = _infer("1,000", "fr_FR", ".", ",", [])
        assert result is not None
        assert result["group_delimiter"] == ","


class TestInferDelimitersDelimiterCounts:
    def test_three_delimiters_returns_none(self):
        result = _infer("1,000.00 00", "en_US", ".", ",", [3, 3, 0])
        assert result is None

    def test_two_delimiters_float_greater_than_one(self):
        result = _infer("1,000.50", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_two_delimiters_reversed_roles(self):
        result = _infer("1.000,50", "de_DE", ",", ".", [3, 3, 0])
        assert result == {"decimal_delimiter": ",", "group_delimiter": "."}


class TestInferDelimitersSingleDelimiter:
    def test_group_of_three_not_decimal(self):
        result = _infer("1,000", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": None, "group_delimiter": ","}

    def test_group_of_three_matching_decimal_is_integer(self):
        result = _infer("1.000", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_short_block_is_fraction_below_one(self):
        result = _infer("1.00", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_fraction_below_one_group_collision_yields_none_group(self):
        # fr_FR avoids the en_US/de_DE canonical-override branches so the
        # real locale group (".") collides with the block delimiter.
        result = _infer(".50", "fr_FR", ".", ".", [3, 3, 0])
        assert result is not None
        assert result["decimal_delimiter"] == "."
        assert result["group_delimiter"] is None

    def test_multi_block_grouping_passes(self):
        result = _infer("10,000,000", "en_US", ".", ",", [3, 3, 0])
        assert result == {"decimal_delimiter": ".", "group_delimiter": ","}

    def test_multi_block_grouping_failure_returns_none(self):
        result = _infer("10,00,000", "en_US", ".", ",", [3, 3, 0])
        assert result is None

    def test_multi_block_decimal_in_delimiters_yields_none_decimal(self):
        # en_US decimal "." appears as the only delimiter with valid grouping,
        # so the decimal slot is reported as None and "." is the group.
        result = _infer("1.000.000", "en_US", ".", ",", [3, 3, 0])
        assert result is not None
        assert result["decimal_delimiter"] is None
        assert result["group_delimiter"] == "."


def test_infer_delimiters_indian_grouping_convention():
    result = _infer("10,00,000", "en_IN", ".", ",", [3, 2, 0])
    assert result is not None


class TestInferDelimitersTwoDelimiterFallback:
    """Lines 222-225: fallback when the two-delimiter scan names no decimal.

    With the real get_delimited_blocks the last block delimiter is always one
    of the two delimiters, so both slots are always filled and line 220 always
    returns. These tests force the defensive state (last-block delimiter
    outside the delimiter set) by stubbing get_delimited_blocks.
    """

    def _infer_with_blocks(self, text, monkeypatch, blocks_ret):
        ctx, conv = _mock_conventions(".", ",", [3, 3, 0])
        monkeypatch.setattr(
            "lexnlp.utils.amount_delimiting.get_delimited_blocks",
            lambda _text: blocks_ret,
        )
        with ctx, conv:
            return infer_delimiters(text, "en_US")

    def test_grouping_failure_returns_none(self, monkeypatch):
        blocks = [DelimitedBlock(length=2, delimiter=";")]
        assert self._infer_with_blocks("1,000.50", monkeypatch, ({",", "."}, blocks)) is None

    def test_grouping_pass_returns_empty_decimal(self, monkeypatch):
        blocks = [DelimitedBlock(length=3, delimiter=";")]
        result = self._infer_with_blocks("1,000.50", monkeypatch, ({",", "."}, blocks))
        assert result is not None
        assert result["decimal_delimiter"] == ""
        assert result["group_delimiter"] in {",", "."}


@pytest.mark.parametrize("bad", ["", "1,2", "x"])
def test_get_delimited_blocks_parametrized_smoke(bad: str):
    if bad == "":
        assert get_delimited_blocks(bad) is None
    else:
        assert get_delimited_blocks(bad) is not None
