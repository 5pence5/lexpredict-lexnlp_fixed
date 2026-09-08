"""Coverage tests for defensive/error branches in FactExtractor.parse_text."""

from __future__ import annotations

import copy

import pytest

from lexnlp.extract.common.annotation_type import AnnotationType
from lexnlp.extract.common.fact_extracting import ExtractorResultFormat, FactExtractor
from lexnlp.extract.common.tests.test_fact_extractor import make_geoconfig

EN_GEO_CONFIG = make_geoconfig()

MONEY_TEXT = """
Three people check into a hotel room.
The manager says the bill is $30, so each guest pays $10.
"""


def test_unknown_language_raises() -> None:
    with pytest.raises(Exception, match='Language "xx" was not found among'):
        FactExtractor.parse_text(MONEY_TEXT, "xx", ExtractorResultFormat.fmt_class)


def test_unknown_language_error_lists_known_langs() -> None:
    with pytest.raises(Exception) as exc_info:
        FactExtractor.parse_text(MONEY_TEXT, "xx", ExtractorResultFormat.fmt_class)
    message = str(exc_info.value)
    assert "en" in message
    assert "de" in message


def test_unsupported_format_raises() -> None:
    saved = FactExtractor.func_by_lang["en"]
    FactExtractor.func_by_lang["en"] = {k: v for k, v in saved.items() if k is not ExtractorResultFormat.fmt_object}
    try:
        with pytest.raises(Exception, match='Format "fmt_object" is not supported for en'):
            FactExtractor.parse_text(
                MONEY_TEXT,
                FactExtractor.LANGUAGE_EN,
                ExtractorResultFormat.fmt_object,
                extract_all=False,
                include_types={AnnotationType.money},
            )
    finally:
        FactExtractor.func_by_lang["en"] = saved


def test_extract_all_false_without_include_types_returns_empty() -> None:
    facts = FactExtractor.parse_text(
        MONEY_TEXT,
        FactExtractor.LANGUAGE_EN,
        ExtractorResultFormat.fmt_class,
        extract_all=False,
        include_types=None,
    )
    assert facts == {}


def test_exclude_types_drops_money_but_keeps_others() -> None:
    FactExtractor.ensure_parser_arguments_en(geo_config=EN_GEO_CONFIG)
    facts = FactExtractor.parse_text(
        MONEY_TEXT,
        FactExtractor.LANGUAGE_EN,
        ExtractorResultFormat.fmt_class,
        extract_all=True,
        exclude_types={AnnotationType.money},
    )
    assert AnnotationType.money not in facts
    assert len(facts) > 0


def test_exclude_all_types_returns_empty() -> None:
    facts = FactExtractor.parse_text(
        MONEY_TEXT,
        FactExtractor.LANGUAGE_EN,
        ExtractorResultFormat.fmt_class,
        extract_all=True,
        exclude_types=set(AnnotationType),
    )
    assert facts == {}


def test_ensure_parser_arguments_de_registers_geoentity_for_all_formats() -> None:
    saved = copy.deepcopy(FactExtractor.parser_extra_arguments)
    try:
        geo_config = ["Berlin", "Munich"]
        FactExtractor.ensure_parser_arguments_de(geo_config=geo_config)
        for fmt in ExtractorResultFormat:
            stored = FactExtractor.parser_extra_arguments[FactExtractor.LANGUAGE_DE][fmt][AnnotationType.geoentity]
            assert stored == (geo_config,)
    finally:
        FactExtractor.parser_extra_arguments.clear()
        FactExtractor.parser_extra_arguments.update(saved)
