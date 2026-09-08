"""Coverage tests for the module-level helpers in ``lexnlp.extract.de.laws``.

Lines 112 (``yield from parser.parse``), 121 (``yield
annotation.to_dictionary()``) and 125 (``return list(get_laws(...))``)
only execute when the module-level ``parser`` is set, which the existing
suite never does. These tests install the real CSV-backed ``LawsParser``
and assert on actual annotation/dict values.
"""

from __future__ import annotations

import types
from unittest.mock import patch

from lexnlp.extract.common.annotations.law_annotation import LawAnnotation
from lexnlp.extract.de import laws
from lexnlp.extract.de.tests.test_laws import setup_parser

TEXT = "Dies ist durch das AAÜG geschehen."


def _patched_parser():
    return patch.object(laws, "parser", setup_parser())


class TestGetLawAnnotations:
    def test_yields_real_annotation(self) -> None:
        with _patched_parser():
            result = laws.get_law_annotations(TEXT)
            assert isinstance(result, types.GeneratorType)
            ants = list(result)
        assert len(ants) == 1
        ant = ants[0]
        assert isinstance(ant, LawAnnotation)
        assert ant.name == "AAÜG"
        assert ant.text == "AAÜG"
        assert ant.coords == (18, 24)
        assert ant.locale == "de"

    def test_language_passthrough(self) -> None:
        with _patched_parser():
            ants = list(laws.get_law_annotations(TEXT, "x"))
        assert len(ants) == 1
        assert ants[0].locale == "x"

    def test_empty_text_yields_nothing(self) -> None:
        with _patched_parser():
            assert list(laws.get_law_annotations("")) == []

    def test_annotation_list_helper(self) -> None:
        with _patched_parser():
            ants = laws.get_law_annotation_list(TEXT)
        assert isinstance(ants, list)
        assert len(ants) == 1
        assert ants[0].name == "AAÜG"
        assert ants[0].coords == (18, 24)


class TestGetLawsHelpers:
    def test_get_laws_yields_dictionaries(self) -> None:
        with _patched_parser():
            result = laws.get_laws(TEXT)
            assert isinstance(result, types.GeneratorType)
            dicts = list(result)
        assert len(dicts) == 1
        dic = dicts[0]
        assert dic["attrs"] == {"start": 18, "end": 24}
        assert dic["tags"]["Extracted Entity Name"] == "AAÜG"
        assert dic["tags"]["Extracted Entity Text"] == "AAÜG"

    def test_get_law_list(self) -> None:
        with _patched_parser():
            result = laws.get_law_list(TEXT)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["attrs"] == {"start": 18, "end": 24}
        with _patched_parser():
            assert result == laws.get_law_list(TEXT)
            assert laws.get_law_list("") == []
