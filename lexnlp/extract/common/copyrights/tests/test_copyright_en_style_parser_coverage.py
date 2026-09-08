"""Coverage tests for copyright_en_style_parser missing lines."""

import pytest

from lexnlp.extract.common.annotations.copyright_annotation import CopyrightAnnotation
from lexnlp.extract.common.copyrights.copyright_en_style_parser import CopyrightEnStyleParser
from lexnlp.extract.de.copyrights import CopyrightDeParser


@pytest.fixture
def real_phrases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the base parser's abstract phrase splitter to the real DE implementation."""

    def _split(cls, sentence: str) -> list[tuple[str, int, int]]:
        return CopyrightDeParser.extract_phrases_with_coords(sentence)

    monkeypatch.setattr(CopyrightEnStyleParser, "extract_phrases_with_coords", classmethod(_split))


class TestExtractPhrases:
    def test_base_implementation_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            CopyrightEnStyleParser.extract_phrases_with_coords("Copyright 2019 Siemens AG")


class TestGetCopyrights:
    def test_without_sources(self, real_phrases: None) -> None:
        assert list(CopyrightEnStyleParser.get_copyrights("Copyright 2019 Siemens AG")) == [
            ("Copyright", "2019", "Siemens AG")
        ]

    def test_with_sources(self, real_phrases: None) -> None:
        assert list(CopyrightEnStyleParser.get_copyrights("Copyright 2019 Siemens AG", return_sources=True)) == [
            ("Copyright", "2019", "Siemens AG", "Copyright 2019 Siemens AG")
        ]


class TestDeriveCompanyName:
    def test_valid_preset_company_kept(self) -> None:
        ant = CopyrightAnnotation(coords=(0, 5), name="Siemens AG 2019", company="Acme Corp")
        CopyrightEnStyleParser.derive_company_name(ant, "Copyright 2019 Siemens AG")
        assert ant.company == "Acme Corp"

    def test_invalid_preset_company_rederived(self) -> None:
        ant = CopyrightAnnotation(coords=(0, 5), name="Siemens AG", company="123")
        CopyrightEnStyleParser.derive_company_name(ant, "Copyright 2019 Siemens AG")
        assert ant.company == "Siemens AG"


class TestTakeBestCompanyName:
    def test_first_matching_name_returned(self) -> None:
        assert CopyrightEnStyleParser.take_best_company_name(["Siemens AG"]) == "Siemens AG"

    def test_fallback_returns_first_name(self) -> None:
        assert CopyrightEnStyleParser.take_best_company_name(["abc", "def"]) == "abc"
