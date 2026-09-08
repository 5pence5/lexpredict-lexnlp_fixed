"""Coverage tests for :mod:`lexnlp.nlp.en.segments.sections` missing lines."""

from __future__ import annotations

from lexnlp.nlp.en.segments.sections import (
    DocumentSection,
    find_section_titles,
    get_document_sections_with_titles,
)
from lexnlp.nlp.en.segments.sentences import get_sentence_span_list


class TestDocumentSectionDunder:
    def test_str_includes_title_and_span(self) -> None:
        section = DocumentSection(title="Hello", start=1, end=5)
        assert str(section) == "Hello [1: 5]"

    def test_eq_against_other_type_returns_not_implemented(self) -> None:
        section = DocumentSection(title="Hello", start=1, end=5)
        assert section.__eq__("Hello") is NotImplemented
        assert (section == "Hello") is False
        assert (section != "Hello") is True

    def test_eq_roundtrip(self) -> None:
        first = DocumentSection(title="A", start=0, end=5, level=1, abs_level=2, text="hi")
        second = DocumentSection(title="A", start=0, end=5, level=1, abs_level=2, text="hi")
        assert first == second


class TestGetDocumentSectionsWithTitles:
    def test_returns_sections_with_titles(self) -> None:
        text = "SECTION 1. Intro\nContent here line.\nSECTION 2. Next\nMore content here.\n"
        sentences = get_sentence_span_list(text)
        assert len(sentences) > 0
        sections = get_document_sections_with_titles(text, sentences, use_ml=False)
        assert len(sections) == 2
        for section in sections:
            assert isinstance(section, DocumentSection)
            assert section.title
            assert section.start < section.end
            assert text[section.title_start : section.title_end] == section.title
        assert sections[0].title == "SECTION 1."
        assert sections[1].title == "SECTION 2."
        assert sections[0].start == 0
        # return_text=False so section bodies are not stored
        assert sections[0].text == ""


class TestFindSectionTitles:
    def test_empty_sections_returns_none(self) -> None:
        sections: list[DocumentSection] = []
        result = find_section_titles(sections, [(0, 5)], "hello")
        assert result is None
        assert sections == []

    def test_section_before_sentence_breaks_inner_loop(self) -> None:
        full_text = "x" * 150
        first = DocumentSection(
            start=0,
            end=5,
            title="A",
            title_start=0,
            title_end=1,
            level=1,
            abs_level=1,
            text="xxxxx",
        )
        second = DocumentSection(
            start=50,
            end=55,
            title="B",
            title_start=50,
            title_end=51,
            level=1,
            abs_level=1,
            text="xxxxx",
        )
        sections = [first, second]
        sentences = [(0, 5), (100, 110)]
        find_section_titles(sections, sentences, full_text)
        # First section picks up the sentence title at (0, 5).
        assert first.title == "xxxxx"
        assert first.title_start == 0
        assert first.title_end == 5
        # Second section ends before sentence (100, 110) starts, so the
        # inner loop breaks without a title and the section is unchanged.
        assert second.title == "B"
        assert second.title_start == 50
        assert second.title_end == 51
