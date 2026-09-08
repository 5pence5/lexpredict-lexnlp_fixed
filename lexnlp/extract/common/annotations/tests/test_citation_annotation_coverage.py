"""Coverage tests for the page-range tag in CitationAnnotation."""

from __future__ import annotations

from lexnlp.extract.common.annotations.citation_annotation import CitationAnnotation


class TestCitationPageRangeTag:
    def test_page_range_tag_present_when_set(self) -> None:
        ann = CitationAnnotation(coords=(0, 10), page=113, page_range="113-115")
        tags = ann.get_dictionary_values().tags
        assert tags["Extracted Entity Page Range"] == "113-115"
        assert tags["Extracted Entity Page"] == "113"

    def test_page_range_tag_absent_when_unset(self) -> None:
        ann = CitationAnnotation(coords=(0, 10), page=113, page_range=None)
        tags = ann.get_dictionary_values().tags
        assert "Extracted Entity Page Range" not in tags
        assert tags["Extracted Entity Page"] == "113"
