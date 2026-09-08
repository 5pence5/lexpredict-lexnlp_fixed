"""Coverage tests for lexnlp.extract.en.citations wrappers and KeyError path."""

from lexnlp.extract.common.annotations.citation_annotation import CitationAnnotation
from lexnlp.extract.en.citations import (
    get_citation_annotation_list,
    get_citation_annotations,
    get_citation_list,
    get_citations,
)


def test_get_citation_list_returns_expected_tuple() -> None:
    text = "see 410 U.S. 113 (1973) for details"
    expected = [(410, "U.S.", "United States Supreme Court Reports", 113, None, None, 1973)]
    assert get_citation_list(text) == expected
    assert get_citation_list(text) == list(get_citations(text))


def test_get_citation_list_return_source_and_dict() -> None:
    text = "see 410 U.S. 113 (1973) for details"
    with_source = get_citation_list(text, return_source=True)
    assert with_source == [
        (410, "U.S.", "United States Supreme Court Reports", 113, None, None, 1973, "410 U.S. 113 (1973)")
    ]
    assert get_citation_list(text, as_dict=True) == list(get_citations(text, as_dict=True))
    as_dict = get_citation_list(text, as_dict=True)
    assert as_dict[0]["volume"] == 410
    assert as_dict[0]["reporter"] == "U.S."
    assert as_dict[0]["year"] == 1973
    assert get_citation_list("no citations here") == []


def test_lowercase_reporter_silently_skipped() -> None:
    # "u.s." matches the case-insensitive pattern but EDITIONS lookup is
    # case-sensitive, so it raises KeyError inside get_citation_annotations
    # and is swallowed (lines 140-141).
    assert get_citation_list("see 410 u.s. 113 (1973)") == []
    assert get_citation_annotation_list("see 410 u.s. 113 (1973)") == []
    assert list(get_citation_annotations("see 410 u.s. 113 (1973)")) == []


def test_mixed_case_text_yields_only_valid_citation() -> None:
    text = "X 410 u.s. 113. Y 410 U.S. 113 (1973) Z"
    result = get_citation_list(text)
    assert len(result) == 1
    assert result[0][0] == 410
    assert result[0][1] == "U.S."
    assert result[0][3] == 113
    assert result[0][6] == 1973


def test_get_citation_annotation_list_fields() -> None:
    text = "see 410 U.S. 113 (1973) for details"
    result = get_citation_annotation_list(text)
    assert isinstance(result, list)
    assert len(result) == 1
    ant = result[0]
    assert isinstance(ant, CitationAnnotation)
    assert ant.volume == 410
    assert ant.reporter == "U.S."
    assert ant.reporter_full_name == "United States Supreme Court Reports"
    assert ant.page == 113
    assert ant.year == 1973
    assert ant.locale == "en"
    gen = list(get_citation_annotations(text))
    assert len(gen) == 1
    assert (gen[0].volume, gen[0].reporter, gen[0].page, gen[0].year) == (410, "U.S.", 113, 1973)
    assert get_citation_annotation_list("no citations here") == []
