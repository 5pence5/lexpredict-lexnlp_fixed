"""Coverage tests for lexnlp.extract.common.definitions.common_definition_patterns."""

import regex as re

from lexnlp.extract.common.definitions.common_definition_patterns import CommonDefinitionPatterns


def test_match_acronyms_no_preceding_words_returns_empty():
    # "(AB)" has no words before it, so get_acronym_words_start hits
    # the `len(words) < 2` early return and the match is skipped.
    assert CommonDefinitionPatterns.match_acronyms("(AB)") == []


def test_get_acronym_words_start_short_phrase_returns_minus_one():
    match = CommonDefinitionPatterns.reg_acronyms.search("(AB)")
    assert match is not None
    assert CommonDefinitionPatterns.get_acronym_words_start("(AB)", match) == -1


def test_match_acronyms_finds_preceded_acronym():
    defs = CommonDefinitionPatterns.match_acronyms("Canal del Futbol (CDF)")
    assert [(d.name, d.start, d.end, d.probability) for d in defs] == [("CDF", 0, 16, 100)]


def test_match_acronyms_mismatched_initials_returns_empty():
    assert CommonDefinitionPatterns.match_acronyms("hello world (ZZ)") == []


def test_collect_regex_matches_returns_positions_and_probability():
    reg = re.compile(r"\d+")
    defs = CommonDefinitionPatterns.collect_regex_matches(
        "abc 123 def 456", reg, 100, lambda _phrase, m: m.start(), lambda _phrase, m: m.end()
    )
    assert [(d.name, d.start, d.end, d.probability) for d in defs] == [
        ("123", 4, 7, 100),
        ("456", 12, 15, 100),
    ]


def test_collect_regex_matches_no_match_returns_empty():
    reg = re.compile(r"\d+")
    assert (
        CommonDefinitionPatterns.collect_regex_matches(
            "abc", reg, 100, lambda _phrase, m: m.start(), lambda _phrase, m: m.end()
        )
        == []
    )
