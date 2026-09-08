"""Coverage tests for :mod:`lexnlp.extract.en.definition_parsing_methods`."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.en.definition_parsing_methods import (
    DefinitionCaught,
    get_definition_list_in_sentence,
    trim_defined_term,
)
from lexnlp.extract.en.introductory_words_detector import IntroductoryWordsDetector
from lexnlp.extract.en.preprocessing.span_tokenizer import SpanTokenizer


def test_repr_shows_name_and_coords() -> None:
    definition = DefinitionCaught("Acme", "some text", (3, 9))
    assert repr(definition) == "Acme [3, 9]"


def test_does_consume_target_unrelated_overlap_returns_zero() -> None:
    left = DefinitionCaught("Alpha", "sentence", (0, 10))
    right = DefinitionCaught("Beta", "sentence", (5, 15))
    assert left.does_consume_target(right) == 0
    assert right.does_consume_target(left) == 0


def test_does_consume_target_containment() -> None:
    outer = DefinitionCaught("Obligations", "sentence", (0, 10))
    inner = DefinitionCaught("Obligation", "sentence", (2, 8))
    assert outer.does_consume_target(inner) == 1
    assert inner.does_consume_target(outer) == -1


def test_does_consume_target_disjoint_returns_zero() -> None:
    left = DefinitionCaught("Alpha", "sentence", (0, 2))
    right = DefinitionCaught("Beta", "sentence", (5, 8))
    assert left.does_consume_target(right) == 0


def test_ellipsis_term_is_dropped() -> None:
    sentence = 'The word "..." includes all things.'
    assert get_definition_list_in_sentence((0, len(sentence), sentence)) == []
    term, _, _, _ = trim_defined_term("...", 10, 13)
    assert term == ""


def test_punctuation_only_term_is_dropped() -> None:
    sentence = 'The word ";" includes all things.'
    assert get_definition_list_in_sentence((0, len(sentence), sentence)) == []
    # ";" survives quote/dot trimming but is pure punctuation, so it is
    # rejected by the punctuation-strip guard.
    term, _, _, _ = trim_defined_term(";", 10, 11)
    assert term == ";"
    from lexnlp.extract.en.definition_parsing_methods import PUNCTUATION_STRIP_STR

    assert len(term.strip(PUNCTUATION_STRIP_STR)) == 0


def test_introductory_only_term_is_dropped() -> None:
    sentence = '"called" means something.'
    assert get_definition_list_in_sentence((0, len(sentence), sentence)) == []
    # "called" is tagged VBN and matches INTRO_VERBS, so the whole term is
    # consumed as an introduction.
    term_pos = list(SpanTokenizer.get_token_spans("called"))
    assert term_pos[0][1] == "VBN"
    assert IntroductoryWordsDetector.remove_term_introduction("called", term_pos) == ""
