"""Definition extraction for English.

This module implements basic definition extraction functionality in English."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"

# pylint: disable=broad-except,bare-except

from typing import List, Optional, Set, Tuple

import unidecode
import regex as re

from lexnlp.extract.common.text_beautifier import TextBeautifier
from lexnlp.extract.en.definition_patterns import (
    DEFAULT_PATTERN_REGISTRY,
    DefinitionPatternRegistry,
)
from lexnlp.extract.en.definition_term_utils import (
    PUNCTUATION_STRIP_STR,
    does_term_are_service_words,
    get_quotes_count_in_string,
    regex_matches_to_word_coords,
    split_definitions_inside_term,
    trim_defined_term,
)
from lexnlp.extract.en.en_language_tokens import EnLanguageTokens
from lexnlp.extract.en.introductory_words_detector import IntroductoryWordsDetector
from lexnlp.extract.en.preprocessing.span_tokenizer import SpanTokenizer
from lexnlp.utils.lines_processing.line_processor import LineProcessor


class DefinitionCaught:
    """Stores a definition's name, text and coordinates within the source."""

    __slots__ = ["name", "text", "coords"]

    def __init__(self, name: str, text: str, coords: Tuple[int, int]):
        self.name = name
        self.text = text
        self.coords = coords

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "%s [%d, %d]" % (self.name, self.coords[0], self.coords[1])

    def does_consume_target(self, target) -> int:
        """Determine whether this definition consumes another target definition."""

        coords_spans = (target.coords[0] >= self.coords[0] and target.coords[0] <= self.coords[1]) or (
            self.coords[0] >= target.coords[0] and self.coords[0] <= target.coords[1]
        )
        if not coords_spans:
            return 0
        if (target.name or "") in (self.name or ""):
            return 1
        if (self.name or "") in (target.name or ""):
            return -1
        return 0


class EnglishDefinitionExtractor:
    """Extracts definition terms from sentences using configurable patterns."""

    PICK_DEFINITION_FROM_QUOTES = True

    def __init__(
        self,
        pattern_registry: Optional[DefinitionPatternRegistry] = None,
        word_processor: Optional[LineProcessor] = None,
    ) -> None:
        self.pattern_registry = pattern_registry or DEFAULT_PATTERN_REGISTRY
        self.word_processor = word_processor or LineProcessor()
        self._strong_trigger_patterns = [
            re.compile(r"^\s*(?:%s)" % trigger.replace(" ", r"\s+"), re.IGNORECASE | re.UNICODE)
            for trigger in self.pattern_registry.strong_triggers
        ]

    def get_definition_list_in_sentence(
        self, sentence_coords: Tuple[int, int, str], decode_unicode: bool = True
    ) -> List[DefinitionCaught]:
        """Find possible definitions within a single sentence."""

        definitions: List[DefinitionCaught] = []
        sentence = TextBeautifier.unify_quotes_braces(sentence_coords[2], empty_replacement=" ")
        sent_start = sentence_coords[0]

        if decode_unicode:
            sentence = unidecode.unidecode(sentence)
            sentence_coords = (sentence_coords[0], sentence_coords[1], sentence)

        candidates = self._collect_candidate_spans(sentence, sent_start)

        for term, start, end in candidates:
            stripped_term, stripped_start, stripped_end = TextBeautifier.strip_pair_symbols((term, start, end))
            stripped_term, stripped_start, stripped_end, was_quoted = trim_defined_term(
                stripped_term, stripped_start, stripped_end, self.pattern_registry
            )

            if self.PICK_DEFINITION_FROM_QUOTES:
                term, start, end = stripped_term, stripped_start, stripped_end
            else:
                term, start, end = stripped_term, stripped_start, stripped_end

            if not stripped_term:
                continue

            term, start, end = TextBeautifier.unify_quotes_braces_coords(term, start, end)

            if len(term.strip(PUNCTUATION_STRIP_STR)) == 0:
                continue

            term_pos = list(SpanTokenizer.get_token_spans(term))
            if does_term_are_service_words(term_pos):
                continue

            term_wo_intro = IntroductoryWordsDetector.remove_term_introduction(term, term_pos)
            if term_wo_intro != term:
                term = TextBeautifier.strip_pair_symbols(term_wo_intro)
            if not term:
                continue

            max_words_per_definition = self.pattern_registry.max_term_tokens
            if was_quoted:
                max_words_per_definition = self.pattern_registry.max_quoted_term_tokens

            words_in_term = sum(
                1 for word in self.word_processor.split_text_on_words(stripped_term) if not word.is_separator
            )
            quotes_in_text = get_quotes_count_in_string(stripped_term)
            possible_definitions = quotes_in_text // 2 if quotes_in_text > 1 else 1
            possible_tokens_count = max_words_per_definition * possible_definitions
            if words_in_term > possible_tokens_count:
                continue

            split_definitions = split_definitions_inside_term(term, sentence_coords, start, end, self.pattern_registry)

            for definition, def_start, def_end in split_definitions:
                definition, def_start, def_end = TextBeautifier.strip_pair_symbols((definition, def_start, def_end))
                definitions.append(DefinitionCaught(definition, sentence, (def_start, def_end)))

        return definitions

    def _collect_candidate_spans(self, sentence: str, sent_start: int) -> Set[Tuple[str, int, int]]:
        """Collect candidate term spans that may represent definitions."""

        result: Set[Tuple[str, int, int]] = set()

        trigger_words_re = self.pattern_registry.get("trigger_words").regex
        extractor_re = self.pattern_registry.get("quoted_term_extractor").regex
        for match in trigger_words_re.finditer(sentence):
            result.update(
                regex_matches_to_word_coords(extractor_re, match.group(), match.start() + sent_start)
            )

        simple_trigger_re = self.pattern_registry.get("simple_trigger").regex
        for match in simple_trigger_re.finditer(sentence):
            term_text = match.group(1)
            result.add((term_text, match.start(1) + sent_start, match.end(1) + sent_start))

        for match in extractor_re.finditer(sentence):
            following_text = sentence[match.end():]
            # Skip up to two trailing word characters left outside the match (e.g., superscripts)
            offset = 0
            while offset < len(following_text) and offset < 2 and following_text[offset].isalnum():
                offset += 1
            following_text = following_text[offset:].lstrip()
            if any(pattern.match(following_text) for pattern in self._strong_trigger_patterns):
                result.add((match.group(), match.start() + sent_start, match.end() + sent_start))

        noun_matches = regex_matches_to_word_coords(
            self.pattern_registry.get("noun").regex, sentence, sent_start
        )
        noun_matches = [
            match for match in noun_matches if not self.pattern_registry.get("noun_anti").regex.fullmatch(match[0])
        ]
        noun_matches = [
            match for match in noun_matches if match[0].lower().strip(" ,;.") not in EnLanguageTokens.pronouns
        ]
        if noun_matches:
            result.update(noun_matches)

        trigger_quoted_re = self.pattern_registry.get("trigger_quoted_definition").regex
        quoted_definition_patterns = [pattern.regex for pattern in self.pattern_registry.group("quoted_definition")]
        for _ in trigger_quoted_re.finditer(sentence):
            for pattern in quoted_definition_patterns:
                result.update(regex_matches_to_word_coords(pattern, sentence, sent_start))
            break

        return result


def filter_definitions_for_self_repeating(definitions: List[DefinitionCaught]) -> List[DefinitionCaught]:
    """Exclude overlapping definitions, leaving unique ones only."""

    for i, definition in enumerate(definitions):
        if not definition.name:
            continue
        for j in range(i + 1, len(definitions)):
            other = definitions[j]
            consumes = definition.does_consume_target(other)
            if consumes == 1:
                other.name = None
            elif consumes == -1:
                definition.name = None

    return [definition for definition in definitions if definition.name is not None]


DEFAULT_EXTRACTOR = EnglishDefinitionExtractor()


def get_definition_list_in_sentence(
    sentence_coords: Tuple[int, int, str], decode_unicode: bool = True
) -> List[DefinitionCaught]:
    """Backward compatible helper delegating to :class:`EnglishDefinitionExtractor`."""

    return DEFAULT_EXTRACTOR.get_definition_list_in_sentence(sentence_coords, decode_unicode)


__all__ = [
    "DefinitionCaught",
    "EnglishDefinitionExtractor",
    "filter_definitions_for_self_repeating",
    "get_definition_list_in_sentence",
]
