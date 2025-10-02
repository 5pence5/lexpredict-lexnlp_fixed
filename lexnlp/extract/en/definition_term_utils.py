"""Helper utilities for cleaning and normalising definition terms."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence, Tuple

from lexnlp.extract.common.annotations.phrase_position_finder import PhrasePositionFinder
from lexnlp.extract.common.special_characters import SpecialCharacters
from lexnlp.extract.common.text_beautifier import TextBeautifier
from lexnlp.extract.en.definition_patterns import DEFAULT_PATTERN_REGISTRY, DefinitionPatternRegistry
from lexnlp.utils.iterating_helpers import count_sequence_matches


# non significant parts of speech
# if defined term consists only of NON_SIG_POSes this is not a definition
NON_SIG_POS: Sequence[str] = (
    'CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP$', 'RB', 'RBR',
    'RBS', 'RP', 'TO', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', ':', '-', ';',
    ')', '(', ']', '{', '}', '[', '*', '/', '\\', '"', "'", '!', '?', '%', '$',
    '^', '&', '@'
)

PUNCTUATION_POS = set({'``', '\t'}).union(SpecialCharacters.punctuation)
PUNCTUATION_STRIP_STR = ''.join(PUNCTUATION_POS)


def get_quotes_count_in_string(text: str) -> int:
    """Calculate the number of quotes present in *text*."""
    counter = Counter(text)
    return sum(filter(None, [counter['"'], counter['”']]))


def does_term_are_service_words(term_pos: Iterable[Tuple[str, str, int, int]]) -> bool:
    """Return True if the term contains only service (non significant) parts of speech."""
    for _, pos, _, _ in term_pos:
        if pos not in NON_SIG_POS:
            return False
    return True


def trim_defined_term(
    term: str,
    start: int,
    end: int,
    registry: DefinitionPatternRegistry = DEFAULT_PATTERN_REGISTRY,
) -> Tuple[str, int, int, bool]:
    """Strip framing symbols and whitespace from a candidate definition term."""
    quoted_text_re = registry.get('quoted_text').regex
    spaces_re = registry.get('spaces').regex
    abbreviation_re = registry.get('abbreviation_ending').regex
    strip_symbols = registry.strip_punct_symbols

    was_quoted = False

    quoted_parts = [match.group() for match in quoted_text_re.finditer(term)]
    if len(quoted_parts) == 1:
        term = quoted_parts[0].strip("\"'“„")
        was_quoted = True

    original_length = len(term)
    original_quote_count = count_sequence_matches(term, lambda char: char in TextBeautifier.QUOTES)
    term, start, end = TextBeautifier.strip_pair_symbols((term, start, end))
    if len(term) < original_length:
        updated_quote_count = count_sequence_matches(term, lambda char: char in TextBeautifier.QUOTES)
        was_quoted = was_quoted or original_quote_count - updated_quote_count > 1

    term = term.replace('\n', ' ')
    term = spaces_re.sub(' ', term)

    term, start, end = TextBeautifier.strip_string_coords(term, start, end, strip_symbols)

    if not abbreviation_re.search(term):
        term, start, end = TextBeautifier.strip_string_coords(term, start, end, '.')
    else:
        term, start, end = TextBeautifier.lstrip_string_coords(term, start, end, '.')

    return term, start, end, was_quoted


def split_definitions_inside_term(
    term: str,
    src_with_coords: Tuple[int, int, str],
    term_start: int,
    term_end: int,
    registry: DefinitionPatternRegistry = DEFAULT_PATTERN_REGISTRY,
) -> List[Tuple[str, int, int]]:
    """Split a phrase containing multiple quoted definitions into separate spans."""
    src_start = src_with_coords[0]
    src_text = src_with_coords[2]

    matches = [match.group() for match in registry.get('split_subdefinitions').regex.finditer(term)]
    if len(matches) < 2:
        matches = [term]

    match_coords = PhrasePositionFinder.find_phrase_in_source_text(
        src_text, matches, term_start - src_start, term_end - src_start
    )

    if len(match_coords) < len(matches):
        return [(term, term_start, term_end)]

    return [(match[0], match[1] + src_start, match[2] + src_start) for match in match_coords]


def regex_matches_to_word_coords(pattern, text: str, phrase_start: int = 0) -> List[Tuple[str, int, int]]:
    """Convert regex matches into (text, start, end) tuples with coordinates."""
    return [
        (match.group(), match.start() + phrase_start, match.end() + phrase_start)
        for match in pattern.finditer(text)
    ]


__all__ = [
    'NON_SIG_POS',
    'PUNCTUATION_POS',
    'PUNCTUATION_STRIP_STR',
    'does_term_are_service_words',
    'get_quotes_count_in_string',
    'trim_defined_term',
    'split_definitions_inside_term',
    'regex_matches_to_word_coords',
]
