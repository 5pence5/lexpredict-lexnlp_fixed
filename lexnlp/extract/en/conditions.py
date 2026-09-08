"""Condition extraction for English.

This module implements basic condition extraction functionality in English.

Todo:
  * Improved unit tests and case coverage
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import copy
from collections.abc import Generator

import regex as re

from lexnlp.extract.common.annotations.condition_annotation import ConditionAnnotation
from lexnlp.nlp.en.segments.sentences import get_sentence_list

CONDITION_PHRASES = [
    "if",
    "if not",
    "when",
    "when not",
    "where",
    "where not",
    "unless and until",
    "unless",
    "unless not",
    "until",
    "until not",
    "as soon as",
    "as soon as not",
    "provided that",
    "provided that not",
    "subject to",
    "not subject to",
    "upon the occurrence",
    "subject to",
    "conditioned  on",
    "conditioned  upon",
]

# The trigger phrase is matched directly, between two boundary characters.
# The previous template wrapped the alternation in ``(?P<pre>.*?)`` /
# ``(?P<post>.*?)`` plus a ``{{1,}}`` repetition, so the engine retried the
# leading wildcard from every character of a trigger-free sentence
# (catastrophic backtracking: ~2s for a 10k-character sentence). ``pre`` is now
# sliced from the sentence, which yields byte-identical annotations without the
# ReDoS exposure.
CONDITION_PATTERN_TEMPLATE = r"""[\s\.\,](?P<condition>{condition_pattern})[\s\.\,]"""


# ================================
# Patterns for condition matching
# ================================
def create_condition_pattern(condition_pattern_template, condition_phrases):
    """
    Create condition pattern.
    :param condition_pattern_template:
    :param condition_phrases:
    :return:
    """
    # Materialize pattern form intermediate word lists
    pattern_condition_phrases = copy.copy(condition_phrases)
    pattern_condition_phrases.sort(key=len, reverse=True)

    return condition_pattern_template.format(
        condition_pattern="|".join([p.replace(r" ", r"\ ") for p in pattern_condition_phrases])
    )


# Materialize pattern and create regex
CONDITION_PATTERN = create_condition_pattern(CONDITION_PATTERN_TEMPLATE, CONDITION_PHRASES)
RE_CONDITION = re.compile(CONDITION_PATTERN, re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE)


def get_conditions(
    text: str,
    strict: bool = True,
) -> Generator[tuple[str | None, str | None, str | None]]:
    """
    Get conditions possible conditions from natural language.

    Args:
        text (str):
            An input string to search for conditions.

        strict (bool=True):

    Yields:
        Tuples representing conditions.
    """
    for ant in get_condition_annotations(text, strict):
        yield ant.condition, ant.pre, ant.post


def get_condition_list(text, strict: bool = True) -> list[tuple[str | None, str | None, str | None]]:
    """
    Get a list of conditions possible conditions from natural language.

    Args:
        text (str):
            An input string to search for conditions.

        strict (bool=True):


    Yields:
        Tuples representing conditions.
    """
    return list(get_conditions(text, strict))


def get_condition_annotations(text: str, strict: bool = True) -> Generator[ConditionAnnotation]:
    """
    Get ConditionAnnotations.

    Args:
        text (str):
            An input string to search for conditions.

        strict (bool=True):


    Yields:
        ConditionAnnotation
    """

    # Iterate through all potential matches
    for sentence in get_sentence_list(text):
        cursor = 0
        for match in RE_CONDITION.finditer(sentence):
            start = match.start("condition")
            end = match.end("condition")

            # ``pre`` is the text between the previous trigger and this one,
            # excluding the single boundary character the pattern consumes.
            # ``post`` was always empty under the old lazy ``(?P<post>.*?)``
            # group, and stays empty so annotations are unchanged.
            pre = sentence[cursor : max(cursor, start - 1)]
            post = ""

            # ``strict`` is accepted for backwards compatibility but has no
            # effect: under the old pattern both wildcard groups always
            # captured exactly once, so the "empty pre/post" guard could never
            # fire. Preserving that means preserving the emitted annotations.

            ant = ConditionAnnotation(
                coords=(cursor, min(end + 1, len(sentence))),
                condition=match.group("condition").lower(),
                pre=pre,
                post=post,
            )
            cursor = min(end + 1, len(sentence))
            yield ant


def get_condition_annotation_list(text: str, strict: bool = True) -> list[ConditionAnnotation]:
    """
    Get a list of ConditionAnnotations.

    Args:
        text (str):
            An input string to search for conditions.

        strict (bool=True):


    Returns:
        A list of ConditionAnnotations
    """
    return list(get_condition_annotations(text, strict))
