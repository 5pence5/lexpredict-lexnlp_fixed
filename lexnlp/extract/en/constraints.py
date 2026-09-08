"""Constraint extraction for English.

This module implements basic constraint extraction functionality in English.

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

from lexnlp.extract.common.annotations.constraint_annotation import ConstraintAnnotation
from lexnlp.nlp.en.segments.sentences import get_sentence_list

CONSTRAINT_PHRASES = [
    "after",
    "at least",
    "at most",
    "before",
    "equal to",
    "exactly",
    "first of",
    "greater",
    "greater of",
    "greater than",
    "greater than or equal to",
    "greatest of",
    "last of",
    "least of",
    "lesser",
    "lesser of",
    "lesser than",
    "less than",
    "less than or equal to",
    "maximum of",
    "maximum",
    "minimum of",
    "minimum",
    "more than",
    "more than or equal to",
    "no earlier than",
    "no later than",
    "no less than",
    "no more than",
    "not equal to",
    "not to exceed",
    "earlier than",
    "later than",
    "within",
    "exceed",
    "exceeds",
    "prior to",
    "highest",
    "least",
]

# The trigger phrase is matched directly, between boundary characters. The
# previous template nested ``(?P<pre>.*?)``, ``(?P<post>.)*?`` and a lazy
# ``)+?`` repetition around the alternation, which backtracks catastrophically
# on trigger-free input (~2s for a 10k-character sentence). ``pre`` and
# ``post`` are now sliced from the sentence, reproducing the old annotations
# exactly without the ReDoS exposure. The leading boundary is optional so a
# constraint at the very start of a sentence is still found, which is what the
# second branch of the old alternation did.
CONSTRAINT_PATTERN_TEMPLATE = r"""
(?:^|[\s\.\,\;])(?P<constraint>{constraint_pattern})(?=[\s\.\,\;]|$)
"""


# ================================
# Patterns for duration matching
# ================================
def create_constraint_pattern(constraint_pattern_template, constraint_phrases):
    """
    Create constraint pattern.
    :param constraint_pattern_template:
    :param constraint_phrases:
    :return:
    """
    # Materialize pattern form intermediate word lists
    pattern_constraint_phrases = copy.copy(constraint_phrases)
    pattern_constraint_phrases.sort(key=len, reverse=True)

    return constraint_pattern_template.format(
        constraint_pattern="|".join([p.replace(r" ", r"\ ") for p in pattern_constraint_phrases])
    )


# Materialize pattern and create regex
CONSTRAINT_PATTERN = create_constraint_pattern(CONSTRAINT_PATTERN_TEMPLATE, CONSTRAINT_PHRASES)
RE_CONSTRAINT = re.compile(CONSTRAINT_PATTERN, re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE)


def get_constraints(
    text: str,
    strict: bool = False,
) -> Generator[tuple[str | None, str | None, str | None]]:
    """
    Find possible constraints in natural language.
    :param text:
    :param strict:
    :return:
    """

    # Iterate through all potential matches
    for ant in get_constraint_annotations(text, strict):
        yield ant.constraint, ant.pre, ant.post


def get_constraint_list(
    text: str,
    strict: bool = False,
) -> list[tuple[str | None, str | None, str | None]]:
    """
    Find possible constraints in natural language.
    :param text:
    :param strict:
    :return:
    """
    return list(get_constraints(text, strict))


def get_constraint_annotations(text: str, strict: bool = False) -> Generator[ConstraintAnnotation]:
    """
    Find possible constraints in natural language.
    :param text:
    :param strict:
    :return:
    """

    # Iterate through all potential matches
    for sentence in get_sentence_list(text):
        lowered = sentence.lower()
        cursor = 0
        for match in RE_CONSTRAINT.finditer(lowered):
            start = match.start("constraint")
            end = match.end("constraint")
            constraint = match.group("constraint").lower()

            if start == 0:
                # Trigger at the very start of the sentence: the old pattern's
                # second branch captured the whole remainder as ``post``.
                pre = ""
                post = lowered[min(end + 1, len(lowered)) :]
                coords = (cursor, len(lowered))
            else:
                # Trigger inside the sentence: the old first branch captured
                # the preceding text as ``pre`` and left ``post`` empty,
                # because ``(?P<post>.)*?`` is lazy.
                pre = lowered[cursor : max(cursor, start - 1)]
                post = ""
                coords = (cursor, min(end + 1, len(lowered)))

            # Skip if strict and empty pre/post
            if strict and not pre and not post:
                cursor = coords[1]
                continue

            # A trailing word of ``pre`` may complete a longer phrase, e.g.
            # "no less than" where "less than" matched on its own.
            if not post and pre:
                combined = f"{pre} {constraint}".lower().strip()
                if combined in CONSTRAINT_PHRASES:
                    constraint = combined

            ant = ConstraintAnnotation(coords=coords, constraint=constraint, pre=pre, post=post)
            cursor = coords[1]
            yield ant


def get_constraint_annotation_list(text: str, strict: bool = False) -> list[ConstraintAnnotation]:
    """
    Find possible constraints in natural language.
    :param text:
    :param strict:
    :return: A list of ConstraintAnnotations
    """
    return list(get_constraint_annotations(text, strict))
