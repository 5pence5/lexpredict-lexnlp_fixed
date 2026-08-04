"""Utility methods for segmentation classifiers

This module implements utility methods for segmentation, such as shared methods to generate
document character distributions.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


import string
from numbers import Integral
from typing import Dict, Union
from lexnlp.utils.decorators import handle_invalid_text


TRAINED_LINE_WINDOW_PRE = 3
TRAINED_LINE_WINDOW_POST = 3


def has_compatible_line_window(
    actual_pre: int,
    actual_post: int,
    trained_pre: int = TRAINED_LINE_WINDOW_PRE,
    trained_post: int = TRAINED_LINE_WINDOW_POST,
) -> bool:
    """Return whether line-offset semantics match the fitted model schema."""

    values = {
        "actual_pre": actual_pre,
        "actual_post": actual_post,
        "trained_pre": trained_pre,
        "trained_post": trained_post,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    return (int(actual_pre), int(actual_post)) == (
        int(trained_pre),
        int(trained_post),
    )


def _estimator_feature_widths(estimator: object) -> frozenset[int]:
    """Return every positive fitted width declared by one estimator."""

    widths = set()
    for attribute in ("n_features_in_", "n_features_"):
        value = getattr(estimator, attribute, None)
        if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
            widths.add(int(value))

    tree = getattr(estimator, "tree_", None)
    value = getattr(tree, "n_features", None)
    if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
        widths.add(int(value))
    return frozenset(widths)


def resolve_model_feature_width(model: object) -> int | None:
    """Resolve one consistent fitted width across an estimator or forest.

    Some re-exported ExtraTrees artifacts do not expose ``n_features_in_`` on
    the forest itself. In that case every fitted child must independently
    declare the same width. Missing or contradictory metadata fails closed.
    """

    direct_widths = _estimator_feature_widths(model)
    if len(direct_widths) > 1:
        return None

    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return next(iter(direct_widths)) if direct_widths else None

    flattened = getattr(estimators, "flat", None)
    try:
        children = tuple(flattened if flattened is not None else estimators)
    except TypeError:
        return None
    if not children:
        return None

    child_width_sets = tuple(
        _estimator_feature_widths(child) for child in children
    )
    if any(len(widths) != 1 for widths in child_width_sets):
        return None
    widths = set(direct_widths)
    widths.update(next(iter(child_widths)) for child_widths in child_width_sets)
    return next(iter(widths)) if len(widths) == 1 else None


def has_compatible_feature_width(model: object, actual_width: int) -> bool:
    """Fail closed unless a matrix exactly matches fitted estimator metadata."""

    if isinstance(actual_width, bool) or not isinstance(actual_width, Integral):
        raise TypeError("actual_width must be an integer")
    if actual_width < 0:
        raise ValueError("actual_width must not be negative")
    expected_width = resolve_model_feature_width(model)
    return expected_width is not None and int(actual_width) == expected_width


@handle_invalid_text(return_value={})
def build_document_distribution(
    text: str,
    characters=string.printable,
    norm=True
) -> Dict[str, Union[int, float]]:
    """
    Build document character distribution based on fixed character, optionally norming.
    :param text:
    :param characters:
    :param norm:
    :return:
    """
    # Build character vector
    char_vector = {}
    for character in characters:
        char_vector["doc_char_{0}".format(character)] = text.count(character)

    # Norm if requested
    if norm:
        total = float(sum(char_vector.values()))
        for key in char_vector:
            char_vector[key] = char_vector[key] / total

    return char_vector


@handle_invalid_text(return_value={})
def build_document_line_distribution(
    text: str,
    characters=string.printable,
    norm=True
) -> Dict[str, Union[int, float]]:
    """
    Build document and line character distribution for section segmenting based
    on fixed character, optionally normalizing vector.
    """

    # Build character vector
    feature_vector = {}
    for character in characters:
        feature_vector[f"doc_char_{character}"] = text.count(character)
        feature_vector[f"doc_startchar_{character}"] = 0
    feature_vector["doc_startchar_other"] = 0

    # Build line start vector
    for line in text.splitlines():
        if len(line.strip()) > 0:
            character = line.strip()[0]
            if character in characters:
                feature_vector["doc_startchar_{0}".format(character)] += 1
            else:
                feature_vector["doc_startchar_other"] += 1
        else:
            continue

    # Norm if requested
    if norm:
        total_char = float(sum([b for a, b in feature_vector.items() if a.startswith("doc_char")]))
        total_startchar = float(sum([b for a, b in feature_vector.items() if a.startswith("doc_startchar")]))

        for character in feature_vector.keys():
            if character.startswith("doc_char"):
                feature_vector[character] = feature_vector[character] / total_char
            elif character.startswith("doc_startchar"):
                feature_vector[character] = feature_vector[character] / total_startchar if total_startchar != 0.0 else 0.0

    return feature_vector
