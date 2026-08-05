"""Paragraph segmentation for English.

This module implements paragraph segmentation in English using simple
machine learning classifiers.

Todo:
  * Standardize model (re-)generation
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


# standard library imports
import os
import string
import unicodedata
from re import Pattern, compile as re_compile
from typing import Dict, Final, Generator, List, Set, Tuple, Union, Optional

# third-party imports
from pandas import DataFrame

# LexNLP
from lexnlp.nlp.en.segments.utils import (
    TRAINED_LINE_WINDOW_POST,
    TRAINED_LINE_WINDOW_PRE,
    build_document_line_distribution,
    has_compatible_feature_width,
    has_compatible_line_window,
)
from lexnlp.utils.unpickler import load_joblib_model


# Setup module path


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load segmenters
PARAGRAPH_SEGMENTER_MODEL: Final = load_joblib_model(
    os.path.join(MODULE_PATH, "./paragraph_segmenter.pickle")
)

# regular expression for newlines
RE_NEW_LINE: Final[Pattern] = re_compile(r'(?P<line>[^\r\n]*)((\r\n)|(\n\r)|\n|\r)')


def build_paragraph_break_features(
    lines: List[str],
    line_id: int,
    line_window_pre: int,
    line_window_post: int,
    characters=string.printable,
    include_doc=None,
) -> Dict[str, Union[int, bool]]:
    """
    Build a feature vector for a given line ID with given parameters.
    """
    # Feature vector
    feature_vector = {}

    # Keep the global schema fixed while clipping unavailable values per row.
    line_window_pre = min(line_window_pre, line_id)
    line_window_post = min(line_window_post, len(lines) - line_id - 1)

    # Iterate through window
    for i in range(-line_window_pre, line_window_post + 1):
        try:
            line = lines[line_id + i]
        except IndexError:
            continue

        # Count length
        feature_vector["line_len_{0}".format(i)] = len(line)
        feature_vector["line_lenstrip_{0}".format(i)] = len(line.strip())
        feature_vector["line_title_case_{0}".format(i)] = line == line.title()
        feature_vector["line_upper_case_{0}".format(i)] = line == line.upper()

        # Count characters
        feature_vector["line_n_alpha_{0}".format(i)] = sum([1 for c in line if unicodedata.category(c).startswith("L")])
        feature_vector["line_n_number_{0}".format(i)] = sum(
            [1 for c in line if unicodedata.category(c).startswith("N")])
        feature_vector["line_n_punct_{0}".format(i)] = sum([1 for c in line if unicodedata.category(c).startswith("P")])
        feature_vector["line_n_whitespace_{0}".format(i)] = sum(
            [1 for c in line if unicodedata.category(c).startswith("Z")])

    # Simple checks
    line = lines[line_id]
    line_stripped = line.strip()
    len_line_stripped = len(line_stripped)
    feature_vector["first_char_punct"] = (line_stripped[0] in string.punctuation) if len_line_stripped > 0 else False
    feature_vector["last_char_punct"] = (line_stripped[-1] in string.punctuation) if len_line_stripped > 0 else False
    feature_vector["first_char_number"] = (line_stripped[0] in string.digits) if len_line_stripped > 0 else False
    feature_vector["last_char_number"] = (line_stripped[-1] in string.digits) if len_line_stripped > 0 else False

    # Build character vector
    for character in characters:
        feature_vector["char_{0}".format(character)] = lines[line_id].count(character)

    # Add doc if requested
    if include_doc:
        feature_vector.update(include_doc)

    return feature_vector


def get_paragraph_break_feature_names(
    lines_count: int,
    line_window_pre: int,
    line_window_post: int,
    characters=string.printable,
    include_doc=None
) -> Set[str]:
    """
    Build a feature vector for a given line ID with given parameters.
    """
    # Feature vector
    feature_vector: Set[str] = {
        'first_char_punct',
        'last_char_punct',
        'first_char_number',
        'last_char_number',
    }

    # The fitted model owns one fixed global offset schema. Missing edge-row
    # values are filled later; document length must never remove columns.
    # ``lines_count`` remains in the public signature for compatibility.
    _ = lines_count

    # Iterate through the complete requested window
    for i in range(-line_window_pre, line_window_post + 1):

        # Count length
        feature_vector.add(f'line_len_{i}')
        feature_vector.add(f'line_lenstrip_{i}')
        feature_vector.add(f'line_title_case_{i}')
        feature_vector.add(f'line_upper_case_{i}')
        # Count characters
        feature_vector.add(f'line_n_alpha_{i}')
        feature_vector.add(f'line_n_number_{i}')
        feature_vector.add(f'line_n_punct_{i}')
        feature_vector.add(f'line_n_whitespace_{i}')

    # Build character vector
    for character in characters:
        feature_vector.add(f"char_{character}")

    # Add doc if requested
    if include_doc:
        feature_vector.update(set(include_doc.keys()))

    return feature_vector


def splitlines_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    lines: List[str] = []
    spans: List[Tuple[int, int]] = []
    if text is None:
        return lines, spans
    # Start from offset 0 so single-line inputs without newlines keep full text.
    last_line_end = 0
    for m in RE_NEW_LINE.finditer(text):
        line = m.group('line')
        span = m.span()
        lines.append(line)
        spans.append(span)
        last_line_end = span[1]
    if last_line_end < len(text):
        lines.append(text[last_line_end:len(text)])
        spans.append((last_line_end, len(text)))
    return lines, spans


def _normalise_paragraph_breaks(
    lines: List[str],
    predicted_breaks: List[int],
) -> List[int]:
    """Apply deterministic ownership for blank-line separator runs.

    Leading separators belong to the first content paragraph, trailing
    separators to the last, and an internal separator run belongs to the
    preceding paragraph. Therefore an internal run creates exactly one break
    at the following nonblank line, regardless of model scores inside the run.
    """

    breaks = set(predicted_breaks)
    seen_content = False
    blank_start: Optional[int] = None

    for line_id, line in enumerate(lines):
        if not line.strip():
            if blank_start is None:
                blank_start = line_id
            continue

        if blank_start is not None:
            if seen_content:
                breaks.difference_update(range(blank_start, line_id))
                breaks.add(line_id)
            else:
                # A leading run and its first content line form one paragraph.
                breaks.difference_update(range(blank_start, line_id + 1))
            blank_start = None
        seen_content = True

    if blank_start is not None:
        # Never strand a trailing separator run in a filtered blank paragraph.
        breaks.difference_update(range(blank_start, len(lines)))

    content_prefix = [0]
    for line in lines:
        content_prefix.append(content_prefix[-1] + int(bool(line.strip())))

    accepted: List[int] = []
    previous = 0
    for boundary in sorted(breaks):
        if boundary <= 0 or boundary >= len(lines):
            continue
        if not lines[boundary].strip():
            continue
        if content_prefix[boundary] == content_prefix[previous]:
            continue
        accepted.append(boundary)
        previous = boundary
    return accepted


def _form_potential_paragraph(
    pos0: int,
    pos1: Optional[int],
    text: str,
    line_spans: List[Tuple[int, int]],
) -> Optional[Tuple[int, int, str]]:
    """
    """
    span: Tuple[int, int] = (
        line_spans[pos0][0],
        line_spans[pos1][0] if pos1 is not None else len(text)
    )
    paragraph = text[span[0]:span[1]]
    if len(paragraph.strip()) > 0:
        return span[0], span[1], paragraph


def get_paragraph_spans(
    text: str,
    window_pre=TRAINED_LINE_WINDOW_PRE,
    window_post=TRAINED_LINE_WINDOW_POST,
    score_threshold=0.5,
) -> Generator[Tuple[int, int, str], None, None]:
    """
    Get paragraph spans (start, end, paragraph) from text.

    Args:
        text (str):
            Input text whence to extract paragraphs.

        window_pre (int=3):
            The left-side line window distance.

        window_post (int=3):
            The right-side line window distance.

        score_threshold (float=0.5):
            The minimum probability a predicted paragraph break must meet in order
            to be considered a valid paragraph break.
    """
    lines, line_spans = splitlines_with_spans(text)
    if not lines:
        return
    if not has_compatible_line_window(window_pre, window_post):
        if text:
            yield 0, len(text), text
        return

    # Get document character distribution only for an eligible model schema.
    doc_distribution: Dict[str, float] = build_document_line_distribution(text)
    feature_data: List[Dict] = [
        build_paragraph_break_features(
            lines=lines,
            line_id=line_id,
            line_window_pre=window_pre,
            line_window_post=window_post,
            include_doc=doc_distribution,
        )
        for line_id in range(len(lines))
    ]

    # Predict page breaks
    column_names = list(
        get_paragraph_break_feature_names(
            lines_count=len(lines),
            line_window_pre=window_pre,
            line_window_post=window_post,
            include_doc=doc_distribution)
    )
    column_names.sort()
    feature_df: DataFrame = DataFrame(feature_data, columns=column_names).fillna(-1).astype(int)
    if not has_compatible_feature_width(
        PARAGRAPH_SEGMENTER_MODEL,
        feature_df.shape[1],
    ):
        # Preserve the historical model-schema mismatch fallback.
        if text:
            yield 0, len(text), text
        return

    try:
        # Avoid pandas dtype deprecation noise in sklearn validation by passing a numpy array.
        predicted_lines = PARAGRAPH_SEGMENTER_MODEL.predict_proba(feature_df.to_numpy())
        predicted_df: DataFrame = DataFrame(predicted_lines, columns=["prob_false", "prob_true"])
        predicted_breaks = predicted_df.loc[
            predicted_df["prob_true"] >= score_threshold,
            :,
        ].index.tolist()
        paragraph_breaks = _normalise_paragraph_breaks(lines, predicted_breaks)

        if len(paragraph_breaks) > 0:
            # Get first break
            pos0 = 0
            pos1 = paragraph_breaks[0]

            maybe_paragraph = _form_potential_paragraph(pos0, pos1, text, line_spans)
            if maybe_paragraph is not None:
                yield maybe_paragraph

            # Iterate through section breaks
            for i in range(len(paragraph_breaks) - 1):
                # Get breaks
                pos0 = paragraph_breaks[i]
                pos1 = paragraph_breaks[i + 1]
                # Get text
                maybe_paragraph = _form_potential_paragraph(pos0, pos1, text, line_spans)
                if maybe_paragraph is not None:
                    yield maybe_paragraph

            # Yield final section
            pos0 = paragraph_breaks[-1]
            pos1 = None
            maybe_paragraph = _form_potential_paragraph(pos0, pos1, text, line_spans)
            if maybe_paragraph is not None:
                yield maybe_paragraph
        else:
            yield 0, len(text), text
    except ValueError as e:
        if 'Number of features of the model must match the input' in str(e):
            yield 0, len(text), text
        else:
            raise e


def get_paragraph_span_list(
    text: str,
    window_pre=TRAINED_LINE_WINDOW_PRE,
    window_post=TRAINED_LINE_WINDOW_POST,
    score_threshold=0.5,
) -> List[Tuple[int, int, str]]:
    """
    Get a list of paragraph spans (start, end, paragraph) from text.

    Args:
        text (str):
            Input text whence to extract paragraphs.

        window_pre (int=3):
            The left-side line window distance.

        window_post (int=3):
            The right-side line window distance.

        score_threshold (float=0.5):
            The minimum probability a predicted paragraph break must meet in order
            to be considered a valid paragraph break.
    """
    return list(
        get_paragraph_spans(
            text=text,
            window_pre=window_pre,
            window_post=window_post,
            score_threshold=score_threshold,
        )
    )


def get_paragraphs(
    text: str,
    window_pre=TRAINED_LINE_WINDOW_PRE,
    window_post=TRAINED_LINE_WINDOW_POST,
    score_threshold=0.5,
) -> Generator[str, None, None]:
    """
    Get paragraphs from text.

    Args:
        text (str):
            Input text whence to extract paragraphs.

        window_pre (int=3):
            The left-side line window distance.

        window_post (int=3):
            The right-side line window distance.

        score_threshold (float=0.5):
            The minimum probability a predicted paragraph break must meet in order
            to be considered a valid paragraph break.
    """
    for _, _, paragraph in get_paragraph_spans(
        text=text,
        window_pre=window_pre,
        window_post=window_post,
        score_threshold=score_threshold,
    ):
        yield paragraph


def get_paragraph_list(
    text: str,
    window_pre=TRAINED_LINE_WINDOW_PRE,
    window_post=TRAINED_LINE_WINDOW_POST,
    score_threshold=0.5,
) -> List[str]:
    """
    Get a list of paragraphs from text.

    Args:
        text (str):
            Input text whence to extract paragraphs.

        window_pre (int=3):
            The left-side line window distance.

        window_post (int=3):
            The right-side line window distance.

        score_threshold (float=0.5):
            The minimum probability a predicted paragraph break must meet in order
            to be considered a valid paragraph break.
    """
    return list(
        get_paragraphs(
            text=text,
            window_pre=window_pre,
            window_post=window_post,
            score_threshold=score_threshold,
        )
    )
