"""Coverage tests for :mod:`lexnlp.extract.en.constraints` wrappers and strict mode."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.en.constraints import (
    get_constraint_annotation_list,
    get_constraint_annotations,
    get_constraint_list,
)


def test_get_constraint_list() -> None:
    assert get_constraint_list("The value is within limits.") == [("within", "the value is", "")]
    assert get_constraint_list("no identifiers here at all") == []


def test_get_constraint_annotation_list() -> None:
    annotations = get_constraint_annotation_list("The value is within limits.")
    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.constraint == "within"
    assert annotation.pre == "the value is"
    assert annotation.post == ""
    assert annotation.coords == (0, 20)


def test_strict_mode_skips_bare_trigger_sentence() -> None:
    relaxed = list(get_constraint_annotations("Maximum"))
    assert len(relaxed) == 1
    assert relaxed[0].constraint == "maximum"
    assert relaxed[0].pre == ""
    assert relaxed[0].post == ""
    # A lone trigger has neither pre nor post context, so strict mode drops it.
    assert list(get_constraint_annotations("Maximum", strict=True)) == []
    assert get_constraint_list("Maximum", strict=True) == []


def test_strict_mode_keeps_trigger_with_context() -> None:
    relaxed = get_constraint_list("The value is within limits.", strict=True)
    assert relaxed == [("within", "the value is", "")]


def test_comma_joined_prefix_combines_into_longer_phrase() -> None:
    # "no,less than" cannot match "no less than" directly (comma instead of
    # a space), so the regex matches short "less than" with pre "no"; the
    # combined "no less than" is a known phrase and becomes the constraint.
    assert get_constraint_list("no,less than 5 dollars are due") == [("no less than", "no", "")]
    annotations = get_constraint_annotation_list("no,less than 5 dollars are due")
    assert len(annotations) == 1
    assert annotations[0].constraint == "no less than"
    assert annotations[0].pre == "no"
    assert annotations[0].post == ""
    assert annotations[0].coords == (0, 13)


def test_comma_joined_at_least_combines() -> None:
    assert get_constraint_list("at,least five dollars") == [("at least", "at", "")]
    annotations = get_constraint_annotation_list("at,least five dollars")
    assert len(annotations) == 1
    assert annotations[0].constraint == "at least"
    assert annotations[0].pre == "at"
    assert annotations[0].post == ""
    assert annotations[0].coords == (0, 9)


def test_uncombinable_prefix_keeps_short_match() -> None:
    # pre + constraint is not a known phrase, so the short match is kept.
    assert get_constraint_list("value no,less than five") == [("less than", "value no", "")]
    assert get_constraint_list("pay at,least five dollars now") == [("least", "pay at", "")]
