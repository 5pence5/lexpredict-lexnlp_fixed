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
