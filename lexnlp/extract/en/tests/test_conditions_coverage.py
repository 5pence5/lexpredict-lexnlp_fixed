"""Coverage tests for lexnlp.extract.en.conditions missing lines."""

from lexnlp.extract.common.annotations.condition_annotation import ConditionAnnotation
from lexnlp.extract.en.conditions import (
    get_condition_annotation_list,
    get_condition_annotations,
    get_condition_list,
    get_conditions,
)


def test_get_condition_list_single_trigger() -> None:
    text = "Hello if you come, we will go."
    actual = get_condition_list(text)
    assert actual == [("if", "Hello", "")]
    assert actual == list(get_conditions(text))


def test_get_condition_list_empty() -> None:
    assert get_condition_list("No trigger here.") == []
    assert get_condition_list("") == []
    assert get_condition_annotation_list("No trigger here.") == []
    assert get_condition_annotation_list("") == []


def test_get_condition_annotation_list_fields() -> None:
    text = "Hello if you come, we will go."
    actual = get_condition_annotation_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, ConditionAnnotation)
    assert ant.condition == "if"
    assert ant.pre == "Hello"
    assert ant.post == ""
    assert ant.coords == (0, 9)
    assert ant.locale == "en"
    expected = list(get_condition_annotations(text))
    assert len(expected) == 1
    assert actual[0].condition == expected[0].condition
    assert actual[0].pre == expected[0].pre
    assert actual[0].coords == expected[0].coords


def test_get_condition_list_variants_and_strict() -> None:
    text = "Payment is due unless and until delivery occurs."
    assert get_condition_list(text) == [("unless and until", "Payment is due", "")]
    assert get_condition_list(text, strict=False) == get_condition_list(text, strict=True)
    text2 = "The agreement is subject to approval."
    assert get_condition_list(text2) == [("subject to", "The agreement is", "")]
    ants = get_condition_annotation_list(text2)
    assert len(ants) == 1
    assert ants[0].condition == "subject to"
    assert ants[0].pre == "The agreement is"
    assert ants[0].post == ""
