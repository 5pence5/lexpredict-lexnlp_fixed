"""Coverage tests for lexnlp.extract.de.durations missing lines."""

from decimal import Decimal

from lexnlp.extract.common.annotations.duration_annotation import DurationAnnotation
from lexnlp.extract.de.durations import (
    DeDurationParser,
    get_duration_annotations,
    get_duration_annotations_list,
    get_duration_list,
)


def test_get_duration_annotations_list_simple() -> None:
    text = "seit fünfundzwanzig Jahren"
    actual = get_duration_annotations_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert isinstance(ant, DurationAnnotation)
    assert ant.coords == (4, 26)
    assert ant.duration_type == "jahren"
    assert ant.duration_type_en == "year"
    assert ant.amount == Decimal(25)
    assert ant.duration_days == Decimal(9125)
    assert ant.locale == "de"
    expected = list(get_duration_annotations(text))
    assert len(expected) == 1
    assert actual[0].coords == expected[0].coords
    assert actual[0].text == expected[0].text
    assert actual[0].duration_type == expected[0].duration_type


def test_get_duration_annotations_list_complex() -> None:
    text = "Vier Wochen, 3 Tage und 151 Sekunden."
    actual = get_duration_annotations_list(text)
    assert isinstance(actual, list)
    assert len(actual) == 1
    ant = actual[0]
    assert ant.duration_days == Decimal("31.0017")
    assert ant.is_complex is True
    assert ant.duration_type_en == "second"
    assert ant.duration_type == "sekunden"
    assert ant.value_dict == {"wochen": 4.0, "tage": 3.0, "sekunden": 151.0}


def test_get_duration_annotations_list_empty() -> None:
    assert get_duration_annotations_list("Kein Dauerwert hier") == []
    assert get_duration_annotations_list("") == []


def test_unicode_lookalike_unit_match_is_skipped() -> None:
    # U+017F (long s) matches the case-insensitive duration pattern, but
    # str.lower() preserves it, so the case-sensitive unit lookup finds
    # nothing and the match is skipped via `continue`.
    text = "3 ſekunden"
    assert len(list(DeDurationParser.DURATION_PTN_RE.finditer(text))) == 1
    assert DeDurationParser.get_annotations(text) == []
    assert get_duration_annotations_list(text) == []
    assert get_duration_list(text) == []


def test_unicode_lookalike_dotted_capital_i_match_is_skipped() -> None:
    # U+0130 matches "i" case-insensitively, but lower() yields "i" + a
    # combining dot, which the case-sensitive lookup does not contain.
    text = "Mİnute"
    assert len(list(DeDurationParser.DURATION_PTN_RE.finditer(text))) == 1
    assert DeDurationParser.get_annotations(text) == []


def test_ascii_lookalike_control_yields_annotation() -> None:
    text = "3 sekunden"
    actual = get_duration_annotations_list(text)
    assert len(actual) == 1
    ant = actual[0]
    assert ant.text == "3 sekunden"
    assert ant.duration_type == "sekunden"
    assert ant.duration_type_en == "second"
    assert ant.amount == Decimal("3.0")
    assert ant.duration_days == Decimal("0.0000")
