"""The court keyword guard must not treat ``re.IGNORECASE`` as a start offset.

``UniversalCourtsParser.parse`` short-circuits when the configured keyword
(``tribunal``, ``juízo``, ``gericht`` ...) does not appear. The guard called
``pattern.search(text, re.IGNORECASE)`` -- but the second positional argument
of :meth:`re.Pattern.search` is ``pos``, not ``flags``. ``re.IGNORECASE`` is
``2``, so the scan silently started at offset 2 and every court whose keyword
began at index 0 or 1 was discarded.

That is why ``"Tribunal Superior de Justicia de Andalucía"`` -- row 1 of
``es_courts.csv`` -- extracted nothing, while ``"Supremo Tribunal Federal"``
(keyword at index 8) worked.
"""

from __future__ import annotations

import pytest

from lexnlp.extract.es.courts import get_court_annotations as get_es_courts
from lexnlp.extract.pt.courts import get_court_annotations as get_pt_courts


@pytest.mark.parametrize(
    "text",
    [
        "Tribunal Superior de Justicia de Andalucía",
        "Tribunal Superior de Justicia de Aragón",
        "Tribunal Superior de Justicia de Canarias",
    ],
)
def test_spanish_court_at_offset_zero_is_found(text: str) -> None:
    """These are verbatim rows of es_courts.csv and must round-trip."""
    found = [annotation.name for annotation in get_es_courts(text)]
    assert found, f"court starting at index 0 was dropped: {text!r}"


def test_spanish_court_is_still_found_when_not_at_offset_zero() -> None:
    found = [a.name for a in get_es_courts("ante el Tribunal Superior de Justicia de Madrid")]
    assert found


@pytest.mark.parametrize(
    "text",
    [
        "Tribunal Superior do Trabalho",  # keyword at index 0 -- was dropped
        "Supremo Tribunal Federal",  # keyword at index 8 -- always worked
    ],
)
def test_portuguese_courts_found_regardless_of_keyword_offset(text: str) -> None:
    assert [a.name for a in get_pt_courts(text)], f"dropped: {text!r}"


def test_text_without_the_keyword_still_short_circuits() -> None:
    """The guard must keep rejecting texts that genuinely lack the keyword."""
    assert not list(get_es_courts("Este documento no menciona ningun organo judicial."))
    assert not list(get_pt_courts("Este documento nao menciona nenhum orgao judicial."))
