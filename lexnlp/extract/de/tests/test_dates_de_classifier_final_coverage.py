"""Final-coverage tests for lexnlp.extract.de.dates_de_classifier missing lines."""

from __future__ import annotations

import pytest
from num2words import num2words

from lexnlp.extract.de import dates_de_classifier
from lexnlp.extract.de.dates_de_classifier import WRITTEN_DATE_NUMS, get_written_date_num


def test_below_twenty_beyond_table_uses_num2words_ten_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Line 582: with a short lookup table, day numbers below 20 fall back to num2words + "ten"."""
    full_table = list(WRITTEN_DATE_NUMS)
    monkeypatch.setattr(dates_de_classifier, "WRITTEN_DATE_NUMS", full_table[:10])
    for num in (11, 15, 19):
        assert get_written_date_num(num) == num2words(num, lang="de") + "ten"
        assert get_written_date_num(num) == full_table[num - 1]
    assert get_written_date_num(20) == num2words(20, lang="de") + "sten"
