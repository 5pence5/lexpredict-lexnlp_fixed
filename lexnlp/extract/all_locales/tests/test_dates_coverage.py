"""Coverage tests for the English branch of the all-locales date dispatcher."""

from __future__ import annotations

import datetime

from lexnlp.extract.all_locales.dates import get_date_annotations


class TestEnglishDateDispatch:
    def test_english_branch_yields_annotation(self) -> None:
        results = list(get_date_annotations("en", "The meeting is on June 1, 2017."))
        assert len(results) == 1
        assert results[0].date == datetime.date(2017, 6, 1)
        assert results[0].text == "on June 1, 2017"
        assert results[0].coords == (15, 30)

    def test_english_branch_forwards_options(self) -> None:
        results = list(
            get_date_annotations(
                "en-US",
                "The meeting is on June 1, 2017.",
                strict=False,
                base_date=datetime.datetime(2017, 6, 15),
                threshold=0.50,
            )
        )
        assert len(results) == 1
        assert results[0].date == datetime.date(2017, 6, 1)
