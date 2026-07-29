#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from pathlib import Path

import pytest

from lexnlp import get_module_path
from lexnlp.nlp.en.segments import titles
from lexnlp.nlp.en.segments.titles import get_titles
from unittest import TestCase


TEST_DATA_PATH = Path(get_module_path()).parent / "test_data"


class TestTitles(TestCase):

    def test_title_1(self):
        """
        Test first example title.
        """
        file_text = (TEST_DATA_PATH / "1205332_2008-05-08_3").read_text(encoding="utf-8")
        self.assertEqual(['LEASE AGREEMENT'], list(get_titles(file_text)))

    def test_title_2(self):
        """
        Test second example title.
        """
        file_text = (TEST_DATA_PATH / "1100644_2016-11-21").read_text(encoding="utf-8")
        self.assertEqual(['VALIDIAN SOFTWARE LICENSE AGREEMENT'], list(get_titles(file_text)))

    def test_title_3(self):
        """
        test failure
        """
        text = """
        (1279209, 'en', 'C-106 TRANSPORTATION AND PUBLIC WORKS PERFORMANCE
        MEASURES Actual Forecast FY14 FY15 FY16 FY17 FY18 Traffic Engineering
         # of Miles of Roadway Striping 20 miles 10 miles 70 miles 10 miles
         10 Miles # of Signs Replaced 1,300 1,800 2,100 1,600 1,600 # of
         Traffic Signal Upgrades 2 15 18 25 30 Engineering Average Plan Review
          Time 6.9 days 6.13 days 8.34 days 7 days 7 days % Plan Reviews
          Completed Within 14 / 7 days 94% / 59% 94% / 67% 97% / 74% 100% /
           75% 100%/ 75% # Roadway Miles Receiving Major Roadway Maintenance
           57.8 miles 47 miles 45.0 miles 45.0 miles 45 miles Streets &
           Drainage Average Response Time for Street immediate Work Requests
           1 day 1 day 1 day 1 day 1day Percent of Street immediate work
           requests completed in 3 days 95% 90% 96% 95% 95% Percentage of
           staff hours utilized on recurring work activities 30% 33% 40% 45%
           45% Stormwater Utility Bill Collection Rate 94% 98% 95% 95% 95%
           Average Response Time for...', 1, , ...)"""
        self.assertEqual(0, len(list(get_titles(text))))


def test_title_training_download_has_timeout(monkeypatch):
    class Response:
        text = "training document"

        def raise_for_status(self):
            calls.append("raise_for_status")

    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(titles.requests, "get", fake_get)

    assert titles._download_training_document("https://example.test/document") == (
        "training document"
    )
    assert calls == [
        (
            "https://example.test/document",
            {"timeout": titles.TITLE_TRAINING_REQUEST_TIMEOUT},
        ),
        "raise_for_status",
    ]


def test_title_training_download_rejects_http_error(monkeypatch):
    class Response:
        text = "<html>not found</html>"

        def raise_for_status(self):
            raise titles.requests.HTTPError("404 Client Error")

    monkeypatch.setattr(
        titles.requests,
        "get",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(titles.requests.HTTPError, match="404"):
        titles._download_training_document("https://example.test/missing")
