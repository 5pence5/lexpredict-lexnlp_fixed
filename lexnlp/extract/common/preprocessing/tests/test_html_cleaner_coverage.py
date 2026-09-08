"""Coverage tests for html_cleaner.clean_html empty-input guard."""

from __future__ import annotations

from lexnlp.extract.common.preprocessing.html_cleaner import clean_html


class TestCleanHtmlCoverage:
    def test_empty_string_returns_empty(self) -> None:
        assert clean_html("") == ""

    def test_drops_noise_but_keeps_prose(self) -> None:
        out = clean_html("<p>hi<script>x()</script></p>")
        assert "<script" not in out.lower()
        assert "<p>hi</p>" in out
