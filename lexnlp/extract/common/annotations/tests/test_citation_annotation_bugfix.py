"""Regression tests for CitationAnnotation.get_dictionary_values court/source fix.

Prior to this PR the conditional checks for ``Extracted Entity Court`` and
``Extracted Entity Source`` incorrectly tested ``self.reporter`` instead of
``self.court`` and ``self.source`` respectively.  This file pins the correct
behaviour so future refactors cannot silently re-introduce the regression.

Bug (before fix):
    if self.reporter:   # WRONG — should be self.court
        df.tags["Extracted Entity Court"] = str(self.court)
    if self.reporter:   # WRONG — should be self.source
        df.tags["Extracted Entity Source"] = str(self.source)

Fix (after PR):
    if self.court:
        df.tags["Extracted Entity Court"] = str(self.court)
    if self.source:
        df.tags["Extracted Entity Source"] = str(self.source)
"""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.common.annotations.citation_annotation import CitationAnnotation


class TestCitationAnnotationCourtSourceBugfix:
    """Regression suite for the court/source conditional fix."""

    # ------------------------------------------------------------------
    # Core regression: court present, reporter absent
    # ------------------------------------------------------------------

    def test_court_tag_included_when_court_set_and_reporter_none(self) -> None:
        """Extracted Entity Court must appear even when reporter is None.

        Before the fix, the guard was ``if self.reporter``, which meant a
        citation with a court but no reporter would silently drop the court
        tag from the dictionary.
        """
        ann = CitationAnnotation(
            coords=(0, 20),
            court="SCOTUS",
            reporter=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Court" in d.tags, (
            "Extracted Entity Court should be present when court='SCOTUS' and reporter=None"
        )
        assert d.tags["Extracted Entity Court"] == "SCOTUS"

    def test_source_tag_included_when_source_set_and_reporter_none(self) -> None:
        """Extracted Entity Source must appear even when reporter is None."""
        ann = CitationAnnotation(
            coords=(0, 30),
            source="Roe v. Wade, 410 U.S. 113 (1973)",
            reporter=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Source" in d.tags, (
            "Extracted Entity Source should be present when source is set and reporter=None"
        )
        assert d.tags["Extracted Entity Source"] == "Roe v. Wade, 410 U.S. 113 (1973)"

    # ------------------------------------------------------------------
    # Both fields absent → no spurious tags
    # ------------------------------------------------------------------

    def test_court_tag_absent_when_court_is_none(self) -> None:
        """When court is None the tag must not appear — even when reporter is set."""
        ann = CitationAnnotation(
            coords=(0, 20),
            reporter="U.S.",
            court=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Court" not in d.tags

    def test_source_tag_absent_when_source_is_none(self) -> None:
        """When source is None the tag must not appear — even when reporter is set."""
        ann = CitationAnnotation(
            coords=(0, 20),
            reporter="U.S.",
            source=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Source" not in d.tags

    # ------------------------------------------------------------------
    # Both fields present → both tags included
    # ------------------------------------------------------------------

    def test_both_tags_present_when_both_fields_set(self) -> None:
        """When court and source are both set, both tags appear."""
        ann = CitationAnnotation(
            coords=(0, 50),
            court="9th Cir.",
            source="United States v. Foo, 999 F.3d 1 (9th Cir. 2021)",
            reporter=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Court" in d.tags
        assert "Extracted Entity Source" in d.tags
        assert d.tags["Extracted Entity Court"] == "9th Cir."
        assert d.tags["Extracted Entity Source"] == "United States v. Foo, 999 F.3d 1 (9th Cir. 2021)"

    # ------------------------------------------------------------------
    # Reporter still controls its own tag
    # ------------------------------------------------------------------

    def test_reporter_tag_included_when_reporter_set(self) -> None:
        """Reporter field is independent — its tag should appear when set."""
        ann = CitationAnnotation(
            coords=(0, 15),
            reporter="F.3d",
            court=None,
            source=None,
        )
        d = ann.get_dictionary_values()
        assert "Extracted Entity Reporter" in d.tags
        assert d.tags["Extracted Entity Reporter"] == "F.3d"
        # Neither court nor source should appear
        assert "Extracted Entity Court" not in d.tags
        assert "Extracted Entity Source" not in d.tags

    def test_reporter_absent_does_not_suppress_court_and_source(self) -> None:
        """The fix: all three fields are now independent of each other."""
        ann = CitationAnnotation(
            coords=(0, 60),
            reporter=None,
            court="D.C. Cir.",
            source="Smith v. Jones, 1 D.C. 1 (D.C. Cir. 2000)",
        )
        d = ann.get_dictionary_values()
        # reporter tag absent
        assert "Extracted Entity Reporter" not in d.tags
        # court and source tags present
        assert d.tags["Extracted Entity Court"] == "D.C. Cir."
        assert d.tags["Extracted Entity Source"] == "Smith v. Jones, 1 D.C. 1 (D.C. Cir. 2000)"

    # ------------------------------------------------------------------
    # Full construction — all fields set — everything appears
    # ------------------------------------------------------------------

    def test_all_tags_present_when_all_fields_set(self) -> None:
        ann = CitationAnnotation(
            coords=(5, 80),
            text="410 U.S. 113",
            volume=410,
            reporter="U.S.",
            reporter_full_name="United States Reports",
            page=113,
            year=1973,
            court="SCOTUS",
            source="Roe v. Wade",
        )
        d = ann.get_dictionary_values()
        assert d.tags["Extracted Entity Reporter"] == "U.S."
        assert d.tags["Extracted Entity Reporter Full Name"] == "United States Reports"
        assert d.tags["Extracted Entity Court"] == "SCOTUS"
        assert d.tags["Extracted Entity Source"] == "Roe v. Wade"

    # ------------------------------------------------------------------
    # Minimal construction — no optional fields → only text tag
    # ------------------------------------------------------------------

    def test_minimal_construction_produces_only_text_tag(self) -> None:
        ann = CitationAnnotation(coords=(0, 0), text="bare")
        d = ann.get_dictionary_values()
        assert "Extracted Entity Text" in d.tags
        # Optional tags must be absent
        for key in (
            "Extracted Entity Volume",
            "Extracted Entity Year",
            "Extracted Entity Page",
            "Extracted Entity Reporter",
            "Extracted Entity Court",
            "Extracted Entity Source",
        ):
            assert key not in d.tags, f"Unexpected tag '{key}' in minimal annotation"
