"""Tests for SsnAnnotation.get_cite_value_parts None-safety fix.

Before this PR, ``get_cite_value_parts`` returned ``[self.number]``, which
could smuggle a ``None`` value into a ``list[str]`` and cause downstream
``str.join`` / concatenation failures.

After the fix it returns ``[self.number or ""]``, coercing a missing number
to an empty string.
"""

from __future__ import annotations

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.common.annotations.ssn_annotation import SsnAnnotation


class TestSsnAnnotationGetCiteValueParts:
    """Pin the None-safe behaviour of get_cite_value_parts."""

    def test_none_number_returns_list_with_empty_string(self) -> None:
        """When number is None, get_cite_value_parts must return [''] not [None].

        Returning [None] would violate the list[str] contract and cause
        downstream str.join failures.
        """
        ann = SsnAnnotation(coords=(0, 10), number=None)
        parts = ann.get_cite_value_parts()
        assert parts == [""], f"Expected [''] for None number, got {parts!r}"
        # Every element must be str — not None
        for part in parts:
            assert isinstance(part, str), f"Non-str element {part!r} in cite value parts"

    def test_populated_number_returned_as_is(self) -> None:
        """When number has a value it must appear in the result unchanged."""
        ann = SsnAnnotation(coords=(0, 11), number="123-45-6789")
        parts = ann.get_cite_value_parts()
        assert parts == ["123-45-6789"]

    def test_empty_string_number_returned_unchanged(self) -> None:
        """An explicitly set empty string is already a valid str and passes through."""
        ann = SsnAnnotation(coords=(0, 5), number="")
        parts = ann.get_cite_value_parts()
        assert parts == [""]

    def test_returns_list_of_length_one(self) -> None:
        """Contract: exactly one element regardless of value."""
        for number in (None, "", "000-00-0000", "987-65-4321"):
            ann = SsnAnnotation(coords=(0, 0), number=number)
            parts = ann.get_cite_value_parts()
            assert len(parts) == 1, f"Expected 1 element for number={number!r}, got {len(parts)}"

    def test_get_dictionary_values_handles_none_number(self) -> None:
        """get_dictionary_values also uses 'or ""' — verify the tag is not None."""
        ann = SsnAnnotation(coords=(0, 10), number=None)
        d = ann.get_dictionary_values()
        tags = d["tags"]
        assert tags["Extracted Entity SSN"] == "", "SSN tag must be empty string when number is None"

    def test_get_dictionary_values_populated(self) -> None:
        ann = SsnAnnotation(coords=(0, 11), text="SSN: 123-45-6789", number="123-45-6789")
        d = ann.get_dictionary_values()
        tags = d["tags"]
        assert tags["Extracted Entity SSN"] == "123-45-6789"
        assert tags["Extracted Entity Text"] == "SSN: 123-45-6789"
