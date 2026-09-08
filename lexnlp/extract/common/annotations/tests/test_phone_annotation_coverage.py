"""Coverage tests for PhoneAnnotation.get_dictionary_values."""

from __future__ import annotations

from lexnlp.extract.common.annotations.phone_annotation import PhoneAnnotation


class TestPhoneAnnotationDictionaryValues:
    def test_values_with_phone_and_text(self) -> None:
        ann = PhoneAnnotation(coords=(0, 12), text="555-123-4567", phone="555-123-4567")
        assert ann.get_dictionary_values() == {
            "tags": {
                "Extracted Entity Phone": "555-123-4567",
                "Extracted Entity Text": "555-123-4567",
            }
        }

    def test_values_without_phone_uses_empty_string(self) -> None:
        ann = PhoneAnnotation(coords=(0, 12), text="555-123-4567")
        assert ann.phone is None
        assert ann.get_dictionary_values() == {
            "tags": {"Extracted Entity Phone": "", "Extracted Entity Text": "555-123-4567"}
        }

    def test_to_dictionary_merges_values(self) -> None:
        ann = PhoneAnnotation(coords=(0, 12), text="555-123-4567", phone="555-123-4567")
        result = ann.to_dictionary()
        assert result["attrs"] == {"start": 0, "end": 12}
        assert result["tags"]["Extracted Entity Type"] == "phone"
        assert result["tags"]["Extracted Entity Phone"] == "555-123-4567"
        assert result["tags"]["Extracted Entity Text"] == "555-123-4567"

    def test_cite_value_parts(self) -> None:
        ann = PhoneAnnotation(coords=(0, 12), text="555-123-4567", phone="555-123-4567")
        assert ann.get_cite_value_parts() == ["555-123-4567"]
