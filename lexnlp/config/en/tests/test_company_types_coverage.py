"""Coverage tests for CompanyDescriptor string forms."""

from __future__ import annotations

from lexnlp.config.en.company_types import COMPANY_TYPES, CompanyDescriptor, get_company_types


class TestCompanyDescriptorStr:
    def test_str_format(self) -> None:
        desc = CompanyDescriptor(alias="inc", abbreviation="Inc.", label="Corporation")
        assert str(desc) == "inc: Inc. (Corporation)"

    def test_repr_matches_str(self) -> None:
        desc = CompanyDescriptor(alias="ltd", abbreviation="Ltd.", label="Limited")
        assert repr(desc) == "ltd: Ltd. (Limited)"
        assert repr(desc) == str(desc)

    def test_loaded_types_str_roundtrip(self) -> None:
        types = get_company_types()
        assert len(types) > 0
        desc = types["inc"]
        assert str(desc) == f"{desc.alias}: {desc.abbreviation} ({desc.label})"
        assert repr(desc) == str(desc)
        assert COMPANY_TYPES["inc"].abbreviation == desc.abbreviation
