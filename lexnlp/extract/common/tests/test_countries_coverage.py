"""Coverage tests for :mod:`lexnlp.extract.common.countries` fallback paths."""

from __future__ import annotations

import pycountry

from lexnlp.extract.common.countries import lookup_country


class TestLookupOfficialName:
    def test_official_name_exact_case(self) -> None:
        # "United States of America" is the official_name, not the name
        # ("United States"), so the direct name index misses and the
        # official_name fallback loop must resolve it.
        assert pycountry.countries.get(name="United States of America") is None
        info = lookup_country("United States of America")
        assert info is not None
        assert info.alpha_2 == "US"
        assert info.alpha_3 == "USA"
        assert info.name == "United States"
        assert info.official_name == "United States of America"

    def test_official_name_lowercase(self) -> None:
        info = lookup_country("united states of america")
        assert info is not None
        assert info.alpha_2 == "US"

    def test_germany_official_name(self) -> None:
        assert pycountry.countries.get(name="Federal Republic of Germany") is None
        info = lookup_country("Federal Republic of Germany")
        assert info is not None
        assert info.alpha_2 == "DE"
        assert info.name == "Germany"

    def test_germany_official_name_lowercase(self) -> None:
        info = lookup_country("federal republic of germany")
        assert info is not None
        assert info.alpha_2 == "DE"
