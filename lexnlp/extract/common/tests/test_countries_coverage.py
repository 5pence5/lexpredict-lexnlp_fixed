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


class TestLookupNameFallback:
    def test_name_fallback_when_direct_index_misses(self, monkeypatch) -> None:
        # The installed pycountry lowercases its name index, so a direct
        # hit normally pre-empts the loop below. Simulate an index miss
        # (older case-sensitive releases) and require the case-insensitive
        # iteration fallback to still resolve the exact name.
        lookup_country.cache_clear()
        real_get = pycountry.countries.get

        def missing_name(**kwargs):
            if "name" in kwargs:
                return None
            return real_get(**kwargs)

        monkeypatch.setattr(pycountry.countries, "get", missing_name)
        info = lookup_country("Japan")
        assert info is not None
        assert info.alpha_2 == "JP"
        assert info.alpha_3 == "JPN"
        assert info.name == "Japan"

    def test_name_fallback_resolves_lowercase_query(self, monkeypatch) -> None:
        lookup_country.cache_clear()
        real_get = pycountry.countries.get

        def missing_name(**kwargs):
            if "name" in kwargs:
                return None
            return real_get(**kwargs)

        monkeypatch.setattr(pycountry.countries, "get", missing_name)
        info = lookup_country("brazil")
        assert info is not None
        assert info.alpha_2 == "BR"
        assert info.name == "Brazil"


class TestLookupToleratesLookupError:
    def test_lookup_error_in_direct_index_falls_back(self, monkeypatch) -> None:
        # Older pycountry releases raise KeyError (a LookupError) instead of
        # returning None; the direct-index loop must tolerate that and still
        # resolve via the iteration fallback.
        lookup_country.cache_clear()

        def raising_get(**kwargs):
            raise LookupError("simulated old-pycountry behaviour")

        monkeypatch.setattr(pycountry.countries, "get", raising_get)
        info = lookup_country("Germany")
        assert info is not None
        assert info.alpha_2 == "DE"
        assert info.alpha_3 == "DEU"
        assert info.name == "Germany"

    def test_lookup_error_with_unknown_country_returns_none(self, monkeypatch) -> None:
        lookup_country.cache_clear()

        def raising_get(**kwargs):
            raise LookupError("simulated old-pycountry behaviour")

        monkeypatch.setattr(pycountry.countries, "get", raising_get)
        assert lookup_country("Narnia") is None
