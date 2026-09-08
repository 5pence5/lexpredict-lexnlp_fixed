"""Coverage tests for lexnlp.extract.en.entities.nltk_maxent."""

import pytest

from lexnlp.extract.en.entities.nltk_maxent import get_geopolitical, get_parties_as


def test_get_geopolitical_strict_drops_short_gpe():
    # With strict=True "US" is never joined to "France", and the
    # len(gpe) <= 2 cleanup branch skips it.
    result = list(get_geopolitical("US and France signed the treaty.", strict=True))
    assert result == ["France"]
    assert "US" not in result


def test_get_geopolitical_trailing_ampersand_stripped():
    # "&" joins "France" (CC branch), then "really" breaks the window,
    # leaving the "France &" form that the trailing-" &" cleanup strips.
    assert list(get_geopolitical("France & really went to Paris.")) == ["France", "Paris"]


def test_get_parties_as_skips_date_party_type():
    assert list(get_parties_as("Acme LLC as of March 1, 2020 shall pay.")) == []


def test_get_parties_as_no_match_yields_nothing():
    assert list(get_parties_as("There is nothing to see here.")) == []


def test_get_parties_as_company_match_raises_attribute_error():
    # Real behaviour, pinned: get_companies() yields plain tuples, but
    # get_parties_as dereferences ant.name/ant.company_type, so any
    # party_string containing a company raises AttributeError.
    # Maintainer note: this looks like a latent bug — the loop probably
    # meant to consume CompanyAnnotation objects.
    with pytest.raises(AttributeError):
        list(get_parties_as("Acme LLC as the Seller shall pay."))
