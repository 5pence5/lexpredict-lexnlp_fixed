__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.de.geoentities import get_geoentities
from lexnlp.extract.en.dict_entities import DictionaryEntry, DictionaryEntryAlias


def _georgien_entries() -> list[DictionaryEntry]:
    return [
        DictionaryEntry(
            id=1,
            name="Georgien",
            priority=800,
            name_is_alias=False,
            aliases=[DictionaryEntryAlias.entity_alias("Georgien", "de")],
            entity_name="Georgia",
            category="Countries",
            extra_columns={"iso_3166_2": "GE", "iso_3166_3": "GEO"},
        )
    ]


class TestDeGeoentitiesCoverage(TestCase):
    def test_get_geoentities_default_min_alias_len(self):
        text = "some odd text and Georgien mentioned inside it"
        results = list(get_geoentities(text, _georgien_entries()))
        self.assertEqual(1, len(results))
        hit = results[0]
        self.assertEqual(18, hit["location_start"])
        self.assertEqual(26, hit["location_end"])
        self.assertEqual("Georgien", hit["Alias"])
        self.assertEqual(1, hit["Entity ID"])

    def test_get_geoentities_no_match(self):
        results = list(get_geoentities("nothing geographic here", _georgien_entries()))
        self.assertEqual([], results)

    def test_get_geoentities_explicit_min_alias_len(self):
        text = "some odd text and Georgien mentioned inside it"
        results = list(get_geoentities(text, _georgien_entries(), min_alias_len=5))
        self.assertEqual(1, len(results))
        self.assertEqual("Georgien", results[0]["Alias"])
