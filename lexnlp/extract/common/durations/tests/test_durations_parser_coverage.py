__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from decimal import Decimal
from unittest import TestCase

from lexnlp.extract.common.annotations.duration_annotation import DurationAnnotation
from lexnlp.extract.common.durations.durations_parser import DurationParser


def _make_ant(
    start: int,
    end: int,
    amount: int,
    days: int,
    duration_type: str,
) -> DurationAnnotation:
    return DurationAnnotation(
        coords=(start, end),
        locale="en",
        text=f"{amount} {duration_type}",
        amount=Decimal(amount),
        duration_days=Decimal(days),
        duration_type=duration_type,
        duration_type_en=duration_type,
    )


class TestDurationParserCoverage(TestCase):
    def test_sum_annotations_accumulates_same_type(self):
        first = _make_ant(0, 8, 5, 1825, "year")
        second = _make_ant(10, 18, 3, 1095, "year")
        summed = DurationParser.sum_annotations([first, second])
        self.assertEqual(summed.coords, (0, 18))
        self.assertTrue(summed.is_complex)
        self.assertEqual(summed.duration_days, Decimal(2920))
        self.assertEqual(summed.amount, Decimal(2920))
        self.assertEqual(summed.duration_type, "year")
        self.assertEqual(summed.duration_type_en, "year")
        self.assertEqual(summed.locale, "en")
        self.assertEqual(summed.value_dict, {"year": 8.0})

    def test_sum_annotations_distinct_types(self):
        first = _make_ant(0, 8, 1, 365, "year")
        second = _make_ant(10, 19, 2, 60, "month")
        summed = DurationParser.sum_annotations([first, second])
        self.assertEqual(summed.duration_days, Decimal(425))
        self.assertEqual(summed.value_dict, {"year": 1.0, "month": 2.0})
        self.assertEqual(summed.duration_type, "month")

    def test_get_all_annotations_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            DurationParser.get_all_annotations("5 years")
