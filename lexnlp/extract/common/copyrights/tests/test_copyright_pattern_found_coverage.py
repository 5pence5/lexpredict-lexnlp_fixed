__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.common.copyrights.copyright_pattern_found import CopyrightPatternFound
from lexnlp.extract.common.pattern_found import PatternFound


def _base(name="Copyright 2019 Acme", start=0, end=19, prob=100) -> PatternFound:
    ptrn = PatternFound()
    ptrn.name = name
    ptrn.start = start
    ptrn.end = end
    ptrn.probability = prob
    return ptrn


class TestInit:
    def test_default_init_without_pattern(self) -> None:
        found = CopyrightPatternFound()
        assert found.name is None
        assert found.start == 0
        assert found.end == 0
        assert found.probability == 0
        assert found.company == ""
        assert found.start_year == 0
        assert found.end_year == 0

    def test_init_copies_fields_from_pattern(self) -> None:
        found = CopyrightPatternFound(_base())
        assert found.name == "Copyright 2019 Acme"
        assert found.start == 0
        assert found.end == 19
        assert found.probability == 100
        assert found.company == ""
        assert found.start_year == 0
        assert found.end_year == 0


class TestReprAndLength:
    def test_repr_format(self) -> None:
        found = CopyrightPatternFound(_base())
        assert repr(found) == '[0: 19]: "Copyright 2019 Acme", 100%'

    def test_get_length(self) -> None:
        found = CopyrightPatternFound(_base(start=4, end=19))
        assert found.get_length() == 15


class TestDetalizationLevel:
    def test_empty_text_part_scores_zero(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 0
        assert found.get_detalization_level("") == 0

    def test_uppercase_text_adds_level(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 5
        assert found.get_detalization_level("ACME!") == 1

    def test_company_adds_level(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 5
        found.company = "Acme"
        assert found.get_detalization_level("acme!") == 1

    def test_start_year_adds_level(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 5
        found.start_year = 1996
        assert found.get_detalization_level("acme!") == 1

    def test_end_year_adds_level(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 5
        found.end_year = 2019
        assert found.get_detalization_level("acme!") == 1

    def test_all_signals_stack_to_four(self) -> None:
        found = CopyrightPatternFound()
        found.start = 0
        found.end = 4
        found.company = "Acme"
        found.start_year = 1996
        found.end_year = 2019
        assert found.get_detalization_level("ACME") == 4


class TestPatternWorseThanTarget:
    def test_no_span_overlap_returns_false(self) -> None:
        found = CopyrightPatternFound(_base(start=0, end=5))
        other = CopyrightPatternFound(_base(start=10, end=15))
        assert found.pattern_worse_than_target(other, "x" * 20) is False

    def test_lower_level_is_worse(self) -> None:
        text = "ACME Copyright 1996-2019"
        found = CopyrightPatternFound(_base(start=0, end=22))
        other = CopyrightPatternFound(_base(start=5, end=22))
        other.company = "Acme"
        other.start_year = 1996
        other.end_year = 2019
        assert found.pattern_worse_than_target(other, text) is True

    def test_higher_level_is_not_worse(self) -> None:
        text = "ACME Copyright 1996-2019"
        found = CopyrightPatternFound(_base(start=0, end=22))
        found.company = "Acme"
        found.start_year = 1996
        found.end_year = 2019
        other = CopyrightPatternFound(_base(start=5, end=22))
        assert found.pattern_worse_than_target(other, text) is False

    def test_equal_level_shorter_target_is_worse(self) -> None:
        found = CopyrightPatternFound(_base(start=0, end=22))
        other = CopyrightPatternFound(_base(start=2, end=10))
        assert found.pattern_worse_than_target(other, "x" * 30) is True

    def test_equal_level_longer_target_is_not_worse(self) -> None:
        found = CopyrightPatternFound(_base(start=2, end=10))
        other = CopyrightPatternFound(_base(start=0, end=22))
        assert found.pattern_worse_than_target(other, "x" * 30) is False

    def test_span_detected_via_end_inside(self) -> None:
        # Only p.end falls inside self's span; still counts as overlap.
        found = CopyrightPatternFound(_base(start=10, end=30))
        other = CopyrightPatternFound(_base(start=0, end=12))
        assert found.pattern_worse_than_target(other, "x" * 40) is True
