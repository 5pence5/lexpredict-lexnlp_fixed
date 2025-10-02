import datetime

from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.en import dates
from lexnlp.extract.en.date_validation import check_date_parts_are_in_date


def test_reject_impossible_formats_multiple_months():
    finder = dates.DateFinder(base_date=datetime.datetime(2020, 1, 1))
    locale = Locale('en-US')
    candidate = dates.CandidateDate(
        raw_text='Jan Feb',
        span=(0, 7),
        props={
            'months': ['Jan', 'Feb'],
            'digits_modifier': [],
            'digits': [],
            'days': [],
            'delimiters': [],
            'extra_tokens': []
        },
        index=0,
        date_finder=finder,
        locale=locale
    )

    reason = dates._reject_impossible_formats(candidate)
    assert reason == 'multiple_month_tokens'


def test_pipeline_accepts_valid_date():
    text = 'Dated as of June 1, 2017.'
    locale = Locale('en-US')
    outcomes = list(dates._candidate_outcomes(text, False, datetime.datetime(2017, 1, 1), locale))

    accepted = [outcome for outcome in outcomes if outcome.accepted]
    assert accepted
    assert accepted[0].date == datetime.date(2017, 6, 1)


def test_pipeline_rejects_decimal_without_month():
    text = 'Section on 6.25'
    locale = Locale('en-US')
    outcomes = list(dates._candidate_outcomes(text, False, datetime.datetime(2017, 1, 1), locale))

    rejections = [outcome for outcome in outcomes if not outcome.accepted]
    assert rejections
    assert rejections[0].reason == 'decimal_without_month'


def test_check_date_parts_are_in_date_month_validation():
    base_date = datetime.datetime(2017, 6, 1)
    props = {
        'digits': ['1', '2017'],
        'digits_modifier': [],
        'months': ['June'],
        'days': [],
        'delimiters': []
    }
    assert check_date_parts_are_in_date(base_date, props, month_lookup=dates.MONTH_BY_NAME)

    mismatched_props = {
        'digits': ['1', '2017'],
        'digits_modifier': [],
        'months': ['July'],
        'days': [],
        'delimiters': []
    }
    assert not check_date_parts_are_in_date(base_date, mismatched_props, month_lookup=dates.MONTH_BY_NAME)
