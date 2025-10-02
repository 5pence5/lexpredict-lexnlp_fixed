"""Validation helpers for English date extraction."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


def _ordinal_to_cardinal(value: str) -> Optional[int]:
    digits = ''.join(char for char in value if char.isdigit())
    return int(digits) if digits else None


def _collect_candidate_values(date_props: Dict[str, Any],
                              month_lookup: Mapping[str, int]) -> Dict[str, Iterable[int]]:
    digits: Sequence[str] = date_props.get('digits', [])
    months: Sequence[str] = date_props.get('months', [])
    modifiers: Sequence[str] = date_props.get('digits_modifier', [])

    digit_values = [int(d) for d in digits if d.isdigit()]
    month_values = [month_lookup.get(month.lower()) for month in months if month_lookup.get(month.lower())]
    modifier_values = [value for value in (_ordinal_to_cardinal(mod) for mod in modifiers) if value]

    return {
        'digits': digit_values,
        'months': month_values,
        'modifiers': modifier_values,
    }


def check_date_parts_are_in_date(
        date: datetime.datetime,
        date_props: Dict[str, Any],
        month_lookup: Optional[Mapping[str, int]] = None) -> bool:
    """Ensure parsed ``date`` still references the tokens found in ``date_props``."""
    month_lookup = month_lookup or {}

    units_of_time = ('year', 'month', 'day', 'hour', 'minute')
    date_values = {unit: getattr(date, unit, None) for unit in units_of_time if hasattr(date, unit)}

    candidate_values = _collect_candidate_values(date_props, month_lookup)

    if candidate_values['months']:
        month = date_values.get('month')
        if month and month not in candidate_values['months']:
            return False

    combined = [
        *candidate_values['digits'],
        *candidate_values['months'],
        *candidate_values['modifiers']
    ]
    difference = set(combined).difference(value for value in date_values.values() if value is not None)

    removeable = []
    reassembled_date: Dict[str, int] = {}

    for unit, value in date_values.items():
        if value is None:
            continue
        if unit == 'year':
            short_year = (value - 100 * (value // 100)) if value > 1000 else value
            if short_year in combined:
                reassembled_date[unit] = value
                removeable.append(short_year)
                continue
        if value in combined:
            reassembled_date[unit] = value
            removeable.append(value)

    diff_digits = [digit for digit in difference if digit not in removeable]
    diff_units = set(date_values.keys()) - set(reassembled_date.keys())

    if any(unit for unit in diff_units if unit in units_of_time[:3]):
        if diff_digits:
            return False
    return True


__all__ = ['check_date_parts_are_in_date']
