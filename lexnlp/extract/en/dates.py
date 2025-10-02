"""Date extraction for English.

This module implements date extraction functionality in English.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


# pylint: disable=bare-except

# Standard imports
import calendar
import datetime
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

# Third-party packages
import regex as re

# LexNLP imports
from lexnlp.extract.all_locales.languages import Locale
from lexnlp.extract.common.annotations.date_annotation import DateAnnotation
from lexnlp.extract.common.date_parsing.datefinder import DateFinder
from lexnlp.extract.common.dates import DateParser
from lexnlp.extract.common.dates_classifier_model import build_date_model, get_date_features
from lexnlp.extract.en.date_model import MODEL_DATE, MODULE_PATH, DATE_MODEL_CHARS
from lexnlp.extract.en.date_validation import check_date_parts_are_in_date


# Distance in characters to use to merge two date strings


DATE_MERGE_WINDOW = 10

# Maximum date length
DATE_MAX_LENGTH = 40

# Setup regular expression for "as of" strings
AS_OF_PATTERN = r"""
(made|dated|date)
[\s]+?
as
[\s]+?
of[\s]+?
(.{{0,{max_length}}})
""".format(max_length=DATE_MAX_LENGTH)

RE_AS_OF = re.compile(AS_OF_PATTERN, re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE)

# 'january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec'
EN_MONTHS = [['january', 'jan'], ['february', 'feb', 'febr'], ['march', 'mar'],
             ['april', 'apr'], ['may'], ['june', 'jun'],
             ['july', 'jul'], ['august', 'aug'], ['september', 'sep', 'sept'],
             ['october', 'oct'], ['november', 'nov'], ['december', 'dec']]


def get_month_by_name():
    return {month_name: ix + 1 for ix, month_names in enumerate(EN_MONTHS) for month_name in month_names}


MONTH_BY_NAME = get_month_by_name()

MONTH_FULLS = {v.lower(): k for k, v in enumerate(calendar.month_name)}


@dataclass
class CandidateDate:
    """Container for a potential date extracted by :class:`DateFinder`."""

    raw_text: str
    span: Tuple[int, int]
    props: Dict[str, Any]
    index: int
    date_finder: DateFinder
    locale: Locale
    normalized_text: str = field(init=False)

    def __post_init__(self) -> None:
        self.normalized_text = self.raw_text


@dataclass
class CandidateOutcome:
    """Result of evaluating a :class:`CandidateDate`."""

    candidate: CandidateDate
    accepted: bool
    date: Optional[datetime.date] = None
    reason: Optional[str] = None


def get_raw_date_list(text, strict=False, base_date=None, return_source=False, locale=None) -> List:
    return list(get_raw_dates(
        text, strict=strict, base_date=base_date, return_source=return_source, locale=locale))


def _copy_date_props(date_props: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in date_props.items():
        if isinstance(value, list):
            copied[key] = value.copy()
        elif isinstance(value, tuple):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _normalize_day_of(candidate: CandidateDate,
                      previous_candidate: Optional[CandidateDate],
                      previous_outcome: Optional[CandidateOutcome]) -> CandidateDate:
    props = candidate.props
    if not props:
        return candidate

    extra_tokens: List[str] = props.get("extra_tokens", [])
    if not extra_tokens:
        return candidate

    has_of_token = any(token.lower() == "of" for token in extra_tokens)
    if not has_of_token or previous_candidate is None or previous_outcome is None:
        return candidate

    if previous_outcome.accepted:
        return candidate

    previous_modifiers: List[str] = previous_candidate.props.get("digits_modifier", [])
    if len(previous_modifiers) != 1:
        return candidate

    modifier_token = previous_modifiers[-1]
    normalized_modifier = (modifier_token
                           .replace("st", "")
                           .replace("nd", "")
                           .replace("rd", "")
                           .replace("th", ""))
    props["digits_modifier"] = previous_modifiers + props.get("digits_modifier", [])
    previous_candidate.props["digits_modifier"] = previous_modifiers[:-1]
    candidate.normalized_text = f"{normalized_modifier}{candidate.normalized_text}"
    return candidate


def _reject_impossible_formats(candidate: CandidateDate) -> Optional[str]:
    props = candidate.props
    text = candidate.normalized_text

    num_month = len(props.get("months", []))
    num_digits_modifier = len(props.get("digits_modifier", []))
    num_digits = len(props.get("digits", []))
    num_days = len(props.get("days", []))
    delimiters: Sequence[str] = props.get("delimiters", [])
    num_slash = delimiters.count("/")
    num_point = delimiters.count(".")
    num_hyphen = delimiters.count("-")

    if num_month > 1:
        return "multiple_month_tokens"

    extra_tokens: List[str] = props.get("extra_tokens", [])
    if (num_month == 1 and extra_tokens
            and (props["months"][0] + extra_tokens[-1]) in text):
        return "month_followed_by_word"

    if num_digits_modifier > 0 and num_digits == 0:
        return "modifier_without_digits"

    if num_days > 0 and num_digits == 0:
        return "day_without_digits"

    if num_month == 0 and num_digits_modifier == 0 and num_digits <= 1:
        return "insufficient_components"

    if re.match(r"\d{1,2}\s+\d{1,2}", text):
        return "ambiguous_double_numbers"

    if num_point and not num_month and not re.match(r"\d{2}\.\d{2}\.\d{2,4}", text):
        return "decimal_without_month"

    if re.search(r"\d{2,4}\.\s*[A-Za-z]", text):
        return "number_dot_word"

    if (num_slash == 1 or num_hyphen == 1) and num_digits > 2:
        return "fraction_like_number"

    found_triple = any(len(digit) == 3 for digit in props.get("digits", []))
    found_double_zero = any(digit.startswith("00") for digit in props.get("digits", []))
    if found_triple or found_double_zero:
        return "invalid_digit_length"

    month_string = "".join(props.get("months", [])).lower()
    if num_digits == 0 and num_days == 0 and month_string == "may":
        return "standalone_may"

    if (num_digits > 0 and (num_point + num_slash + num_hyphen) > 0 and month_string == "may"):
        return "punctuated_may"

    return None


def _normalize_tokens(candidate: CandidateDate) -> CandidateDate:
    props = candidate.props
    text = candidate.normalized_text
    extra_tokens = sorted(props.get("extra_tokens", []), key=len, reverse=True)
    for token in extra_tokens:
        if token.lower() in ["to", "t"]:
            continue
        text = text.replace(token, "")
    candidate.normalized_text = text.strip()
    props["extra_tokens"] = []
    return candidate


def _reject_post_normalization(candidate: CandidateDate) -> Optional[str]:
    text = candidate.normalized_text
    if len(text) > DATE_MAX_LENGTH:
        return "exceeds_max_length"

    match_delims = set("".join(candidate.props.get("delimiters", [])))
    bad_delims = {",", " ", "\n", "\t"}
    len_diff_set = len(match_delims - bad_delims)
    num_month = len(candidate.props.get("months", []))
    if len_diff_set == 0 and num_month == 0:
        return "digits_without_month"

    return None


def _parse_candidate(candidate: CandidateDate) -> CandidateOutcome:
    date_props = candidate.props
    date_finder = candidate.date_finder
    base_string = candidate.normalized_text

    try:
        date_string_tokens = base_string.split()
        date: Optional[datetime.datetime] = None
        for cutter in range(len(date_string_tokens)):
            for direction in (0, 1):
                working_string = base_string
                if cutter > 0:
                    if direction:
                        working_tokens = date_string_tokens[cutter:]
                    else:
                        working_tokens = date_string_tokens[:-cutter]
                    working_string = " ".join(working_tokens)
                try:
                    date = date_finder.parse_date_string(working_string, date_props, locale=candidate.locale)
                except Exception:  # pylint: disable=broad-except
                    date = None
                if date:
                    break
            if date:
                break
    except TypeError:
        return CandidateOutcome(candidate, False, reason="type_error")

    if not date:
        return CandidateOutcome(candidate, False, reason="parse_failure")

    if not check_date_parts_are_in_date(date, date_props, month_lookup=MONTH_BY_NAME):
        return CandidateOutcome(candidate, False, reason="date_parts_mismatch")

    if hasattr(date, "tzinfo"):
        try:
            _ = date.isoformat()
        except ValueError:
            return CandidateOutcome(candidate, False, reason="invalid_timezone")

    result_date: datetime.date
    if isinstance(date, datetime.datetime) and date.hour == 0 and date.minute == 0:
        result_date = date.date()
    else:
        result_date = date

    return CandidateOutcome(candidate, True, date=result_date)


def _iter_candidate_evaluations(text: str,
                                date_finder: DateFinder,
                                locale: Locale,
                                strict: bool) -> Iterable[CandidateOutcome]:
    possible_dates = list(date_finder.extract_date_strings(text, strict=strict))

    previous_candidate: Optional[CandidateDate] = None
    previous_outcome: Optional[CandidateOutcome] = None

    for index, possible_date in enumerate(possible_dates):
        candidate = CandidateDate(
            raw_text=possible_date[0],
            span=possible_date[1],
            props=_copy_date_props(possible_date[2]),
            index=index,
            date_finder=date_finder,
            locale=locale,
        )

        candidate = _normalize_day_of(candidate, previous_candidate, previous_outcome)

        rejection_reason = _reject_impossible_formats(candidate)
        if rejection_reason is not None:
            outcome = CandidateOutcome(candidate, False, reason=rejection_reason)
        else:
            candidate = _normalize_tokens(candidate)
            post_rejection = _reject_post_normalization(candidate)
            if post_rejection is not None:
                outcome = CandidateOutcome(candidate, False, reason=post_rejection)
            else:
                outcome = _parse_candidate(candidate)

        yield outcome
        previous_candidate = candidate
        previous_outcome = outcome


def _candidate_outcomes(text: str,
                        strict: bool,
                        base_date: Optional[datetime.datetime],
                        locale: Optional[Locale]) -> Iterable[CandidateOutcome]:
    if isinstance(locale, str):
        locale_obj = Locale(locale)
    elif locale is None:
        locale_obj = Locale('')
    else:
        locale_obj = locale

    if not base_date:
        base_date = datetime.datetime.now().replace(
            day=1, month=1, hour=0, minute=0, second=0, microsecond=0)

    date_finder = DateFinder(base_date=base_date)
    for extra_token in date_finder.EXTRA_TOKENS_PATTERN.split('|'):
        if extra_token != 't':
            date_finder.REPLACEMENTS[extra_token] = ' '

    return _iter_candidate_evaluations(text, date_finder, locale_obj, strict)


def get_raw_dates(text, strict=False, base_date=None,
                  return_source=False, locale=None) -> Generator:
    """
    Find "raw" or potential date matches prior to false positive classification.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param base_date: base date to use for implied or partial matches
    :param return_source: whether to return raw text around date
    :param locale: locale object
    :return:
    """
    for outcome in _candidate_outcomes(text, strict, base_date, locale):
        if not outcome.accepted or outcome.date is None:
            continue
        if return_source:
            yield outcome.date, outcome.candidate.span
        else:
            yield outcome.date
def get_dates_list(text, **kwargs) -> List:
    return list(get_dates(text, **kwargs))


def get_dates(text: str,
              strict=False,
              base_date=None,
              return_source=False,
              threshold=0.50,
              locale='') -> Generator:
    """
    Find dates after cleaning false positives.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param base_date: base date to use for implied or partial matches
    :param return_source: whether to return raw text around date
    :param threshold: probability threshold to use for false positive classifier
    :param locale: locale string
    :return:
    """
    # Get raw dates
    for ant in get_date_annotations(text, strict, locale, base_date, threshold):
        if return_source:
            yield ant.date, ant.coords
        else:
            yield ant.date


def get_date_annotations(text: str,
                         strict: Optional[bool] = None,
                         locale: Optional[str] = '',
                         base_date: Optional[datetime.datetime] = None,
                         threshold: float = 0.50) \
        -> Generator[DateAnnotation, None, None]:
    """
    Find dates after cleaning false positives.
    :param text: raw text to search
    :param strict: whether to return only complete or strict matches
    :param locale: locale string
    :param base_date: base date to use for implied or partial matches
    :param threshold: probability threshold to use for false positive classifier
    :return:
    """

    # Get raw dates
    strict = strict if strict is not None else False
    outcomes = _candidate_outcomes(text, strict, base_date, locale)

    for outcome in outcomes:
        if not outcome.accepted or outcome.date is None:
            continue

        start, end = outcome.candidate.span
        feature_row = get_date_features(text, start, end, characters=DATE_MODEL_CHARS)
        feature_list = len(feature_row) * [0.0]
        for i, col in enumerate(MODEL_DATE.columns):
            feature_list[i] = feature_row[col]
        date_score = MODEL_DATE.predict_proba([feature_list])
        if date_score[0, 1] >= threshold:
            annotation = DateAnnotation(
                coords=(start, end),
                text=text[slice(start, end)],
                date=outcome.date,
                score=date_score[0, 1]
            )
            yield annotation


def train_default_model(save=True):
    """
    Train default model.
    :return:
    """
    examples = [("""No later than 2017-06-01.""", [datetime.date(2017, 6, 1)]),
                ("""Dated as of June 1, 2017""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by June 2017""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by June""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by the 1st day of June, 2017""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by the 1st day of June 2017""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by the 1st of June, 2017""", [datetime.date(2017, 6, 1)]),
                ("""Will be completed by the 1st of June 2017""", [datetime.date(2017, 6, 1)]),
                ("""section on 6.25""", []),
                (
                    """All work shall be completed in accordance with WDD sketch dated 15 March 2005 and Hansen 
                    Mechanical COR.""",
                    [datetime.date(2005, 3, 15)]),
                (
                    """Cost Plus Incentive Construction Contract with JH Kelly LLC On August 8, 2007, Hoku Materials, 
                    Inc. entered into a construction agreement, the Construction Agreement, with JH Kelly LLC., or JH 
                    Kelly, for construction services for the construction of a polysilicon production plant with an 
                    annual capacity of 2,000 metric tons.""",
                    [datetime.date(2007, 8, 8)]),
                (
                    """SUBTOTAL 35,842,000 PROJECT REQUIREMENTS 1,281,000 CONSTRUCTION CONTINGENCY 1,075,000 TRAILER 
                    RELOCATION ALLOWANCE 50,000 SCHEDULE ADJUSTMENT 357,000 HOIST/ELEVATOR OPERATOR 104,000 GENERAL 
                    CONDITIONS 1,481,000 FEE 1,306,000 G. C. PAYMENT & PERFORMANCE BOND 204,000 - ----------------- 
                    ============================================================ ================== ================= 
                    TOTAL CONSTRUCTION COSTS 41,700,000 - ----------------- 
                    ============================================================ ================== 
                    =================""",
                    []),
                ("""4-7-98 Date Date""", [datetime.date(1998, 4, 7)]),
                ("""4/7/98 Date Date""", [datetime.date(1998, 4, 7)]),
                (
                    """This monthly maintenance and support arrangement will have an initial term of six (6) months. 
                    The arrangement will then automatically renew for an additional twelve (12) months at the above 
                    rates and conditions unless written notification to US/INTELICOM of Licensee's intent to cancel 
                    the arrangement is received no later than September 1, 1998. Unless Licensee elects to cancel this 
                    arrangement at the end of the first six months, the "initial term" of the arrangement will be 
                    through September 30, 1999.""",
                    [datetime.date(1998, 9, 1), datetime.date(1999, 9, 30)]),
                (
                    """CLASSIFICATION	  	STRAIGHT TIME TIME & A HALF    	 	DOUBLE TIME PIPEFITTERS LOCAL #26 LV 
                    ZONE 5,6,7""",
                    []),
                (
                    """Action	NBBB Milestone Dates - All AFC	Date required Owner	General Arrangement Frozen	[*] 
                    Jensen/Owner	Steel design Main Deck and Below	[*] Jensen/Owner	General Arrangement	[*] 
                    Jensen/Owner	Tonnage Openings	[*] Jensen/Owner	Electrical Single Line	[*] Owner	
                    Interior finish schedule	[*] Owner	Room layouts Cabins	[*] Owner	Food Service Space 
                    Layouts	[*] Owner	Public Space layouts	[*] Jensen/Owner	GW Piping Diagrams	[*] 
                    Jensen/Owner	BW Piping Diagrams	[*] Jensen/Owner	PW Piping Diagrams	[*] Jensen/Owner 
                    Fire Main Piping Diagrams	[*] Jensen/Owner	Steel design Modules 4 and 5	[*] 
                    Jensen/Owner	Steel design Modules 6,7,8	[*] Jensen/Owner	Fire Zones	[*] Jensen/Owner	
                    Heat load data (inc OFE)	[*] Owner	Owner Equip Information/Heat Load data	[*] 
                    Jensen/Owner	Main Wireway routing	[*] NBBBJensen	FGS Diagram	[*] NBBBJensen	PAGA 
                    Diagram	[*] NBBBJensen	Telephone Diagram	[*]""",
                    []),
                ("""9.6.6,9,8.2,9.9.3,9.10.1,9.103, 12.3""", []),
                (
                    """NOW, THEREFORE, for good and valuable consideration, the receipt and sufficiency of which are 
                    hereby acknowledged, the parties mutually agree as follows: 1. License Grant. NECTAR hereby grants 
                    to Siboney an exclusive license and right in the United States, its territories and possessions, 
                    to use, revise, modify and create derivative works of the "MathTrek 1,2,3", "MathTrek 4,5,6" and 
                    "Math Trek 7,8,9" software program series (the "Licensed Software") for use on Macintosh and 
                    Windows operating systems, and to repackage, manufacture, market, distribute, sell, lease, 
                    license and sub-license such revised and/or modified Licensed Software. Such revisions, 
                    modifications and derivative works are referred to herein as "Modified Software". The foregoing 
                    licenses to Siboney are subject to NECTAR's right set forth in Section 2 hereof, and to any 
                    licenses previously granted by NECTAR to end-users of the Licensed Software. Siboney shall have 
                    no rights in the Licensed Software or Modified Software other than as set forth in this Agreement. 
                    NECTAR shall, as reasonably requested by Siboney, consult with Siboney concerning such 
                    revisions, modifications and the like, and all revisions and the like will be subject to 
                    NECTAR's approval, which shall not be unreasonably withheld. NECTAR shall receive one (1) copy 
                    of every commercial product created pursuant to this license.""",
                    []),
                (
                    """NECTAR hereby grants to Siboney an exclusive license and right in the United States, its 
                    territories and possessions, to use, revise, modify and create derivative works of the 
                    "MathTrek 1,2,3", "MathTrek 4,5,6" and "Math Trek 7,8,9" software program series (the "Licensed 
                    Software") for use on Macintosh and Windows operating systems, and to repackage, manufacture, 
                    market, distribute, sell, lease, license and sub-license such revised and/or modified Licensed 
                    Software.""",
                    []),
                (
                    """27.1 Notice of Default 1-71 27.2 Contractor's Default 1-72 27.3 Valuation at Date of 
                    Termination 1-72 27.4 Payment After Termination 1-72 27.5 Effect on Liability for Delay 1-72 27.6 
                    Employer's Default 1-73 27.7 Removal of Contractor's Equipment 1-73 27.8 Payment on Termination 
                    for Employer's Default 1-73""",
                    []),
                (
                    """30.1 Customs, Import Duties and Taxes 1-74 30.2 Clearance Through Customs 1-75 30.3 Taxation 
                    1-75 30.4 Customs and Taxes on Contractor's Equipment 1-75""",
                    []),
                ("""3.18.1, 6.1.1, 7.3.6, 8,2.1, 9.3.2, 9.8.4, 9.9.1,""", []),
                ("""Total OCIP Credits 19,867,980""", []),
                ("""3.11,42.8,7, 8.3.1, 9.3.1.1, 11.4.9""", []),
                (
                    """Change Request Form: 122 Design Action Notification: DAN/ 0007 Increase to the Guaranteed 
                    Maximum Price: HK$ 250,000/ USD 32,052 12. Coat Check-in and Storage Location: First Floor, 
                    Meeting Rooms A separate coat check-in and storage area is added adjacent to the Main Meeting 
                    Rooms at the First Floor GL 20/ V-T. Change Request Form: 123 Design Action Notification: DAN/ 
                    0014 Increase to the Guaranteed Maximum Price: HK$ 50,000/ USD 6,410 13. Hotel Registration 
                    Counter Location: Hotel Main Registration Counter Provision of additional cooling fans to the 
                    Main Registration counter to meet cooling requirements of I.T Equipment. Change Request Form: 
                    125 Design Action Notification: DAN/ 0046 Increase to the Guaranteed Maximum Price: HK$ 76,000/ 
                    USD 9,744 14. VIP Manager’s Office / pantry Location: Ground Floor, VIP Area The private dining 
                    room to the VIP Casino is deleted and is replaced with a VIP Manager’s Office, dry pantry and 
                    connecting corridor. Change Request Form: 126 Design Action Notification: DAN/ 0019 Increase to 
                    the Guaranteed Maximum Price: HK$ 67,000/ USD 8,590""",
                    []),
                ("""5,397,150""", []),
                (
                    """Plaintiff O2 Micro International Limited and Defendants Monolithic Power Systems, Inc. (MPS), 
                    Michael Hsing, Advanced Semiconductor Manufacturing Company, Ltd. (ASMC), ASUSTeK Computer, 
                    Inc., and Compal Electronics, Inc. (collectively, Defendants) dispute the meaning of terms and 
                    phrases used in 02 Micro’s U.S. Patent No. 6,259,615 (the ‘615 patent), its U.S. Patent No. 
                    6,396,722 (the ‘722 patent) and its U.S. Patent No. 6,804,129 (the ‘129 patent) .1 02 Micro 
                    requests that the Court adopts the claim constructions previously adopted by this Court and by 
                    the Eastern District of Texas court. Defendants ask the Court to adopt their proposed construction 
                    of two disputed phrases. In addition, O2 Micro moves for summary judgment based on collateral 
                    estoppel. Defendants oppose the motion and cross-move for summary judgment. O2 Micro opposes 
                    their motion for summary judgment. The matters were heard on October 27, 2006. Having considered 
                    the parties’ papers, the evidence cited therein and oral argument, the Court construes the 
                    disputed terms and phrases as set forth below. In addition, the Court denies O2 Micro’s motion 
                    for summary judgment and grants in part Defendants’ motion for summary judgment and denies it in 
                    part. BACKGROUND I. Patents at issue The ‘615, ‘722 and ‘129 patents are all entitled: 
                    “High-Efficiency Adaptive DC/AC Converter.” They are related to the same technology: the 
                    ‘129 patent is a continuation of the ‘722""",
                    [datetime.date(2006, 10, 27)]),
                ("""6,396,722 (the ‘722 patent) and its U.S. Patent No.""", []),
                (
                    """Japanese Restaurant Noodles Restaurant Italian Restaurant Change Request Form: 130 Design 
                    Action Notification: DAN/ 008 Increase to the Guaranteed Maximum Price: HK$ 75,075/ USD 9,625 17. 
                    Refrigerated wine cabinets Location: First Floor, Italian Restaurant Refrigerated wine coolers 
                    (4nr) are included at the Italian Restaurant. These are to be housed in millwork at the back of 
                    the drinks bar.""",
                    []),
                (
                    """DIVISION 10 DIVISION 10 - SPECIALTIES 10000 Allowance to Install Owner Supplied Huddled Area 
                    Safety Items 3,000 10100 Markerboards 119,000 10160 Metal Toilet Compartments 45,000 10500 Metal 
                    Lockers w/10160 10520 Fire Extinguishers and Cabinets 6,000 10810 Toilet Accessories w/10160 
                    10950 Building Specialties - Corner Guards Allowance 41,000 DIVISION 11 DIVISION 11 - EQUIPMENT 
                    11130 Projection Screens w/10100 11160 Loading Dock Equipment NIC by Base Bldg. 11400 Food 
                    Service Equipment 838,000 11601 Laboratory Fume Hoods w/12345 11604 Laboratory Fittings w/12345 
                    11605 Laboratory Equipment Install Allowance 78,000 DIVISION 12 DIVISION 12 - FURNISHINGS 12345 
                    Laboratory Casework 2,860,000 12514 Vertical Blinds 12515 Roller Shades 110,000 12670 Entrance 
                    Mats 2,000 12710 Fixed Seating 70,000 DIVISION 13 DIVISION 13 - SPECIAL CONSTRUCTION 13038 
                    Controlled Temp Rooms 145,000 DIVISION 14 DIVISION 14 - VERTICAL TRANSPORTATION 14100 Elevators 
                    690,000 14200 Geared Elevators w/14100 14400 Handicap Lift 18,000 DIVISION 15 DIVISION 15 - 
                    MECHANICAL 15300 Fire Protection 698,000 15420 Plumbing 3,150,000 15600 HVAC 9,773,000 AHU 
                    Pre-Purchase 1,438,000 15900 Building Control System 1,169,000 DIVISION 16 DIVISION 16 - 
                    ELECTRICAL 16100 Electrical 7,025,000 16500 Archetechural Lighting Fixtures w/16100 16700 
                    Tele / Data 839,000 18000 Commissioning w/trades ================= 
                    ============================================================= ================== 
                    =================""",
                    []),
                (
                    """As of March 15, 2016 Premier Pacific Construction, Inc. had 5,169,000 shares of common stock 
                    outstanding.""",
                    [datetime.date(2016, 3, 15)]),
                (
                    """2.13 EQUITY. In addition to the Required Equity Funds, Borrower has used the following sums 
                    for construction purposes at the following Projects and such sums shall not be reimbursed from 
                    the proceeds of the Loan: Aversana $13,144,901.00 Savona 8,681,981.00 Grande Isle I & II 
                    7,073,050.00""",
                    []),
                (
                    """1. The comprehensive fixed, unilateral construction price for this project is RMB ￥500/m² 
                    turkey (including the safety construction fee, and ￥20/m² of unforeseeable fee).  The Contract 
                    Price is tentatively set at RMB ￥36,810,000 (in words: Thirty Six Million and Eight Hundred and 
                    Ten Thousand yuan).  Regardless of any reason, the price shall not be adjusted. The construction 
                    area is subject to the Housing Authority’s mapping data at the time of payment.""",
                    []),
                (
                    """The Contract Price is tentatively set at RMB ￥36,810,000 (in words: Thirty Six Million and 
                    Eight Hundred and Ten Thousand yuan).""",
                    []),
                ("""Totals 6,576,210.93 Loan 12/11/2007	 	 	 	10,000,000.00""",
                 [datetime.date(2007, 12, 11)]),
                (
                    """The Contract Price is USD 70,016,819.- (Seventy Million, Sixteen Thousand, Eight Hundred and 
                    Nineteen).""",
                    []),
                (
                    """Contract Price: The contract price of the DRILLSHIP, including the drilling equipment package 
                    and the subsea equipment package, is United States Dollars Five Hundred Fifty Seven Million 
                    (USD 557,000,000.00) plus or minus the amount, if any, set forth in subclause 3, below (the 
                    “CONTRACT PRICE”), inclusive of the PC SUM, net receivable by BUILDER, which is exclusive of 
                    BUYER’S SUPPLIES.""",
                    []),
                (
                    """59. Increase 5no. of doors’ width Location: First Floor, Service Lift Lobby The opening width 
                    of 5 nr doors within the service lift lobby is increased to allow easier access for housekeeping 
                    service cart. Change Request Form: 192 Design Action Notification: DAN/ 0053 Increase to the 
                    Guaranteed Maximum Price: Nil 60. Upgrade Lift Lobby finishes Location: Multi-storey Car Park 
                    The lift and lift lobby interior finishes at the car park are upgraded from a “back of house” 
                    to a “front of house” standard and air conditioning is introduced to the lobbies. Change Request 
                    Form: 194 Design Action Notification: DAN/ 0060 Increase to the Guaranteed Maximum Price: HK$ 
                    800,000/ USD 102,565 61. Revised layout of Staff Entrance Location: Podium External Entry The 
                    internal layout at the Staff Entry is revised to incorporate a separate entry to the recruitment 
                    office with access/ egress control by automatic turnstiles. Change Request Form: 196 Design 
                    Action Notification: DAN/ 0068 Increase to the Guaranteed Maximum Price: Nil 62. Marquee Sign 
                    foundation Location: North West Corner of Site The reinforced concrete foundation for the Wynn 
                    Marquee sign is added to the Contractor’s scope of work. Change Request Form: 197 Design Action 
                    Notification: DAN/ 0061 Increase to the Guaranteed Maximum Price: HK$ 8,000,000/ USD 1,025,641""",
                    []),
                (
                    """Current liabilities Current maturities of long-term debt $ 745 $ 1,130 Accounts payable 
                    100,816 90,111 Billings in excess of costs and estimated earnings 87,149 57,412 Accrued expenses 
                    and other current liabilities 81,311 82,924 Total current liabilities 270,021 231,577 Long-term 
                    debt 138,364 63,891 Other long-term liabilities 9,607 6,370 Deferred income taxes 31,540 31,540 
                    Commitments and contingencies Stockholders' equity Preferred stock, $0.01 par value, authorized 
                    3,000,000 shares, none outstanding -- -- Common stock, $0.01 par value, authorized 100,000,000 
                    shares; issued and outstanding 41,107,224 shares in 2001 and 40,881,908 in 2000 411 409 
                    Additional paid-in capital 61,421 56,381 Retained earnings 338,066 330,172 Accumulated other 
                    comprehensive loss (112) -- 399,786 386,962 Unearned compensation (13,818) (9,198) 385,968 377,764 
                    $ 835,500 $ 711,142 ========= ========= </TABLE>""",
                    []),
                (
                    """Tenant Improvements/Vanilla (559,150) (356,250) (198,400) (32,000) (1,922,139) (1,928,150) 
                    6,011 Leasing Commissions (26,348) (55,250) (45,435) (54,500) (301,065) (394,900) 93,835 
                    Renovation and Replacements 0 0 0 0 (218,536) (200,000) (18,536) TOTAL CAPITAL (585,498) 
                    (411,500) (243,835) (86,500) (2,441,740) 2,523,050 81,310""",
                    []),
                (
                    """Power and data for additional ATMs Location: Podium Areas Power and data provisions are 
                    included for additional ATMs at the following locations: Chinese Restaurant – 1nr Food Court 
                    – 1nr Retail Promenade – 2nr Staff Dining – 2nr Change Request Form: 133 Design Action 
                    Notification: DAN/ 0015 Increase to the Guaranteed Maximum Price: HK$ 30,000/ USD 3,831 19.""",
                    []),
                ("""Revised MCA OCIP Deduction 26,236,418 Mutually Agreed Upon Credit Adjustments (2,000,000	)""",
                 []),
                (
                    """percent (66 2/3%) of the outstanding principal balance of the Loan held by Non-Defaulting 
                    Lenders.""",
                    []),
                (
                    """REQUIRED LENDERS. As of any date of determination prior to termination of the Commitments, 
                    Lenders (excluding Defaulting Lenders) whose aggregate Loan Percentages constitute at least 
                    sixty-six and two-thirds percent (66 2/3%) of the Commitments held by Non-Defaulting Lenders. As 
                    of any date of determination occurring after the termination of the Commitments, Lenders 
                    (excluding Defaulting Lenders) holding at least sixty-six and two-thirds percent (66 2/3%) 
                    of the outstanding principal balance of the Loan held by Non-Defaulting Lenders.""",
                    []),
                (
                    """22.1 Definition of Force Majeure 1-64 22.2 Effect of Force Majeure 1-64 22.3 Notice of 
                    Occurrence 1-64 22.4 Performance to Continue 1-65 22.5 Additional Cost Caused by Force Majeure 
                    1-65 22.6 Damage Caused by Force Majeure 1-65 22.7 Termination in Consequence of Force Majeure 
                    1-65 22.8 Payment on Termination for Force 1-65 Majeure 22.9 Release from Performance 1-66 22.10 
                    Force Majeure Affecting Engineer's 1-66 Duties""",
                    []),
                ("""19.1 Procedure 1-61 19.2 Assessment 1-62""", []),
                (
                    """18.1 Methods of Application 1-57 18.2 Issue of Certificate of payment 1-57 18.3 Corrections 
                    to Certificates of Payment 1-58 18.4 Payment 1-58 18.5 Delayed Payment 1-58 18.6 Remedies on 
                    Failure to Certify or Make Payment 1-58 18.7 Application for Final Certificate of Payment 1-59 
                    18.8 Issue of final Certificate of Payment 1-59 18.9 Final Certificate of Payment conclusive 1-60 
                    18.10 Advance Payment 1-60 18.11 Advance Payment Guarantee 1-60 18.12 Terms of Payment 1-60 18.13 
                    Retention 1-61""",
                    []),
                ("""56 of March 22nd, 1983; Seismic Code by Executive Decree No.""", [datetime.date(1983, 3, 22)]),
                (
                    """16.1 Engineer's Right to Vary 1-53 16.2 Variation in Excess of 5% 1-54 16.3 Variation Order 
                    Procedure 1-54 16.4 Disagreement on Adjustment of Contract Price 1-55 16.5 Variation on 
                    Manufacture and Drawings 1-55 16.6 Contractor to Proceed 1-56 16.7 Records of Costs 1-56 16.8 
                    Monthly Variations Statement 1-56""",
                    []),
                (
                    """2.1.1, 3.3.3, 3.12.4, 3.12.8, 3.12.10, 4.1.2, 4.2.1, 4.2.2, 4.2.3, 4.2.6, 4.2.7, 4.2.10, 
                    4,2.12, 4.2.13, 4.4, 5.2.1, 7.4, 9.4.2, 9.6.4, 9.6.6""",
                    []),
                (
                    """11.1 Notice of Tests 1-41 11.2 Time for Tests 1-41 11.3 Delayed Tests 1-42 11.4 Facilities for 
                    Tests on Completion 1-42 11.5 Notice of Test Results 1-42 11.6 Retesting 1-42 11.7 Disagreement as 
                    to Result of Test 1-43 11.8 Consequences of Failure to Pass Tests 1-43 on Completion 11.9 Test 
                    Certificate 1-43 11.10 Test by Employer's Operators 1-44""",
                    []),
                (
                    """All notices, consents, requests and other communications hereunder shall be in writing and shall
                     be deemed to have been duly given when (a) delivered by hand, (b) sent by telecopier (with receipt 
                     confirmed), provided that a copy is sent in the manner provided in clause (c), or (c) when 
                     received by the addressee, if sent by DHL, Federal Express, Airborne Express or other generally 
                     recognized international express delivery service (receipt requested), in each case to the 
                     appropriate addresses and telecopier numbers set forth below (or to such other addresses and 
                     telecopier numbers as a party may designate as to itself by notice to the other parties): (i) 
                     If to CCKK: SoftBank Corp. 3-42-3 Nihonbashi-Hamacho Chuo-ku Tokyo 103 Japan Telecopier No.""",
                    []),
                ("""4.2.2, 4.2.9, 4.3,4, 9.4.2, 9.8.3, 9.9.2, 9.10.1, 13.5""", []),
                ("""Size 19 1/2" L x 7/8" H.""", []),
                (
                    """3. The Delivery of the Vessel shall be deferred and the revised Delivery Date shall be 30 July 
                    2017. Parties shall, following execution of this Amendment No. 1, discuss and agree on the 
                    necessary changes to the Programme (i.e. Paragraph 4.1 of the Contract), so as to effect the 
                    revised Delivery Date. To the extent the Parties have yet to agree or have not agreed on necessary 
                    changes to the Programme, all references to the Programme in the Contract shall operate on the 
                    basis that (1) KD7 (Commencement of Commissioning Process) is revised to 31 October 2016 and (2) 
                    KD 11 (Delivery of the Vessel) is revised to 30 July 2017, and all other milestones and Key Dates
                     are inoperative.""",
                    [datetime.date(2017, 7, 30), datetime.date(2016, 10, 31)]),
                ("""""", []),
                ]

    # Add random examples
    p = 0.01
    for k in range(1980, 2010):
        for j in range(1, 13):
            for i in range(1, 31):
                if random.random() <= p:
                    year = k
                    month = j
                    day = i
                    try:
                        d = datetime.date(year, month, day)
                        n = random.randint(2, 30)
                        d2 = d + datetime.timedelta(days=n)
                        examples.append(("""on {0}-{1}-{2}""".format(year, month, day),
                                         [d]))
                        examples.append(("by " + d.strftime("%b %d, %Y"),
                                         [d]))
                        examples.append(("on " + d.strftime("%B %d, %Y"),
                                         [d]))
                        examples.append(("{0} to {1}".format(d, d2),
                                         [d, d2]))
                        examples.append(("{0} to {1}".format(d.strftime("%b d, %Y"), d2.strftime("%b d, %Y")),
                                         [d, d2]))
                        examples.append(("{0} through {1}".format(d.isoformat(), d2.isoformat()),
                                         [d, d2]))
                        examples.append(("{0} through {1}".format(d.strftime("%b d, %Y"), d2.strftime("%b d, %Y")),
                                         [d, d2]))
                    except ValueError:
                        continue

    # Output
    output_path = 'test_date_model.pickle'
    if save:
        output_path = os.path.join(MODULE_PATH, 'date_model.pickle')

    build_date_model(examples, output_path,
                     lambda date_str: get_raw_date_list(date_str, strict=False, return_source=True),
                     characters=DATE_MODEL_CHARS)
    if not save:
        os.unlink("test_date_model.pickle")


parser = DateParser(DATE_MODEL_CHARS, enable_classifier_check=True, locale=Locale('en-US'), classifier_model=MODEL_DATE)
_get_dates = parser.get_dates
_get_date_list = parser.get_date_list
