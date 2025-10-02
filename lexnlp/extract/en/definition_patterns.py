"""Registry of definition extraction patterns used by the English parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import regex as re

from lexnlp.extract.en.en_language_tokens import EnLanguageTokens


@dataclass
class DefinitionPattern:
    """Container describing a compiled definition-related regular expression."""

    name: str
    template: str
    flags: int
    description: str = ""
    group: str = "general"
    _compiled: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.template, self.flags))

    @property
    def regex(self) -> re.Pattern:
        """Return the compiled regular expression for the pattern."""
        return self._compiled


class DefinitionPatternRegistry:
    """Registry collecting the patterns used by the definition extractor."""

    DEFAULT_STRIP_PUNCT_SYMBOLS = ",:-"

    def __init__(
        self,
        max_term_chars: int = 64,
        max_term_tokens: int = 5,
        max_quoted_term_tokens: int = 7,
    ) -> None:
        self.max_term_chars = max_term_chars
        self.max_term_tokens = max_term_tokens
        self.max_quoted_term_tokens = max_quoted_term_tokens

        self._patterns: Dict[str, DefinitionPattern] = {}

        self.strong_triggers = self._sorted(
            [
                "shall have the meaning",
                r"includes?",
                "as including",
                "shall mean",
                r"means?",
                r"shall (?:not\s+)?include",
                "shall for purposes",
                "have meaning",
                "referred to",
                "known as",
                "refers to",
                "shall refer to",
                "as used",
                "for purpose[sd]",
                "shall be deemed to",
                "may be used",
                "is hereby changed to",
                "is defined",
                "shall be interpreted",
            ]
        )
        self.weak_triggers = self._sorted([r"[\(\)]", "in "])
        self.all_triggers = self._sorted(self.strong_triggers + self.weak_triggers)

        self.articles: Sequence[str] = ["the", "a", "an"]
        self.anchor_words_case5: Sequence[str] = [
            "called",
            "herein",
            "herein as",
            "collectively(?:,)?",
            "collectively as",
            "individually(?:,)?",
            "individually as",
            "together(?:,)?",
            "together with",
            "referred to as",
            "being",
            "shall be",
            "definition as",
            "known as",
            "designated as",
            "hereinafter",
            "hereinafter as",
            "hereafter",
            "hereafter as",
            "its",
            "our",
            "your",
            "any of the foregoing,",
            "in such capacity,",
            "in this section,",
            "in this paragraph,",
            r"in this \(noun\),",
            "each such",
            "this",
        ]
        self.anchor_words_case6: Sequence[str] = ["such", "any such", "together"]

        self._build_patterns()

    @staticmethod
    def _sorted(collection: Sequence[str]) -> List[str]:
        return sorted(collection, key=len, reverse=True)

    @staticmethod
    def _join_collection(collection: Iterable[str]) -> str:
        return "|".join([w.replace(" ", r"\\s+") for w in collection])

    def _build_patterns(self) -> None:
        join_triggers = self._join_collection(self.all_triggers)
        join_strong_triggers = self._join_collection(self.strong_triggers)
        join_articles = self._join_collection(self.articles)
        join_anchor_case5 = self._join_collection(self.anchor_words_case5)
        join_anchor_case6 = self._join_collection(self.anchor_words_case6)

        trigger_words_template = r"""
(?:(?:word|term|phrase)s?\s+|[:,\.]\s*|^)
['"“].{{1,{max_term_chars}}}['"”]\w{{0,2}}\s*
(?:{trigger_list})[\s,]
""".format(max_term_chars=self.max_term_chars, trigger_list=join_triggers)

        simple_trigger_template = r"""
(['"“].{{1,{max_term_chars}}}['"”]\w{{0,2}})\s*(?:{trigger_list})
""".format(max_term_chars=self.max_term_chars, trigger_list=join_triggers)


        extract_template = r"""
“(.+?)“|
"(.+?)"|
'(.+?)'
"""

        noun_base_template = r"""
(
    (?:[A-Z][-A-Za-z']*+(?:\s*[A-Z][-A-Za-z']*){{0,{max_term_tokens}}})
    |
    (?:[A-Z][-A-Za-z'])
)
""".format(max_term_tokens=self.max_term_tokens)

        noun_template = r"""
(?:^|\s)
(?:
    {noun_ptn_base}
    |
    "{noun_ptn_base}"
    |
    “{noun_ptn_base}”
)
\s+(?=(?:{trigger_list})\W)
""".format(noun_ptn_base=noun_base_template, trigger_list=join_strong_triggers)

        noun_anti_template = r"""the\s*"""

        paren_quote_template = r"""\((?:each(?:,)?\s+)?(?:(?:{articles})\s+)?['"“](.{{1,{max_term_chars}}}?)\.?['"”]\)""".format(
            articles=join_articles, max_term_chars=self.max_term_chars
        )

        paren_template = r"""\((?:E|each(?:,)?\s+)?(?:(?:{articles})\s+)?([A-Z][^\)]{{1,{max_term_chars}}}?)\.?\)""".format(
            articles=join_articles, max_term_chars=self.max_term_chars
        )

        colon_template = r"""((['](.{{1,{max_term_chars}}})['])|(([\"](.{{1,{max_term_chars}}})[\"]))|(([“](.{{1,{max_term_chars}}})[”])))""" r"""[:\s]""".format(
            max_term_chars=self.max_term_chars
        )

        anchor_quotes_template = r"""(?:(?:{anchor})\s+)(?:(?:{articles})\s+)?['"“](.{{1,{max_term_chars}}}?)['"”]""".format(
            anchor=join_anchor_case5, articles=join_articles, max_term_chars=self.max_term_chars
        )

        anchor_subject_quotes_template = r"""(?:(?:{anchor})\s+?)(?:.{{1,{max_term_chars}}}\s+?)(?:(?:{articles})\s+)?""" r"""((('(.{{1,{max_term_chars}}}?)')|(\"(.{{1,{max_term_chars}}}?)\")|(“(.{{1,{max_term_chars}}}?)”)))""".format(
            anchor=join_anchor_case6, articles=join_articles, max_term_chars=self.max_term_chars
        )

        trigger_quoted_definition_template = r"""['"“][^'"“]{{1,{max_term_chars}}}['"”]""".format(
            max_term_chars=self.max_term_chars
        )

        quoted_text_template = r"""(["'“„])(?:(?=(\\?))\2.)+?\1"""

        abbreviation_ptrn = "|".join([a.replace('.', r"\.") for a in EnLanguageTokens.abbreviations])
        abbreviation_ending_template = f"({abbreviation_ptrn})$"

        split_subdefinitions_template = r"""["“](?:[^"“]{{1,{max_term_chars}}})["“]""".format(
            max_term_chars=self.max_term_chars
        )

        spaces_template = r"""\s+"""

        patterns = [
            DefinitionPattern(
                name="trigger_words",
                template=trigger_words_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Term preceded by trigger words and enclosed in quotes",
                group="trigger",
            ),
            DefinitionPattern(
                name="simple_trigger",
                template=simple_trigger_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Fallback trigger detection for quoted terms",
                group="trigger",
            ),
            DefinitionPattern(
                name="quoted_term_extractor",
                template=extract_template,
                flags=re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Extract quoted definitions from trigger match",
                group="extract",
            ),
            DefinitionPattern(
                name="noun",
                template=noun_template,
                flags=re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Capitalised noun followed by strong trigger",
                group="candidate",
            ),
            DefinitionPattern(
                name="noun_anti",
                template=noun_anti_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Anti-pattern for nouns beginning with 'the'",
                group="candidate",
            ),
            DefinitionPattern(
                name="paren_quote",
                template=paren_quote_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Definition enclosed in parentheses and quotes",
                group="quoted_definition",
            ),
            DefinitionPattern(
                name="paren",
                template=paren_template,
                flags=re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Definition enclosed in parentheses",
                group="quoted_definition",
            ),
            DefinitionPattern(
                name="colon",
                template=colon_template,
                flags=re.UNICODE | re.DOTALL | re.MULTILINE,
                description="Definition preceding a colon",
                group="quoted_definition",
            ),
            DefinitionPattern(
                name="anchor_quotes",
                template=anchor_quotes_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Definition following anchor phrases",
                group="quoted_definition",
            ),
            DefinitionPattern(
                name="anchor_subject_quotes",
                template=anchor_subject_quotes_template,
                flags=re.IGNORECASE | re.UNICODE | re.DOTALL | re.MULTILINE | re.VERBOSE,
                description="Definition following anchor phrases with subject",
                group="quoted_definition",
            ),
            DefinitionPattern(
                name="trigger_quoted_definition",
                template=trigger_quoted_definition_template,
                flags=re.DOTALL,
                description="Quick check for quoted definitions",
                group="trigger",
            ),
            DefinitionPattern(
                name="quoted_text",
                template=quoted_text_template,
                flags=re.UNICODE | re.IGNORECASE | re.DOTALL,
                description="General quoted text extractor",
                group="utility",
            ),
            DefinitionPattern(
                name="spaces",
                template=spaces_template,
                flags=0,
                description="Whitespace normaliser",
                group="utility",
            ),
            DefinitionPattern(
                name="abbreviation_ending",
                template=abbreviation_ending_template,
                flags=0,
                description="Detect abbreviation endings",
                group="utility",
            ),
            DefinitionPattern(
                name="split_subdefinitions",
                template=split_subdefinitions_template,
                flags=re.DOTALL,
                description="Identify multiple definitions within a term",
                group="utility",
            ),
        ]

        for pattern in patterns:
            self._patterns[pattern.name] = pattern

    def get(self, name: str) -> DefinitionPattern:
        return self._patterns[name]

    def group(self, name: str) -> List[DefinitionPattern]:
        return [pattern for pattern in self._patterns.values() if pattern.group == name]

    @property
    def strip_punct_symbols(self) -> str:
        return self.DEFAULT_STRIP_PUNCT_SYMBOLS


DEFAULT_PATTERN_REGISTRY = DefinitionPatternRegistry()
"""
Default registry instance used by callers that do not provide their own configuration.
"""
