# Language capability parity — English, Spanish, Portuguese (pt-BR)

**Date:** 2026-09-08 · **Branch:** `pr30-triage`

This document answers one question: *what would it take for `es` and `pt-BR` to
do what `en` does?* Every cell below was produced by running the extractors,
not by reading the source tree — several capabilities that **exist** as modules
turned out to extract nothing, and one that looked absent was merely named
differently.

**Headline:** Portuguese is close to English and the remaining gaps are small
and specific — both cover 15 of the 15 capabilities that are comparable across
the three languages. Spanish covers **6**, and closing that is a build project,
not a wiring job.

---

## 1. Capability matrix (measured)

Each extractor was called with a representative sentence in its own language.
`OK(n)` means *n* annotations returned; `none` means the module exists and ran
but produced nothing; `absent` means there is no such module.

| Capability | en | es | pt-BR | Notes |
| --- | :---: | :---: | :---: | --- |
| money | OK | **absent** | OK | pt correctly resolves `R$` → BRL |
| percents | OK | **absent** | OK | |
| amounts | OK | **absent** | OK | pt parses spelled-out numerals incl. pt-PT and pre-1990 spellings |
| durations | OK | **absent** | OK | |
| distances | OK | **absent** | OK | |
| ratios | OK | **absent** | OK | pt false-positives on CPF/CNPJ (§5) |
| citations | OK | **absent** | OK | |
| trademarks | OK | **absent** | OK | |
| urls | OK | **absent** | OK | |
| dates | OK | OK | OK | all three wired through `all_locales` |
| regulations | OK | OK | OK | es 25 patterns vs pt 78 |
| definitions | OK | partial | OK | es matches `se refiere a` but **not** `significa` or `se entiende por` |
| courts | OK | partial | OK | see §3 — was broken in es until this branch |
| copyrights | OK¹ | OK | OK | ¹ en module is `copyright.py`, singular (§6) |
| national IDs² | OK | OK | OK | ² one row, because the module differs by language: en puts SSN in `pii`, es/pt put NIF/CPF in `identifiers` (§6) |
| **Working total** | **15 / 15** | **6 / 15** | **15 / 15** | counting a `partial` as working |

Portuguese and English tie at 15. That is not a claim that Portuguese is as
good as English — the English dictionaries are far deeper (§3) and English has
extractors Portuguese does not, such as `acts`, `cusip` and `conditions`, which
are outside this comparison because they have no Spanish or Portuguese analogue
to compare against. It means the *capability surface* is comparable, and that
Spanish is the outlier.

**Caveat on the shared row:** the entry points are not interchangeable.
`pt/pii.py` covers only email and phone, so a caller asking a Portuguese
document for its PII gets **no CPF or CNPJ** — those are reachable only through
`pt/identifiers.py`. English does not behave this way. See §4.

### What Spanish is missing outright

`money` · `percents` · `amounts` · `durations` · `distances` · `ratios` ·
`citations` · `trademarks` · `urls` — nine modules that simply do not exist.

There is no Spanish number-word parser, no Spanish currency handling, and no
Spanish unit handling. Note that Spanish and Portuguese share the
`1.500.000,00` convention, so the numeric groundwork in `pt/amounts.py` is
directly adaptable — this is the single highest-leverage starting point.

---

## 2. Dispatcher reachability

A capability is only usable if `all_locales` routes to it. Portuguese modules
existed but were unreachable until this branch — `get_money_annotations("pt-BR", …)`
silently used the **English** parser and reported `R$ 1.500.000,00` as **USD**.
Fixed in `d11fcdb`; see the PR #31 audit, §4.1.

| Dispatcher | Routes to | Gap |
| --- | --- | --- |
| `dates`, `definitions`, `copyrights` | de, en, es, pt | — |
| `money`, `percents`, `amounts`, `durations`, `citations` | de, en, pt | **es** — no module to route to |
| `geoentities` | de, en | es, pt |
| `court_citations` | de | en, es, pt |

`courts` dispatches through a separate config-driven path and is not part of
this table.

**Rule worth keeping:** a dispatcher must resolve the *effective* language
before branching on calling convention. Branching on the requested locale while
selecting the routine through a fallback lets the two disagree — that is how
`fr`/`it`/`nl` started raising `TypeError` in the date dispatcher. The five
dispatchers touched here now normalise first.

---

## 3. Courts — dictionary depth is the remaining gap

Spanish court extraction returned **zero for every input**, including names
copied verbatim from its own CSV. That was the shared-parser offset bug
(audit §4.2), fixed in `daaeb8e`. With it fixed:

| | Dictionary rows | Status |
| --- | ---: | --- |
| en (`us_courts` + `us_state_courts` + …) | 2,217 | comprehensive |
| de | 173 | good |
| pt-BR | 98 | good — STF/STJ/TST with aliases |
| **es** | **17** | **regional high courts only** |

The Spanish dictionary contains the 17 *Tribunales Superiores de Justicia* and
nothing else. `Tribunal Supremo`, `Tribunal Constitucional`, `Audiencia
Nacional`, the *Audiencias Provinciales* and the *Juzgados* are all absent —
i.e. the courts that appear most often in Spanish legal text.

Both es and pt are **accent-sensitive**: `Superior Tribunal de Justiça` matches,
`Superior Tribunal de Justica` (OCR output, or a keyboard without a cedilla)
does not. For a library whose stated purpose includes OCR-mangled text, folding
diacritics at lookup time is worth doing.

---

## 4. Identifiers and PII — shapes verified against `faker`

Shapes were taken from `faker`'s locale providers, and 60 generated values per
shape were run through the extractors, bare and embedded in a sentence
(identical results both ways).

| Locale | Shape | Format | Detected |
| --- | --- | --- | ---: |
| pt-BR | CPF | `###.###.###-##` | **60/60** |
| pt-BR | CNPJ | `##.###.###/####-##` | **60/60** |
| pt-BR | phone | `## ####-####` | 59/60 |
| pt-BR | **RG** | `#########` | **0/60** |
| pt-BR | **CEP** | `#####-###` | **0/60** |
| es | NIF | `########X` | **60/60** |
| es | CIF | `X#######X` | **60/60** |
| es | NIE | `[XYZ]#######X` | **60/60** |
| es | **postal code** | `#####` | **0/60** |
| en | SSN | `###-##-####` | **60/60** |
| en | phone | various | **38/60** |

Three findings:

1. **`pt.pii` does not see CPF at all — 0/60.** CPF and CNPJ live in
   `pt/identifiers.py`; `pt/pii.py` covers only email and phone. A caller that
   reasonably asks for "the PII" in a Portuguese document gets **no national
   identifiers**. English does not behave this way — `en/pii.py` returns SSN.
   This is the most likely of these to bite someone in production.
2. **English phone detection is the weakest cell in the table (38/60)** —
   weaker than Portuguese. Extension formats (`...x2020`) and bare 10-digit
   strings are missed. Spanish and Portuguese are ahead of English here.
3. RG, CEP and Spanish postal codes are unimplemented. `faker` gives exact
   generators for all three, which makes them cheap to build and to test.

---

## 5. Known false positives

Portuguese ratio extraction fires inside Brazilian identifiers:
`CNPJ 12.345.678/0001-95` yields a "ratio". This predates the branch — only
the *shape* of the wrong answer changed when the digit-truncation bug was
fixed. Since `pt/identifiers.py` already recognises CPF and CNPJ with 60/60
accuracy, the clean repair is to suppress ratio matches that fall inside a
detected identifier span.

---

## 6. Naming inconsistencies

Cross-language code has to special-case these, and each one is a place for a
dispatcher to silently miss a language:

| Inconsistency | Detail |
| --- | --- |
| `copyright` vs `copyrights` | en uses the singular; de/es/pt use the plural |
| PII vs identifiers | en puts SSN in `pii`; es/pt put national IDs in `identifiers` |
| `EsLanguageTokens.conjunctions` | contains `["und", "oder"]` — **German** (see audit §4.3 for why the fix is coupled and deferred) |

---

## 7. What parity would take

Ordered by value per unit of work.

### Tier 1 — small, high value

1. **`pt.pii` should surface CPF/CNPJ.** Delegate to `pt/identifiers.py`.
   Removes a silent PII miss. *Small.*
2. **Suppress pt ratio matches inside identifier spans** (§5). *Small.*
3. **Add RG, CEP and Spanish postal codes**, tested against `faker` shapes.
   *Small.*
4. **Fold diacritics in court lookup**, so OCR text matches. *Small.*
5. **Improve English phone coverage** to Portuguese's level. *Small.*

### Tier 2 — moderate, unblocks Spanish

6. **Expand `es_courts.csv`** from 17 rows to cover Tribunal Supremo, Tribunal
   Constitucional, Audiencia Nacional, the Audiencias Provinciales and the
   Juzgados. Data entry, not engineering. *Moderate.*
7. **Broaden Spanish definition patterns** to `significa`, `se entiende por`,
   `a los efectos de`. *Moderate.*
8. **Fix `EsLanguageTokens` and its `line_breaks` coupling together** (§6).
   *Moderate — must be done as one change.*

### Tier 3 — the actual Spanish gap

9. **`es/amounts.py`** — port the numeral and separator logic from
   `pt/amounts.py`; the `1.500.000,00` convention is shared. Everything below
   depends on this. *Large.*
10. **`es/money.py`** (EUR, and Latin-American pesos if in scope),
    **`es/percents.py`**, **`es/durations.py`**, **`es/distances.py`**,
    **`es/ratios.py`** — each is small *once* step 9 exists. *Large in
    aggregate.*
11. **`es/citations.py`**, **`es/trademarks.py`**, **`es/urls.py`**. `urls` is
    language-independent and could be promoted to `common` rather than
    reimplemented three times. *Moderate.*
12. **Register each new module in its `all_locales` dispatcher** as it lands,
    with a routing test. Without this the module exists and nobody can reach
    it — exactly the state Portuguese was in before this branch.

### Sequencing note

Steps 9–11 should land with the dispatcher registration (12) in the *same*
change. The Portuguese experience is the argument: five working extractors sat
unreachable in the tree, and the failure was invisible because the dispatcher
returned confident English answers instead of an error.

---

## 8. How to reproduce

The measurement scripts are not committed — they are throwaway instruments.
Both are described in the PR #31 audit, §1: a differential probe across
`master` and the branch, and a `faker`-driven identifier sweep. `faker` is not
a project dependency; it was installed into the dev virtualenv for measurement
only, and the committed tests use fixed literal samples instead.
