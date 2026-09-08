# PR #31 — regression and artifact audit

**Branch:** `pr30-triage` → `master` (cicero-im/lexpredict-lexnlp)
**Date:** 2026-09-08
**Question asked:** did PR #31 introduce regressions or lose features, and what
is the truth behind the `30 pickle, 0 skops` artifact figure?

**Answer:** no regressions and no lost features. The one behavioural change
against `master` is a bug *fix*. The `30 pickle, 0 skops` figure was wrong in
both halves. Three genuine defects surfaced *during* this audit — all of them
older than PR #31 — and all three are now fixed.

---

## 1. Method

Claims about "no regressions" are worthless unless they are measured, so this
audit compares running code rather than reading diffs.

`master` was checked out into a second worktree and both trees were driven by
the same interpreter and the same probe:

| Instrument | What it does |
| --- | --- |
| Differential probe | Walks `lexnlp.extract`, `lexnlp.nlp`, `lexnlp.utils`, discovers every public callable matching the extraction naming conventions, calls each one against a fixed multi-text corpus, and normalises the result into a comparable form |
| Corpus | 32 texts — 13 English, 7 Portuguese, 6 Spanish, 6 German — covering money, dates, ratios, durations, distances, citations, definitions, copyright, identifiers, OCR-mangled accents, thousands separators in three conventions, and the empty string |
| Public-symbol diff | Imports all 247 modules under both trees and diffs every exported function, class and value |
| Test-file audit | Diffs every pre-existing test file and counts assertion deltas |

Set-valued results are sorted before comparison, because two of them are
genuinely unordered and would otherwise show up as false differences.

## 2. Result: no regressions

| Check | master | branch | verdict |
| --- | --- | --- | --- |
| Modules importable | 247 | 252 | 5 added, **0 removed** |
| Public symbols removed | — | — | **none real** (see §2.1) |
| Probes executed | 307 | 309 | 2 new entry points |
| Probes differing | — | **4** | all one module, a fix (§2.2) |
| Import errors | 0 | 0 | — |
| Files deleted vs master | — | 2 | `.travis.yml`, `readthedocs.yml` — both intentionally retired |
| Pre-existing tests deleted | — | 0 | — |
| Pre-existing tests weakened | — | 0 | 18 modified, all balanced or strengthened (§2.3) |

### 2.1 The only "removed" symbols are not API

`lexnlp.ml.catalog.download` stopped exporting `CaseInsensitiveDict`,
`Generator` and `md5`. All three belong to `requests.structures`,
`collections.abc` and `hashlib` respectively — `master` re-exported them by
accident simply by importing them at module scope. The branch reorganised the
imports (`hashlib.md5` is now called qualified, `Mapping` replaced
`CaseInsensitiveDict`). The module's own API, including `verify_md5`, is
intact, and the digest call now passes `usedforsecurity=False`.

Nothing a caller could reasonably depend on was removed.

### 2.2 The one behavioural change is a fix

All four differing probes are the same module — `lexnlp.extract.pt.ratios` —
seen through its four entry points. The branch added a `(?!\d)` guard that
stops the regex engine backtracking the right-hand operand to a shorter number
in order to escape the clock-time lookahead.

`master` was silently truncating the right operand of every ratio:

| Portuguese input | master | branch |
| --- | --- | --- |
| `Lei 1/2020` | 1 : **202** | 1 : **2020** |
| `Lei 12/2021` | 12 : **202** | 12 : **2021** |
| `o indice 1/100000` | 1 : **100** | 1 : **100000** |
| `Decreto 9.999/2019` | 9999 : **201** | 9999 : **2019** |
| `as 10:30 a.m.` | 10 : 3 *(clock guard defeated)* | *(correctly suppressed)* |

`master` dropped up to three digits from a legal citation's year and its
clock-time guard never fired. This is unambiguously a repair.

**Known limitation, unchanged by the fix:** Portuguese ratio extraction still
fires inside CPF/CNPJ identifiers (`12.345.678/0001-95` yields a "ratio").
Both trees do this; the shape of the false positive differs but its existence
does not. Tracked in the language parity document.

### 2.3 Modified tests were strengthened, not loosened

18 pre-existing test files changed. Assertion counts balance in 17 of them.
The exception, `scripts/tests/test_reexport_bundled_sklearn_models.py`, shows
−3/+2 — because three near-identical tests were merged into one parametrised
test covering **four** error types instead of three.

That file is also the audit's clearest example of why counting assertions is
not enough: the *old* tests patched `lexnlp.utils.unpickler.renamed_load`,
which the code path under test no longer calls. The patches were inert and the
tests passed without exercising anything. The replacements patch
`lexnlp.ml.model_io.pickle.load`, which is actually invoked.

Two things that looked like regressions in that diff, and are not:

- `KeyError` appears to have left the recoverable-exception set. It had not —
  `master`'s clause is byte-identical: `(pickle.UnpicklingError, ValueError,
  EOFError, AttributeError)`. The old docstring was simply wrong.
- `reexport_layered_definition_models` appears renamed. It was not — the
  `_pickle` suffix is already `master`'s name; only the stale test caught up.

---

## 3. Artifacts: the `30 pickle, 0 skops` figure was wrong

That figure appeared in an earlier comparison table of mine. It is wrong in
both halves, and I should not have published it without counting. The measured
truth, on **both** `master` and this branch:

| | Count | What they are |
| --- | --- | --- |
| skops artifacts | **10** | Every bundled scikit-learn model |
| `.pickle` artifacts | **20** | Pure data — no models |

PR #31 changed **zero** artifact files. The counts are identical on both trees.

### 3.1 Where the wrong number came from

Two independent mistakes compounded:

1. **`0 skops` was a globbing error.** The count used `\.skops$`, which matches
   the nine `.skops` files but misses the tenth artifact,
   `definition_model_layered.skops.**zip**`. All ten sklearn models are
   migrated; the README's "10" is correct and my "9" was not.
2. **`30 pickle` conflated categories.** 30 is the total artifact count
   (20 + 10), relabelled as pickles.

So the honest comparison is not "30 pickle vs 10 skops" but **10 skops vs 10
skops** — parity on the security-relevant artifacts — plus 20 data files that
Spencer's branch inherits from the same upstream ancestor.

### 3.2 Are the remaining 20 pickles a problem?

They were examined with `pickletools.genops`, which walks the opcode stream
without executing it. None of them contains a class or reduction gadget:

| Opcode profile | Files |
| --- | --- |
| No `GLOBAL` / `STACK_GLOBAL` opcodes at all | 17 |
| `builtins.set` only | 2 — `stopwords`, `city_name_words` |
| No globals (OCR reference vectors) | 2 |

They are frozen lookup tables: n-gram collocations, Unicode category maps,
stopwords, city-name words, OCR reference vectors. Nothing that `skops` would
serialise better, since skops targets estimators.

**Assessment: acceptable, with one caveat worth stating plainly.**
`pickle.load` on a tampered file is arbitrary code execution regardless of what
the original file contained. These 20 are version-controlled rather than
downloaded, so the trust boundary is the repository itself. The migration that
mattered — the ten estimators, which *were* downloadable and *did* carry
sklearn version coupling — is complete.

The cheap hardening available, if we want it, is to replace the data pickles
with a non-executable format (`.npz`, Parquet, or plain JSON). That is a
mechanical change with no security urgency behind it, so it is recorded as a
recommendation, not done here.

---

## 4. Defects found during the audit

None of these were introduced by PR #31 — all three reproduce on `master` —
but the audit is what surfaced them, and leaving them unfixed would have meant
shipping known-wrong behaviour.

### 4.1 Portuguese was unreachable through the locale dispatchers

`lexnlp.extract.pt` ships native money, percent, amount, duration and citation
extractors. The matching `all_locales` dispatchers registered only `en` and
`de`, so `"pt"` and `"pt-BR"` fell through to the **English** routine:

```
all_locales.money.get_money_annotations("pt-BR", "R$ 1.500.000,00")
  before:  (1500000.00, 'USD')     <- English routine, wrong currency
  after:   (1500000.00, 'BRL')     <- native pt routine
```

Brazilian reais reported as US dollars, in a library whose job is extracting
monetary amounts from contracts. Portuguese durations (`no prazo de 30 dias`)
returned nothing at all before the fix.

Fixed in `d11fcdb`. The dispatchers now resolve the **effective** language
before branching on calling convention — selecting the routine with a fallback
while branching on the *requested* locale is precisely how the date dispatcher
began raising `TypeError` on `fr`/`it`/`nl`, and that trap is now closed by
construction rather than by care.

### 4.2 Courts starting with their own keyword were silently dropped

`UniversalCourtsParser.parse` guarded on the configured keyword with:

```python
self.phrase_match_pattern.search(text, re.IGNORECASE)
```

The second positional parameter of `re.Pattern.search` is `pos`, not `flags`.
`re.IGNORECASE` is `2`, so every scan began at **offset 2** and discarded any
court whose keyword started at index 0 or 1.

Spanish court extraction was effectively dead: `es_courts.csv` holds 17
Tribunales Superiores de Justicia and every single one begins with the word
`Tribunal`. Portuguese lost `Tribunal Superior do Trabalho` for the same
reason while `Supremo Tribunal Federal` (keyword at index 8) worked — which is
exactly the kind of partial success that hides a bug.

Fixed in `daaeb8e`. Case-insensitivity was never coming from that argument; it
comes from how each caller compiles `court_pattern_checker`.

### 4.3 Spanish language tokens contain German words

`EsLanguageTokens.conjunctions` is `["und", "oder"]`. Those are German.

This one is **not** fixed here, deliberately, because the fix is coupled:
`es/courts.py` unions `conjunctions` into `line_breaks`, and `line_breaks` is
matched character-by-character. The German words are harmless there precisely
because they are multi-character no-ops. Substituting the real Spanish
conjunctions (`y`, `e`, `o`, `u`) would inject single characters into the
split set and shatter every court name containing an `e`.

`pt/language_tokens.py` already carries a comment describing this exact trap.
The correct repair is both halves at once — fix the word list *and* stop
unioning it into `line_breaks` — and it belongs with the Spanish work
described in the language parity document rather than bolted onto this audit.

---

## 5. Verification

- Differential probe re-run after every fix: still **zero** lost surface, still
  only the `pt.ratios` improvement differing from `master`.
- `lexnlp/extract`: **2,425 passed**, 3 skipped (Stanford NER, disabled by
  configuration), 0 failed.
- `ruff check` and `ruff format` clean on every file touched.

## 6. Recommendations

1. Statement coverage did not catch §4.1 or §4.2. Both are *silently wrong
   answers* on inputs no test supplied, and 100% line coverage is structurally
   incapable of finding that class of bug. Property-based tests over locale ×
   input-shape would.
2. Keep the differential probe. It is ~120 lines, it runs in minutes, and it
   is the only instrument here that compared behaviour rather than intent.
3. Consider converting the 20 data pickles to a non-executable format (§3.2).
