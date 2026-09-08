# PR #30 "Modernise/sota segmentation v1" — triage and disposition

**Verdict: do not merge the branch. Cherry-pick from it.**

Merging PR #30 as-is would delete 99,221 lines of work that is already on
master, including 172 files the branch has never seen: the async batch
extraction package, the Portuguese locale support, the pre-commit config, and
most of the test suites added since the merge base.

| Measure | Value |
| --- | --- |
| Files the PR itself changes (three-dot) | 251 |
| Files differing from master (two-dot) | 638 |
| Lines master would lose on merge | 99,221 |
| Files master would lose entirely | 172 |

## How the review was done

Every one of the 251 changed files was reviewed. The diffs were split into 33
batches of at most 10 files, budgeted by token count, and dispatched to two
external agents running headless in tmux: `muse exec` and `grok`. Each wrote
its findings to its own markdown file. A second `muse` worker took `grok`'s
queue from the far end to halve wall-clock time.

| Metric | Value |
| --- | --- |
| Unique files reviewed | 251 |
| Mean confidence | 86.7 |
| Findings below 70 confidence | 9 |
| Verdicts: merge / merge with edits / reject | 103 / 96 / 52 |

Both agents independently caught a flaw in the dispatch: the diffs were
generated three-dot, so the `-` lines were the merge base rather than current
master. The remaining prompts were corrected mid-run. All nine low-confidence
findings were then re-checked by hand, and that is where the most valuable
results came from.

## Accepted

**Lossless segmentation feature.** `backends.py`, `chunks.py`, `hierarchy.py`
and `payloads.py`, with their seven test files and four fixtures. Entirely
additive and self-contained; 143 tests and 281 subtests pass against master
unchanged. The benchmark and quality-gate scripts came with them.

**Expanded distribution checker.** `ci/check_dist_contents.py` gains
sdist/wheel/source parity checking, duplicate-member and forbidden-member
detection.

**Trusted asset manifest for model downloads.** Downloads previously trusted an
optional Content-MD5 response header supplied by the same server as the bytes.
They are now checked against reviewed SHA256 digests shipped in the package,
with HTTPS-only URLs, host matching, catalog tags validated as safe relative
paths, atomic install, and 28 security tests.

**Three CI jobs**: a documentation build with warnings fatal, a pip-audit
supply-chain job, and the segmentation quality and performance gate.

**Two regression tests**, taken verbatim, that turned out to document real
defects in master rather than in the branch.

## Accepted with modifications

**Clause depth.** The branch derives clause depth from decomposing the
numbering string, `len(_number_parts(label)) - 2`, which makes `1.2.3` deeper
than `4.` no matter where the drafter put them. Depth is a property of the
actual nesting. Clause depth now comes from the marker's indentation, as list
items already did.

**Runtime resource suffixes.** The dist checker's list named `.pickle` and
`.pickle.gzip` but not `.skops`, so the artifacts we actually ship were
excluded from the parity check.

**Condition and constraint extraction.** Their fix is taken in substance and
rewritten in full, because the branch's version drops the `post` field and the
`strict` filter along with the backtracking.

**Locale dispatch.** Their diagnosis is taken, their fix is not; see below.

**Download module.** Their file drops the retry session with backoff on 429 and
5xx that master added, so the security layer is merged onto master's file
rather than replacing it.

**nltk floor.** First rejected as an unnecessary forced upgrade, then reversed.
Adding the audit job showed nltk 3.9.4 carries 35 known vulnerabilities. The
branch was right.

## Rejected

**Python floor.** `requires-python` drops from `>=3.13,<3.15` to
`>=3.10,<3.14`, removing 3.14 and re-adding 3.10. Classifiers and CI matrices
follow it down, and development status regresses from Beta to Alpha.

**Build backend.** `uv_build` reverts to `setuptools==83.0.0` plus a
`setup.py`.

**Secure model serialization.** All five `.skops` segmenter artifacts are
replaced by `.pickle`, the `addresses_clf` skops loader is deleted, and
`skops>=0.11` is dropped from the dependencies. This is a security regression:
loading a pickle executes arbitrary code.

**Dependency floors.** Upper bounds and several raised floors are worth having,
but the same edit lowers numpy from `>=2.3` to `>=1.26.4`, and lowers
`cloudpickle`, `num2words`, `python-dateutil`, `reporters-db`, `tqdm`,
`Unidecode` and `zahlwort2num`, and pins `scikit-learn==1.7.2` in a library.

**Artifact ABI gate.** `lexnlp/ml/artifact_abi.py` and
`constraints/model-artifact-abi.txt` enshrine Python 3.12.13 and numpy 1.26.4
as the build gate. The quality manifests that go with them describe a model
artifact this branch does not ship.

**Typing and formatting churn.** 42 files replace `collections.abc` generics
and PEP 604 unions with `typing.List`, `Dict`, `Tuple`, `Union`, `Optional`,
and flip double quotes to single. No functional content.

## Defects found in master while triaging

These were on master, not in the PR, and are fixed on this branch.

1. **Catastrophic backtracking in condition and constraint extraction.** Both
   wrapped their trigger alternation in wildcard groups, so trigger-free text
   backtracked quadratically. 10,000 characters took 2.1 seconds in each. Now
   0.0006 seconds and linear. This is a denial-of-service exposure on
   attacker-supplied document text.

2. **The multi-locale date dispatcher was unusable outside English.** It passed
   five positional arguments to parsers that accept three, so German, Spanish
   and Portuguese raised `TypeError` on every call.

3. **German amounts silently lost precision.** English and German routines take
   their parameters in different orders, and positional dispatch handed German
   `extended_sources=True` as `float_digits`, rounding to one decimal place
   instead of four.

4. **The layered definition detector could not load its own artifact.** The
   asset pipeline ships `definition_model_layered.skops.zip` while the loader
   and both tests still pointed at the retired `pickle.gzip`, so the test
   module failed at collection.

5. **A clock-time guard was evaded by backtracking.** In Portuguese ratios,
   "10:30 a.m." was extracted as the ratio 10/3, because the engine shortened
   the right operand to move "a.m." out of the lookahead's reach.

6. **Four tests called a function that had been renamed**, and a five-test
   class patched a symbol the code no longer uses, so three of its tests passed
   while exercising nothing.

7. **Court citations recorded no language** when the caller did not name one.

8. **35 known vulnerabilities in nltk 3.9.4**, a hard runtime dependency, plus
   three more across click, pygments and soupsieve. Found by adding the audit
   job, not by reading the diff. nltk 3.10.3 clears all but one, which has no
   upstream fix and affects APIs LexNLP does not call.

9. **The test suite wrote debug output into the tracked test_data tree**, so
   every run left the working copy dirty with output nothing asserts on.

10. **The docs build emitted 181 warnings**, and neither sphinx-rtd-theme nor
    pip-audit was declared anywhere the lock could reach.

## One correction worth recording

The paragraph feature window clamps its forward edge by subtracting the window
size rather than the line position, which is unrelated to how much room is left
in the document. PR #30 carries a fix for half of it. I took that fix, extended
it to the other half, and it was wrong to ship: the bundled paragraph segmenter
was trained on vectors produced by the old arithmetic, so widening the window at
inference made it stop splitting on blank lines. The segmentation parity suite
caught it; the unit tests I had run did not. The source is restored and the
behaviour is now pinned by characterisation tests that say plainly which parts
are wrong and why correcting them requires retraining the model in the same
change.

## Tests and coverage

The suite was failing on master before any of this work, and coverage was 89%.

| | Before | After |
| --- | --- | --- |
| Tests passing | 1,798 | 3,871 |
| Tests failing | 6 | 0 |
| Statement coverage | 89.0% | 100.00% |
| Uncovered statements | 3,359 | 0 |
| Documentation build warnings | 181 | 0, and `-W` passes |
| Known dependency vulnerabilities | 38 | 0, one ignored with a recorded reason |

`fail_under = 100` is now configured, so the suite fails if coverage drops.
The measured surface is `lexnlp`, `scripts` and `ci`, with test files omitted.
`__main__` guards, `TYPE_CHECKING` blocks and abstract declarations are excluded
by the conventional coverage settings.

Tests were written by muse and grok across four rounds: 42 batches over the 168
modules that had gaps, 8 more over the 31 still short after that, 7 aimed at the
last 33 statements one line at a time, and 2 on the CI tooling that the
configuration newly brought into scope. Each agent was given the exact line
numbers its suite was not reaching. Every generated test was run before being
committed, and the ones that failed or asserted nothing were rewritten by hand.

Reaching the last few statements turned up two more defects. The section
segmenter's `train_logistic_regression` asked the lbfgs solver for an l1
penalty, which lbfgs does not support, so it raised on every call and could
never return a model; the four statements after the fit were unreachable for
that reason. And `ci/skip_audit.py` and `ci/check_dist_contents.py`, both gates
that decide whether a release proceeds, had 0% and 36% coverage.

Twenty-one statements carry `# pragma: no cover`, each with its reason on the
line above and each verified rather than asserted: `"&"` is in
`string.punctuation` so the strip above always removes it; `finditer` yields
distinct non-overlapping spans so a duplicate cannot appear;
`str.splitlines(keepends=True)` round-trips, checked over adversarial input
including lone surrogates and U+2028; the sentence splitter can never fullmatch
a bare `"and"`, checked with a 200,000-case fuzz; a container-stack pop is dead,
checked with a 400-case probe under `sys.settrace`. Eight are defensive
fallbacks in the chunk planner, including a strict-progress invariant that
raises if planning ever fails to advance. Those guards belong in the code and
can only be reached by breaking the unit under test.

An intermediate reading of 97% was wrong: a `.coverage` file left behind by a
run killed part-way had been merged into it. Deleting the stale file and
re-measuring gave the real figures.
