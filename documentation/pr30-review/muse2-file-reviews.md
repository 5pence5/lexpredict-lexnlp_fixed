### uv.lock  (batch 32, part 1/2; pasted chunk part 11/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk mostly bumps pins forward relative to master HEAD (ec55727): scipy 1.17.0 -> 1.17.1/1.18.0, setuptools 82.0.0 -> 83.0.0, smart-open 7.5.0 -> 8.0.1, snowballstemmer 3.0.1 -> 3.1.1, soupsieve 2.8.3 -> 2.9.1, plus new sniffio 1.3.1 and sortedcontainers 2.4.0 entries. However the surrounding lock structure is regressed: the file header reverts requires-python from ">=3.13,<3.15" (master) to ">=3.10,<3.14" (PR) with per-Python resolution-markers (3.11.*, <3.11, >=3.12 x win32/emscripten splits) that master deliberately removed. Do not cherry-pick lock hunks; regenerate with `uv lock` from the corrected pyproject instead.
- **Notes:** Verified via `git show ec55727:uv.lock` (scipy 1.17.0 single entry; setuptools 82.0.0; smart-open 7.5.0; no sniffio/sortedcontainers) vs `git show pr30:uv.lock` (scipy split 1.15.3/1.17.1/1.18.0; setuptools 83.0.0; smart-open 8.0.1). PR lock also drops skops 0.13.0 present on master (see batch header check). Newer pins here are genuine upstream releases (timestamps Jun-Jul 2026) but are inseparable from the regressed Python floor in this file.

### uv.lock  (batch 32, part 2/2; pasted chunk part 12/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk continues the same pattern: forward pin bumps (tomli 2.4.0 -> 2.4.1, tomlkit 0.14.0 -> 0.15.1, tqdm 4.67.3 -> 4.70.0) plus newly listed docs/build entries (sphinx-rtd-theme 3.1.0, sphinxcontrib-jquery 4.1, tomli-w 1.2.0) absent from master HEAD. The sphinx dependency-marker rewrites (dropping `python_full_version` qualifiers in favour of split win32/emscripten markers) encode the regressed >=3.10 Python range, and master pyproject declares `skops>=0.11` while the PR pyproject/lock drops it. Same disposition as part 1/2: reject the PR lockfile wholesale and re-lock on master.
- **Notes:** Verified: master lock has no sphinx-rtd-theme, sphinxcontrib-jquery, tomli, tomli-w, sniffio, or sortedcontainers packages; PR lock adds all of them. Master pyproject has `scipy>=1.11.0`, `skops>=0.11`, `tqdm>=4.67`; PR pyproject has `scipy>=1.13.0,<2`, no skops, `tqdm>=4.64.1,<5` (lowered floor) plus `sphinx-rtd-theme>=3,<4`. True diff is `git diff ec55727..pr30 -- uv.lock` (1575+/1381-, not the pasted merge-base diff).
### uv.lock  (batch 30, part 1/2; pasted chunk part 7/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk replaces numpy 1.26.4 (merge-base) with a three-way split — numpy 2.2.6 (python_full_version < '3.11'), 2.4.6 (== '3.11.*'), 2.5.1 (>= '3.12') — plus packaging 26.0 -> 26.2 and a new packageurl-python 0.17.6 entry. Against master HEAD (single numpy 2.4.4, packaging 26.0, no packageurl-python, requires-python ">=3.13,<3.15") the 2.5.1/26.2 pins are forward but inseparable from the regressed header requires-python ">=3.10,<3.14" and per-Python resolution-markers. Do not cherry-pick lock hunks; regenerate with `uv lock` from the corrected pyproject.
- **Notes:** Verified via `git show ec55727:uv.lock` (numpy 2.4.4 single entry, Mar-2026 wheels, cp313-only) vs `git show pr30:uv.lock` (numpy 2.2.6/2.4.6/2.5.1 split with markers). True diff is `git diff ec55727..pr30 -- uv.lock` (1575+/1381-). PR pyproject drops `skops>=0.11` and adds `pip-audit>=2.9.0,<3`, so packageurl-python enters only via that audit closure.

### uv.lock  (batch 30, part 2/2; pasted chunk part 8/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk splits pandas 2.3.3 (master: single entry) into pandas 2.3.3 (<3.11) + 3.0.5 (>=3.11/>=3.12 with numpy 2.4.6/2.5.1 pins), bumps platformdirs 4.7.1 -> 4.11.0, psutil 6.1.1 -> 7.2.2, pycountry 24.6.1 -> 26.2.16, pygments 2.19.2 -> 2.20.0, and adds a new pip-audit closure (pip 26.1.2, pip-api 0.0.34, pip-audit 2.10.1, pip-requirements-parser 32.0.1, py-serializable 2.1.0). The version bumps are genuine forward releases (2026 timestamps) but are embedded in the regressed >=3.10 lock structure, and the audit closure is absent on master. Same disposition as part 1/2: reject the PR lockfile wholesale and re-lock on master.
- **Notes:** Verified: master lock has 1x pandas, no pip/pip-audit/packageurl-python/py-serializable, and retains skops 0.13.0; PR lock has 2x pandas/pandas 3.0.5 and the full pip-audit closure because PR pyproject adds `pip-audit>=2.9.0,<3` while dropping `skops>=0.11`. Pasted `-` lines are merge-base (pandas 2.3.3, psutil 6.1.1), not master — judged against `ec55727` per header correction.
### uv.lock  (batch 28, part 1/2; pasted chunk part 3/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk bumps several pins forward versus master HEAD (cryptography 46.0.7 -> 49.0.0, dateparser 1.2.1 -> 1.4.1, elastic-transport 8.17.1 -> 9.4.2 with new sniffio dep, elasticsearch 8.19.3 -> 9.4.1 with new anyio/sniffio deps, gensim 4.4.0 kept but split to per-Python numpy/scipy pins plus cp313 wheels). But it is embedded in the regressed lock structure (requires-python ">=3.10,<3.14" plus win32/emscripten resolution-markers, e.g. docutils marker split) and adds back deps master removed (cyclonedx-python-lib 11.11.0, defusedxml 0.7.1). Do not cherry-pick; regenerate with `uv lock` from the corrected pyproject.
- **Notes:** Verified via `git show ec55727:uv.lock` (cryptography 46.0.7; dateparser 1.2.1; elastic-transport 8.17.1; elasticsearch 8.19.3; no cyclonedx/defusedxml) vs `git show pr30:uv.lock` (cryptography 49.0.0 2026-06-12; dateparser 1.4.1; elastic-transport 9.4.2; elasticsearch 9.4.1; cyclonedx 11.11.0; defusedxml 0.7.1). Pasted `-` lines are merge-base (cryptography 46.0.5, dateparser 1.1.3), not master. True diff is `git diff ec55727..pr30 -- uv.lock` (1575+/1381-). PR pyproject drops `skops>=0.11` while master lock retains skops 0.13.0.

### uv.lock  (batch 28, part 2/2; pasted chunk part 4/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This hunk continues forward pin bumps versus master HEAD (idna 3.15 -> 3.18, imagesize 1.4.1 -> 2.0.0, isort 7.0.0 -> 8.0.1, jaraco-context 6.1.0 -> 6.1.2, jaraco-functools 4.4.0 -> 4.6.0, jellyfish 0.6.1 -> 1.2.1 with ~60 new wheels, filelock 3.29.0 -> 3.32.0, lexnlp 2.3.0 -> 2.4.0a1 with numpy 2.2.6/2.4.6/2.5.1 and pandas 2.3.3/3.0.5 splits plus threadpoolctl and audit extra). But it re-adds importlib-metadata 9.0.0 / exceptiongroup closure that master (requires-python ">=3.13,<3.15") deliberately dropped, and encodes the regressed >=3.10 range. Same disposition: reject the PR lockfile wholesale and re-lock on master.
- **Notes:** Verified: master lock has single numpy 2.4.4 / pandas 2.3.3 / scipy 1.17.0, jellyfish 0.6.1, no importlib-metadata/exceptiongroup/defusedxml/cyclonedx/sniffio/sortedcontainers, and retains skops 0.13.0; PR lock splits numpy/scipy/pandas per-Python and adds those packages. Master pyproject has `skops>=0.11`, `numpy>=2.3,<3`; PR pyproject has no skops, `numpy>=1.26.4,<3`, `pip-audit>=2.9.0,<3`. Pasted `-` lines (idna 3.11, jellyfish 0.6.1 sdist-only) are merge-base, not master.

### test_data/lexnlp/nlp/en/sota_segmentation/retrieval_gold.json  (batch 26, part 1/8)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 75
- **Summary:** New 61-line synthetic fixture (2 docs, 6 queries with character-span answer anchors) for the PR's segmentation retrieval plumbing. Verified absent on master (`git show ec55727:<path>` = nonexistent), so nothing is overwritten. Anchors are exact substrings of the doc texts by inspection, and the file honestly disclaims SOTA value ("regression/plumbing only; not SOTA evidence").
- **Notes:** Useless alone — cherry-pick together with its consumers (`lexnlp/nlp/en/tests/test_sota_retrieval_quality.py`, `scripts/segmentation_quality_gate.py`). Odd `marker` value `QUALITY_BLOB_RECONSTRUCTION_V3` is cosmetic only.

### test_data/lexnlp/typed_annotations/de/law/laws.txt  (batch 26, part 2/8)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Tightens 4 coords to exact entity spans: (8,39)->(9,38), (18,24)->(19,23), (8,31)->(9,30), (563,576)->(564,575). Verified with Python that the PR spans are exactly right (e.g. `text[9:38]=='Allgemeine\nGebührenverordnung'`, `text[564:575]=='Deutschland'`) while master's spans include padding spaces (e.g. `text[563:576]==' Deutschland '`) left over from the old `(?:^|\W)({})(?:\W|$)` pattern that consumed boundary chars.
- **Notes:** Coherent only with the companion code change (`lexnlp/utils/parse_df.py` SEARCH_PTN -> `(?<!\w)({})(?!\w)` lookarounds). Do NOT apply this fixture alone on master or typed-annotation tests will fail; cherry-pick fixture + `parse_df.py` together.

### test_data/lexnlp/typed_annotations/de/percent/percents.txt  (batch 26, part 3/8)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Fixes `fraction` 20->0.2, 15->0.15, 18->0.18, aligning the DE fixture with the EN convention (master EN fixture uses `fraction=0.3` for 30%). This matches a genuine one-line bug fix in the PR's `lexnlp/extract/de/percents.py`: `round(amount, ...)` -> `round(real_amount, ...)` (master computed `0.01*amount` then overwrote it with `round(amount)`).
- **Notes:** Merge fixture only together with that one-line fix. Exclude the same hunk's cosmetic downgrades (`List`/`Dict` typing imports, single-quote restyling vs master's modern `list[dict]`).

### test_data/lexnlp/typed_annotations/en/citation/citations.txt  (batch 26, part 4/8)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Single coords fix (19,50)->(20,49) for `1 F.2d 1, 2-5 (2d Cir., 1982)`. Verified `text[20:49]` is exactly the citation source while master's `text[19:50]` grabs a leading space and drops the closing paren. Matches the PR's `lexnlp/extract/en/citations.py` change from `match.span()` to `match.span(1)` (group 1 excludes the consumed leading boundary).
- **Notes:** Same cherry-pick caveat as the laws fixture: fixture + `citations.py` span change must land together, never the fixture alone.

### test_data/model_quality/bundled_model_reexport.json  (batch 26, part 5/8)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** New file pinning 9 bundled artifacts as `.pickle` paths with a py3.12.13 / sklearn 1.7.2 runtime. Verified absent on master. It is the manifest side of the PR's pickle-only reexport (`scripts/reexport_bundled_sklearn_models.py` drops `--format skops`), which deletes master's `.skops` artifacts and the `skops>=0.11` dependency — the known SECURITY REGRESSION. Adding this file blesses the insecure artifacts.
- **Notes:** Strategy string "Scoped offline reconstruction of legacy sklearn tree node state" does not change the verdict: the artifact paths (e.g. `lexnlp/extract/de/date_model.pickle`) are the legacy insecure format master moved away from.

### test_data/model_quality/contract_type_baseline_metrics.json  (batch 26, part 6/8)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 55
- **Summary:** Adds genuinely useful provenance (schema_version 2, fixture_sha256, artifact sha/size, training recipe, duplicate-group holdout reporting honest lower holdout metrics: top1 0.748 vs 0.933 on the fixture) but raises the gate thresholds themselves (accuracy_top1 0.8666->0.9333, f1_macro 0.7->0.8, f1_weighted 0.91->0.96) to match the PR's own retrained `pipeline/contract-type/0.2-runtime` model built on a stale py3.12.13 runtime, which defeats the quality gate's purpose.
- **Notes:** Keep the provenance/holdout fields only after re-running `scripts/contract_type_quality_gate.py` + training on the master (3.13) runtime and confirming the numbers; do not accept the new thresholds blind. Low confidence reflects inability to validate the retrained model offline.

### test_data/model_quality/release_asset_manifest.json  (batch 26, part 7/8)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 60
- **Summary:** Benign additive change: `schema_version: 1` plus two entries, `pipeline/is-contract/0.2` and `pipeline/contract-type/0.2-runtime` (sha `3b04a8a9...` consistent with the contract-type baseline artifact in this same PR). No existing entries are altered.
- **Notes:** Verify both tags/files actually exist in the `LexPredict/lexpredict-lexnlp` releases repo before accepting (shas/sizes unverifiable offline). Flag: `pipeline/is-contract/0.2` has no matching `is_contract_baseline_metrics.json` update in this PR (that file is untouched), so either add the gate update or hold the manifest entry.

### uv.lock  (batch 26, part 8/8)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** True diff `ec55727..pr30` confirms the known regression: `requires-python` `>=3.13,<3.15` -> `>=3.10,<3.14` with expanded per-platform resolution markers, and removal of our modern deps (e.g. `annotated-doc 0.0.4`, the typer 0.24.2 stack present in master's lock). Transitive bumps in this hunk (beautifulsoup4 4.14.3->4.15.0, anyio 4.13.0->4.14.2, cffi 2.0.0->2.1.0, certifi 2026.1.4->2026.7.22, build 1.4.0->1.5.0; new boolean-py/cachecontrol) are a stale re-resolve, not our pins — consistent with the prior batch verdict on this file.
- **Notes:** Reject the PR lockfile wholesale (all 13 parts); re-lock from master's `pyproject.toml` instead. Pasted `-` lines are merge-base, not master — verified pins above via `git show ec55727:uv.lock`.

### scripts/model_quality_gate.py  (batch 24, part 1/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 92
- **Summary:** True diff (`ec55727..pr30`) shows the PR downgrades the default contract-model tag from `pipeline/is-contract/0.2` back to `pipeline/is-contract/0.1`, undoing our post-merge-base upgrade (merge base 51c07bc had 0.1, master moved to 0.2). It also swaps `from cloudpickle import load` for `lexnlp.utils.unpickler.load_sklearn_model`, reverts our PEP 585/604 typing modernisation (`list[str]`→`List[str]`, `collections.abc.Sequence`→`typing.Sequence`, commit 6ddc0de), and deletes the `score_pipeline` docstring added in 17a7b87. Nothing in the file is an improvement over master.
- **Notes:** Loader swap is also a functional risk: master deliberately loads catalog models with cloudpickle; stdlib-based `load_sklearn_model` may not read those bytes. Pasted three-dot diff is misleading here only cosmetically; true diff confirms all four regressions.

### scripts/reexport_bundled_sklearn_models.py  (batch 24, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** Master re-exports bundled sklearn artifacts via secure `.skops` (`--format skops` default, `lexnlp.ml.model_io.dump_model/load_model`). The PR version contains zero mentions of skops and reverts to pickle/joblib-only re-export, dropping the `skops>=0.11` direction entirely — the known security regression. It also imports three PR-only modules (`lexnlp.ml.artifact_abi`, `lexnlp.ml.artifact_io`, `scripts._artifact_transaction`), none of which exist on master, so the file cannot even be cherry-picked standalone.
- **Notes:** The `+ Path("lexnlp/extract/en/date_model.pickle")` line looks additive in the pasted diff but already exists on master (line 35) — no-op, not a gain. `--check-current`/`--metadata-output`/atomic-write ideas are nice but would need reimplementation on top of `model_io`, not this file.

### scripts/reexport_contract_model.py  (batch 24, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 80
- **Summary:** Same loader swap as the quality gate: master's `from cloudpickle import load` becomes `load_sklearn_model`, against master's serialisation path (cloudpickle load → stdlib pickle dump). The rewrite additionally depends on PR-only modules (`lexnlp.ml.artifact_io.atomic_pickle_dump`, `lexnlp.ml.artifact_abi`, `scripts._artifact_transaction.publish_staged_files`), so it is unmergeable without the rest of the PR's artifact stack. The staged-candidate plus isolated-quality-gate-catalog flow is a genuine improvement in principle, but it is inseparable from the loader change and the missing modules.
- **Notes:** Watch the `isolated_quality_gate_catalog` helper mutating `NLTK_DATA` env during the gate — side-effectful even on success paths. If the staging idea is wanted, port it onto master's cloudpickle-based flow instead.

### scripts/run_tika.sh  (batch 24, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Master never touched this file (identical to merge base: Tika 1.16, `tika-server-$VERSION.jar`, fragile `ps aux | grep -c` liveness check). The PR is a coherent modernisation: Tika 3.3.2, correct 2.x+ artifact name `tika-server-standard-$version.jar`, `--host/--port` flags, `set -euo pipefail`, HTTP `/version` readiness probe and port-in-use guard. This undoes none of our work.
- **Notes:** Do NOT merge standalone: it demands `tika-server-standard-3.3.2.jar`, which only the PR's rewritten `bootstrap_assets.py`/`download_tika.sh` (other batches) download — master's bootstrap still fetches `tika-server-1.16.jar`. Co-merge with those batches only; the 3.3.2 SHA-512 pins themselves could not be verified offline.

### scripts/segmentation_benchmark.py  (batch 24, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 65
- **Summary:** Purely additive (454 lines, new file, absent on master); it removes nothing. It provides hermetic throughput/memory/determinism gates over the PR's new `segments.chunks`/`segments.hierarchy` modules with no model, corpus, or network dependency. As regression review there is nothing reverted or downgraded.
- **Notes:** Inert without its companion PR-only modules (`lexnlp/nlp/en/segments/chunks.py`, `hierarchy.py`, `lexnlp/nlp/en/tests/segmentation_quality.py` — none on master). Cherry-pick only together with the segmentation feature batch; the `# QUALITY_BLOB_RECONSTRUCTION_V3` header and fixed throughput floors (10k chars/s, 64 MiB) deserve a flakiness look before CI-wiring, but that is outside regression scope.

### scripts/segmentation_quality_gate.py  (batch 24, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 65
- **Summary:** Same shape as part 5/10: brand-new 644-line file, no master code removed or altered. It implements the deterministic segmentation/retrieval regression gate (anchor recall, structural/boundary exact-span PRF, lexical retrieval metrics, external-manifest evaluation) over the PR's new segmentation modules. No downgrade, no deprecated API, no dependency change.
- **Notes:** Same co-dependency as 5/10 — imports exist only on the PR branch, so it only functions alongside the segmentation modules and `sota_segmentation` fixtures. Default thresholds demand 1.0 anchor/structural recall on the bundled fixture; fine for a hermetic gate, not population SOTA evidence (the file's own docstring admits this).

### scripts/tests/test_asset_drift_check.py  (batch 24, part 7/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 93
- **Summary:** Master has a 271-line test file pinning OUR fixed behaviors; the PR replaces it wholesale (136 lines) and deletes every one of those tests. This directly masks two real regressions in the PR's `asset_drift_check.py` (verified in true diff): error format reverted from `({exc.__class__.__name__}: {exc})` to `({exc})`, and SHA comparison reverted from `actual_sha != expected_sha` to the buggy asymmetric `actual_sha.lower() != expected_sha` — both behaviors master's deleted tests explicitly cover. The two replacement tests only exercise the PR's new `download_github_release_to_path` wiring.
- **Notes:** Also reverts typing to `typing.Dict/Iterable` and swaps catalog APIs to PR-only `get_exact_path_from_catalog`/`download_github_release_to_path`. PR also deletes `scripts/tests/__init__.py`, which changes test-package semantics for the survivors.

### scripts/tests/test_bootstrap_assets.py  (batch 24, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 78
- **Summary:** Master has 325 lines testing our behaviors (URL scheme validation, zip-slip protection, `run_selected_tasks` lambda-closure tag fix); the PR substitutes 666 lines written against its own 1056-line `bootstrap_assets.py` rewrite, deleting all of ours. The new tests are individually good (SHA-512-before-install, digest-mismatch rejection, size limits, traversal/symlink rejection, NLTK cache repair, dry-run purity) and even harden policy (HTTPS-only — master's tests assert `http://` proceeds), but the closure-fix coverage has no counterpart (PR test file has zero matches for closure/`contract_type` behavior) and the file cannot run against master's script (changed `download_file` signature with `expected_sha512` etc.).
- **Notes:** Salvage path: keep master's file and port over the genuinely new pin/digest/cache assertions once the `bootstrap_assets.py` rewrite itself is judged (separate batch). Do not accept as a replacement.

### scripts/tests/test_check_dist_contents.py  (batch 24, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 30-line test file; deletes nothing. It tests `normalise_package_name` and `is_forbidden_wheel_member`, which do not exist in master's `ci/check_dist_contents.py` but do exist in the PR's extended version (defs at lines 100/117, part of its 318-line packaging-parity rewrite). Assertions are sane: nested `test_data/...` paths normalise to None, real package members pass through, every `tests/` path component is wheel-forbidden.
- **Notes:** Must land together with the PR's `ci/check_dist_contents.py` change (other batch) — it fails to import without it. Policy call (banning all `tests/` dirs and `scripts/`/`ci/` from wheels) belongs to that batch's review, not this file's.

### scripts/tests/test_create_release_branch.py  (batch 24, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 102-line isolated test (temp-dir fixture repo, stub `uv` on PATH) for the PR's rewritten `create_release_branch.sh`. Master's script is an obsolete interactive helper referencing a `-core` sibling repo, `setup.py`, and `documentation/docs/source/conf.py`; the PR rewrite (single version arg, `release/X.Y.Z` branch, canonical `pyproject.toml` + `__init__.py` version bump, `uv lock` verification) is a strict improvement and this test pins exactly that behavior including dirty-worktree and version-disagreement rejections.
- **Notes:** Must land together with the rewritten `.sh` (other batch) — it fails against master's script by design. Not executed here (working tree carries the master-side script plus unrelated local modifications); verdict rests on line-level consistency between the test and the PR's script.

### lexnlp/utils/tests/test_amount_delimiting.py  (batch 22, part 1/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 95
- **Summary:** The pasted diff looks like a new 21-line file, but against master HEAD (ec55727) the true effect is a deletion: master already has a 393-line suite (`TestDeDEGroupingFix`, `TestEnUSLocaleFix`, `TestGenericEmptyGroupingFallback` plus regression paths) covering the de_DE grouping check, en_US fallback, and generic `elif not grouping` fallback that master's `amount_delimiting.py` still contains. The PR replaces all of that with 2 trivial tests. Keep master's file; cherry-pick nothing from this chunk.
- **Notes:** True diff: `git diff ec55727..pr30 -- lexnlp/utils/tests/test_amount_delimiting.py` shows `-393/+21`. The 21-line version's `infer_delimiters('10.800','de_DE')` assertion is subsumed by master's mocked-convention tests.

### lexnlp/utils/tests/test_decorators.py  (batch 22, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** New 57-line test file with no master counterpart. It pins the PR's genuine `safe_failure` bug fixes in `lexnlp/utils/decorators.py`: scalar-vs-generator split via `isgeneratorfunction` (master calls `func` twice for generators: `res = func(...)` then `yield from func(...)`), `@wraps` preservation, narrowing bare `except:` to `except Exception`, and letting `KeyboardInterrupt`/`SystemExit` propagate. All four behaviors tested are strict improvements.
- **Notes:** Must land together with the `decorators.py` change (same PR, verified present on pr30). The `TypeError`-still-suppressed vs `KeyboardInterrupt`-re-raised distinction is the intended semantic change, not a regression.

### lexnlp/utils/tests/test_dist_contents.py  (batch 22, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** New 52-line test for `ci.check_dist_contents` helpers (`find_resource_failures`, `find_duplicate_members`, `find_duplicate_resources`, `find_violations`). Master's `ci/check_dist_contents.py` only has `find_violations`/`iter_tar_names`/`iter_zip_names`, so this test fails standalone on master by design — its value depends on the PR's `ci/check_dist_contents.py` expansion (verified present on pr30 with all four functions). The assertions themselves look correct and useful.
- **Notes:** Merge only together with the PR's `ci/check_dist_contents.py` expansion (different batch). Consider relocating the test next to the module it covers (`ci/` or top-level tests) instead of `lexnlp/utils/tests/`, which is an odd home for a `ci.*` import.

### lexnlp/utils/tests/test_parse_df.py  (batch 22, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Adds 3 substantive tests (adjacent-entity spans `Peppa,George`, longest-match-wins `Act`/`Act One`, per-cell single-split plus `_row_positions_by_column` enrichment) for the PR's `parse_df.py` fix that master lacks (master still has consuming-boundary `SEARCH_PTN = r"(?:^|\W)({})(?:\W|$)"` and no `_split_cell_value`). The tests encode genuinely better behavior. The chunk also carries pure quote-style churn (`"` to `'`, reformatted `sample_csv`, reindented `get_entries`) that should be dropped.
- **Notes:** True diff vs master is `-8/+92` in test plus the `parse_df.py` source change (`(?<!\w)({})(?!\w)`, `_split_cell_value`, `_row_positions_by_column`). Merge the 3 new tests plus the source fix; discard the quoting/reformatting noise. Low risk: new tests do not alter existing assertions.

### lexnlp/utils/tests/test_unpickler.py  (batch 22, part 5/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** New 120-line test that re-anchors the suite to the reverted artifact world: it loads `lexnlp/nlp/en/segments/page_segmenter.pickle` and `title_locator.pickle` and asserts exact legacy `predict`/`predict_proba` values (`[0.76, 0.24]`, `[0.92, 0.08]`). On master those paths do not exist — master ships `page_segmenter.skops` / `title_locator.skops` (verified via `git ls-tree ec55727`). It also tests `CompatibilityReport`/`load_joblib_model` APIs that exist only in the PR's `unpickler.py`, not master's thin `model_io`-delegating shim.
- **Notes:** Reject together with the PR's `unpickler.py` expansion (part 6/10). The subprocess no-global-patch test is the only salvageable idea, but it tests PR-internal architecture; do not cherry-pick the file.

### lexnlp/utils/unpickler.py  (batch 22, part 6/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** Replaces master's thin `RenameUnpickler` + `renamed_load` shim (which delegates tree-ABI and estimator patching to `lexnlp.ml.model_io._patched_sklearn_tree_loader` / `_patch_legacy_sklearn_estimator`, the skops-migration path) with a 352-line self-contained pickle/joblib compat engine (`CompatibilityReport`, `PatchedTree`, `_modernize_patched_trees`, `restore_legacy_model_state`, `load_joblib_model`, `renamed_load(..., report=None)`). Directionally this undoes the `.skops` migration (drops the `skops>=0.11` dependency per the pyproject chunk) and duplicates `model_io` logic (GaussianNB `sigma_`->`var_`->`variance_`, `base_estimator`->`estimator`, tree-value normalization) in a second home.
- **Notes:** Scoped-unpickler hygiene (per-load mixin instead of global `find_class` patching, `ContextVar` report) is the one good idea, but it belongs as a refactor inside `model_io`, not as a fork that re-centers `.pickle`. The `UnicodeDecodeError` -> `ValueError("Python 2 Joblib pickles are not supported...")` and `compat_mode` DeprecationWarning paths have no master counterpart and serve only legacy artifacts.

### libs/download_stanford_nlp.sh  (batch 22, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Replaces the obsolete direct-`wget`/`unzip` download block (5 hardcoded `nlp.stanford.edu` zips, unquoted vars, no `set -euo pipefail`, `[ "$LEXNLP_USE_STANFORD" = true ]` which errors when unset) with a 10-line wrapper delegating to `scripts/bootstrap_assets.py --stanford --stanford-dir "${STANFORD_NLP_PATH:-.../libs/stanford_nlp}"`. Master HEAD still carries the old wget version verbatim, so this is modernization consistent with AGENTS.md, not a regression.
- **Notes:** `[[ "${LEXNLP_USE_STANFORD:-false}" == "true" ]]` default-off semantics and `STANFORD_NLP_PATH` override are improvements. Merge as-is; confirm `scripts/bootstrap_assets.py --stanford` flag exists at cherry-pick time (it does on master per AGENTS.md).

### notes.md  (batch 22, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 80
- **Summary:** Deletes the 203-line (263 lines at master HEAD) `notes.md` modernization log that still exists on master (`git show ec55727:notes.md` succeeds: uv/pyproject migration, sklearn-compat, bootstrap assets, skip-audit, packaging audit, contract-type runtime notes). The pasted `-` lines are stale (merge-base paths like `/Users/jackeames/Downloads/LexNLP`, Python 3.11) but master has since refreshed the same file; wholesale deletion discards curated operational history.
- **Notes:** If the team wants this scratch log out of the repo, do it as a deliberate master-side decision, not as a side effect of this PR. No content from this deletion should be cherry-picked.

### pyproject.toml  (batch 22, part 9/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Textbook modernization rollback. True diff vs master (`4cf2f3a..902aacc`): build backend `uv_build>=0.9.0,<0.10.0` reverts to `setuptools==83.0.0` + `wheel==0.47.0`; `requires-python ">=3.13,<3.15"` widens down to `">=3.10,<3.14"` (confirmed regression per brief); `Development Status :: 4 - Beta` demotes to `3 - Alpha`; drops `skops>=0.11` (the secure-serialization dependency), `hub`/`arrow`/`ner` extras, PEP-735 `[dependency-groups]`, `[tool.uv.build-backend]` layout, and the entire `[tool.ruff]` section; pins `scikit-learn==1.7.2` over master's `>=1.5`. The handful of fine additions (`[project.urls]`, `docs`/`audit` extras, `tika>=3.1.0,<4`) do not offset this.
- **Notes:** Exact reverts to cite: `requires-python`, `skops`, `uv_build`, classifiers `3.13/3.14` -> `3.10/3.11/3.12/3.13`, `us>=3.2.0` vs master's `us>=2.0.2`, `regex>=2026.7.19` fantasy-pin. Cherry-pick nothing; if `[project.urls]` is wanted, add it directly on master.

### python-requirements-dev.txt  (batch 22, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Deletes the 38-line deprecated `python-requirements-dev.txt` legacy snapshot (ends with `zahlwort2num==0.3.0`, headed by the "DEPRECATED: use pyproject.toml + uv.lock" banner). The file does not exist on master HEAD (`git show ec55727:python-requirements-dev.txt` fails) — master already completed this deletion — so the true net effect of this chunk is a no-op, not a regression. Nothing to do.
- **Notes:** No action needed at cherry-pick time; already satisfied on master. Do not reintroduce the file.

### lexnlp/nlp/en/tests/test_pages.py  (batch 20, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 78
- **Summary:** Purely additive test coverage: all master tests retained (`test_page_examples` still present) plus 8 new tests (~219 lines) locking in fail-closed segmentation hardening — empty-input guard, full 244-column feature schema on short docs, huge-window early return, and incompatible-model-metadata handling. No master assertion was weakened or deleted, so this is not a content regression; but the new tests import `has_compatible_feature_width` / `resolve_model_feature_width`, which do not exist in master `utils.py`, and pin the 244-column width against the PR's pickle-loaded model while master loads `.skops` via `load_bundled_model`.
- **Notes:** True diff `git diff ec55727..pr30` = +219/-5; the -5 are a quote-style nit (`TEST_PATH` double→single) and mode change 100755→100644 (harmless normalization). Cherry-pick together with a skops-compatible rebase of the `utils.py` hardening, not standalone — as written the file ImportErrors on master.

### lexnlp/nlp/en/tests/test_paragraphs.py  (batch 20, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Same pattern as test_pages: every master test retained (all 9 `test_document_distribution_*`, `test_splitlines_with_spans`, `test_date_text`, `test_paragraph_examples` present) plus 12 valuable new tests — 361-column schema check, single-line span-start-at-zero fix, blank-line ownership overriding conflicting model breaks, and forest-metadata consistency checks. The additions test genuinely useful hardening, but they depend on PR-only `utils.py` symbols (`has_compatible_feature_width`, `has_compatible_line_window`, `resolve_model_feature_width`) absent on master, and the diff carries heavy double→single quote reformatting churn (336+/212-).
- **Notes:** `build_paragraph_break_features` / `get_paragraph_break_feature_names` / `build_document_line_distribution` exist on both sides, so only the compat-helper imports block a direct merge. Keep the new tests, drop the quote churn, and pair with the skops-based implementation. Context: PR's `paragraphs.py` also reverts modern hints (`tuple[...]` → `Tuple[List[...]]`) and `load_bundled_model` → `load_joblib_model` — do not take those hunks.

### lexnlp/nlp/en/tests/test_sections.py  (batch 20, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 76
- **Summary:** Additive-only change (`git diff ec55727..pr30` = +139/-23, deletions are quote-style): all 7 master tests retained, 6 new tests added covering the 369-column section schema, empty-input no-model-call, huge/underspecified-window early returns, and same-width custom-window relabeling guard. Good hardening coverage, but like the other two segmenter test files it imports PR-only compat helpers (`has_compatible_feature_width`, `has_compatible_line_window`, `resolve_model_feature_width`) that fail on master.
- **Notes:** The new tests reference `SectionSegmenterModel.SECTION_SEGMENTER_MODEL`, which exists on master (skops-backed) — the 369-width pin must be re-validated against the `.skops` artifact after rebase. Discard the quote-flip churn when cherry-picking.

### lexnlp/nlp/en/tests/test_sota_backends.py  (batch 20, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** Brand-new file (67 lines, missing on master) testing the new `segments/backends.py` SaT adapter with injected fake models: lossless `parts_to_spans` with mixed newlines/unicode, exact-identity enforcement, fail-closed on inexact output, and no-model-call on empty text. Fully hermetic, touches no master code, and asserts no dependency versions or model artifacts.
- **Notes:** Must be cherry-picked together with its implementation counterpart `lexnlp/nlp/en/segments/backends.py`; standalone it ImportErrors on master. No concerns with content.

### lexnlp/nlp/en/tests/test_sota_chunk_quality.py  (batch 20, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 321-line file testing the new `segments/chunks.py` + `hierarchy.py` chunking API using only a deterministic hermetic sentence segmenter — strict char/token budgets, overlap handling, container policies (PRESERVE vs PACK_SIBLINGS), chunk-identity hashing, and lossless invariants. Additive only (absent on master), no master behavior altered, no version pins or legacy-API references.
- **Notes:** Depends on `chunks.py`, `hierarchy.py`, and `tests/segmentation_quality.py` plus `test_data/.../sota_segmentation/` fixtures, all likewise PR-new — cherry-pick as a unit. The `QUALITY_BLOB_RECONSTRUCTION_V3` marker is PR-internal plumbing, not a regression.

### lexnlp/nlp/en/tests/test_sota_hierarchy_quality.py  (batch 20, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 220-line fixture-driven suite for the new `segment_document` hierarchy API: edge-case corpus losslessness/determinism, exact-boundary gold scoring that penalizes false positives and oversegmentation, heading-ancestry nesting checks, statute-vs-conservative profiles, and a 200-sample seeded unicode/newline metamorphic test. Hermetic and additive; nothing on master is modified or removed.
- **Notes:** Requires the PR-new `hierarchy.py`, `segmentation_quality.py`, and `boundary_gold.json` / `legal_edge_cases.json` fixtures — take those together. Asserts exact-span equality, so any future change to heading heuristics will surface here by design.

### lexnlp/nlp/en/tests/test_sota_legacy_parity.py  (batch 20, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 65
- **Summary:** New 54-line file pinning frozen legacy outputs (`legacy_parity.json` fixture) for `get_sentence_list` / `get_paragraph_list` and asserting the new hierarchy/chunk calls do not mutate the shared `SENTENCE_SEGMENTER_MODEL`. Additive with a sound goal (legacy parity guard), but the frozen expectations were generated against the PR's pickle-loaded models, and the file imports `chunk_document` from PR-new `chunks.py` — neither assumption has been validated against master's `.skops` artifacts.
- **Notes:** Before merging, run this file on master with the `.skops` models and confirm the fixture expectations hold bit-for-bit; if the skops re-export is numerically faithful they should, but that is unverified — hence the edit/validate gate and lower confidence.

### lexnlp/nlp/en/tests/test_sota_payloads.py  (batch 20, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** New 190-line contract suite for the new `segments/payloads.py` embedding-payload API: exact-source preservation, carried-ancestry rules, overlap-not-ancestry enforcement, table-context tracing/dedup with `PayloadBudgetExceeded`, and the `ContextFragment` role-owner contract. Hermetic (uses `len` as token counter, hand-built structural spans), additive, and independent of model artifacts or dependency versions.
- **Notes:** Cherry-pick with `payloads.py` (+ `chunks.py`/`hierarchy.py` it builds on). No edits needed to the test file itself.

### lexnlp/nlp/en/tests/test_sota_retrieval_quality.py  (batch 20, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 89-line suite covering retrieval metric helpers (`retrieval_metrics`, `lexical_rank` from `segmentation_quality.py`): union-based scoring for overlapping spans, no 10%-coverage-to-1.0 normalization, reciprocal-rank penalty, invalid/empty-input rejection, plus a synthetic 6-query lexical-retrieval lane over generated chunks. Pure new-module coverage, hermetic, no master code touched.
- **Notes:** Take with `segmentation_quality.py`, `chunks.py`, and the `retrieval_gold.json` fixture. The `>= 0.8` mean-recall threshold is a smoke-level bar on synthetic data, fine as plumbing evidence.

### lexnlp/nlp/en/tests/test_titles.py  (batch 20, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** Converts the network-dependent `test_title_1` (live `requests.get` to raw.githubusercontent.com, which master still performs without `raise_for_status`) into a hermetic local-fixture read via `pathlib`, modernizes `test_title_2` file IO the same way, and adds two tests for the PR's extracted `_download_training_document` helper (timeout passed as `TITLE_TRAINING_REQUEST_TIMEOUT`, HTTP errors raised). Removing live-network dependence from tests is an improvement aligned with our direction, and all three master test methods are retained.
- **Notes:** Two verification gaps require edits: (1) `test_title_1` swaps in a *different* document (`1205332_2008-05-08_3` exists on master, but the old URL pointed at `1000694...AGREEMENT OF LEASE`) while keeping the expected `['LEASE AGREEMENT']` — run it against master to confirm that expectation actually holds for the new fixture; (2) the two helper tests need the `_download_training_document`/`TITLE_TRAINING_REQUEST_TIMEOUT` hunk from PR `titles.py` (a good hunk: preserves the 60s timeout, adds `raise_for_status`) — take that hunk only, and reject that file's `load_bundled_model` → `load_joblib_model` and `to_numpy(dtype=float)` → `to_numpy()` reversions.

### lexnlp/ml/tests/test_artifact_io.py  (batch 18, part 1/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** New 100-line test for PR-only `lexnlp/ml/artifact_io.py` (`atomic_output_path`, `atomic_pickle_dump`), a module that does not exist on master ec55727. Master persists models via `lexnlp/ml/model_io.py` (`dump_model`/`load_model`/`load_bundled_model` on secure `.skops` with `skops>=0.11`). Merging this test locks in the regressed pickle persistence path (`pickle.dumps(..., HIGHEST_PROTOCOL)`, `model.pickle` fixtures) that master deliberately replaced.
- **Notes:** Atomic-write mechanics themselves (mkstemp sibling + chmod preserve/0o644 default + fsync + symlink refusal) are sound, but the subject is wrong. Salvage only by porting the test to master's `model_io`/`dump_model` skops path; as written it asserts pickle bytes.

### lexnlp/ml/tests/test_predictor_compatibility.py  (batch 18, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** New 50-line test for PR-only predictor API (`ProbabilityPredictorIsContract(pipeline=...).compatibility_report`, `restore_legacy_model_state`, `CompatibilityReport.estimator_attribute_upgrades == 3`). True diff `git diff ec55727..pr30 -- lexnlp/ml/predictor.py` shows the PR reverts master (`from lexnlp.ml.model_io import load_model`, `load_model(path, trusted=True)`, inline `_patch_legacy_estimator_attributes`) to `from lexnlp.utils.unpickler import load_sklearn_model` plus raw `open(path,'rb')`. Master already repairs `sigma_->var_/variance_` and `MinMaxScaler.clip` without a report object, so this test cannot pass on master and entrenches the pickle-loader regression.
- **Notes:** Covers `MinMaxScaler.clip=False` aliasing and `sigma_/var_/variance_` patching, but via the wrong loader. Also drags typing backwards (`Pipeline | None` -> `Optional[Pipeline]`).

### lexnlp/nlp/en/segments/__init__.py  (batch 18, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Turns the bare metadata-only `__init__.py` on master (verified via `git show ec55727:...`, just `__author__/__version__ = "2.3.0"`) into the public re-export for the new SOTA segmentation stack (`backends`, `chunks`, `hierarchy`, `payloads`, full `__all__`). No master code is removed or downgraded; change is purely additive plus a `2.3.0` -> `2.4.0a1` version bump, copyright `2015-2021` -> `2015-2026`, license `.../blob/2.3.0/LICENSE` -> `.../blob/master/LICENSE`, and mode `100755` -> `100644`.
- **Notes:** Only merge together with its companion new modules (`backends.py`, `chunks.py`, `hierarchy.py`, `payloads.py`); importing it alone breaks. Reconcile the `2.4.0a1` bump with the release process.

### lexnlp/nlp/en/segments/backends.py  (batch 18, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** New 106-line module with no counterpart on master (`git ls-tree ec55727 -- lexnlp/nlp/en/segments/` has no `backends.py`). Adds a `SplitModel` Protocol (`split(text, **kwargs)`), exact `parts_to_spans` lossless validator, `SaTSentenceSegmenter` adapter requiring caller-pinned `backend_id`, and a `legacy_sentence_segmenter` seam over `get_sentence_span`. No downgrade, no deleted master work; dependency-light by design (no `wtpsplit` import).
- **Notes:** No regression risk; correctness of SaT span fidelity not fully audited here, but interface is clean.

### lexnlp/nlp/en/segments/chunks.py  (batch 18, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** New 1770-line deterministic authenticated chunking module, absent on master. Adds `ChunkingManifest`/`DocumentChunk` with `source_sha256`/`manifest_id`/`chunk_metadata_sha256`/`provenance_sha256`, `iter_chunks`/`chunk_document`/`reconstruct_chunks` over character and token (`MONOTONIC`/`ARBITRARY`) budgets with overlap, boundary respect, and `ContainerPolicy`. Purely additive SOTA feature; nothing on master is reverted.
- **Notes:** Large new surface (token-search envelopes `DEFAULT_ARBITRARY_TOKEN_SEARCH_MAX_CALLS=10_000`, `..._INPUT_BYTES=16_000_000`, `..._STEPS=100_000`, `CHUNKING_SCHEMA_VERSION=2`); regression verdict is firm but full correctness audit is out of scope for this pass.

### lexnlp/nlp/en/segments/hierarchy.py  (batch 18, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** New 1458-line lossless source-mapped hierarchy module, absent on master. Adds `Segment`/`DocumentHierarchy`/`HierarchyManifest`/`StructuralSpan`/`SegmentKind`/`StructuralMode`/`StructureProfile`, builtin legal-structure detectors (explicit/numeric headings, clauses, list items, pipe/tab delimited tables), laminar validation, `MAX_HIERARCHY_DEPTH=128`, and tree-SHA256 manifests. Additive feature; no master file is touched.
- **Notes:** Same caveat as chunks.py: no-regression confidence is high (verified absent on master), functional correctness of the heuristics is not fully audited here.

### lexnlp/nlp/en/segments/page_segmenter.pickle  (batch 18, part 7/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Binary revert of master's secure serialization. `git diff ec55727..pr30 --name-status` shows `D page_segmenter.skops` + `A page_segmenter.pickle` (same for paragraph/section/sentence/title assets). Master ships `.skops` via `skops>=0.11`; the PR restores executable-on-load `.pickle`. Merging reintroduces the arbitrary-code-execution load path master removed.
- **Notes:** Keep `lexnlp/nlp/en/segments/page_segmenter.skops`; do not accept the `.pickle`. Paired source regression is `pages.py` switching `load_bundled_model` -> `load_joblib_model`.

### lexnlp/nlp/en/segments/pages.py  (batch 18, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Mix of a security/modernisation revert and genuine fixes. True diff reverts master (`from lexnlp.ml.model_io import load_bundled_model`, `load_bundled_model(...page_segmenter.pickle)`, `collections.abc.Generator`, f-strings) to `from lexnlp.utils.unpickler import load_joblib_model` + `.format()` + `typing.Generator`, which must not merge. Buried inside are real improvements worth salvaging: fixes the buggy `line_window_post = len(lines) - line_window_post - 1` to `min(line_window_post, len(lines) - line_id - 1)`, freezes the feature schema against `lines_count`, and adds empty-input plus `has_compatible_line_window`/`has_compatible_feature_width` early returns with `TRAINED_LINE_WINDOW_PRE/POST`.
- **Notes:** Cherry-pick only the window-clipping fix, fixed-schema `get_page_break_feature_names`, and the guards; reimplement them on master's `load_bundled_model`/`.skops` loader and modern `list[str]`/f-string style. Default change `window_pre=3` -> `window_pre=TRAINED_LINE_WINDOW_PRE` is value-neutral (still 3).

### lexnlp/nlp/en/segments/paragraph_segmenter.pickle  (batch 18, part 9/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Same secure-serialization revert as part 7/10 for the paragraph model: `D paragraph_segmenter.skops` + `A paragraph_segmenter.pickle`. Master loads the `.skops` sibling via `load_bundled_model`; the PR forces the pickle path via `load_joblib_model`. Accepting the binary undoes the `.skops` migration and its `skops` dependency.
- **Notes:** Keep `paragraph_segmenter.skops`; reject the `.pickle`. See `paragraphs.py` (part 10/10) for the paired loader revert.

### lexnlp/nlp/en/segments/paragraphs.py  (batch 18, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Same pattern as `pages.py`: reverts master (`load_bundled_model`, `Final`, `list[str]`/`dict[str, int | bool]`/`tuple[...]`, `collections.abc.Generator`, f-strings) to `load_joblib_model` plus `List/Dict/Set/Optional`, `typing.Generator`, `.format()`, which must not merge. Contains salvageable work: same `len(lines)-line_id-1` window fix, fixed global offset schema, schema-mismatch fallback (`yield 0, len(text), text`), and new `_normalise_paragraph_breaks` deterministic blank-run ownership (leading run to first content, internal run to one break, trailing run dropped).
- **Notes:** Port the window/schema fixes and `_normalise_paragraph_breaks` onto master's `load_bundled_model`/`.skops` base with modern types; verify the normaliser against paragraph tests since it overrides model scores inside separator runs. `window_pre=3` -> `TRAINED_LINE_WINDOW_PRE` is value-neutral.

### lexnlp/extract/en/tests/test_courts.py  (batch 16, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Core change replaces remote `pandas.read_csv("https://raw.githubusercontent.com/.../us_courts.csv")` in `test_courts`/`test_courts_rs` with local `COURTS_DATA_PATH = Path(get_module_path()) / "config" / "en" / "us_courts.csv"`. That file exists on master and `get_module_path()` exists, so this removes network dependence and is an improvement worth cherry-picking.
- **Notes:** True diff `git diff ec55727..pr30` also strips master-added docstrings, reformats double→single quotes, and flips mode 644→755. Keep only the `COURTS_DATA_PATH` hunks; restore master docstrings/formatting.

### lexnlp/extract/en/tests/test_cusip.py  (batch 16, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Adds `test_typed_annotation_serializes_text_and_checksum` asserting `annotation.text == '837649128'`, `get_extracted_text(...) == '837649128'` and `tags['Extracted Entity Checksum'] == 8`. The test is valuable (covers checksum `0`-vs-`None` handling already fixed on master) but it fails on master alone because master `get_cusip_annotations` never sets `text=` (defaults to `""`); it depends on the companion PR fix in `lexnlp/extract/en/cusip.py` adding `text=code`.
- **Notes:** True diff is mostly black→single-quote reformatting plus mode 644→755 churn. Cherry-pick only the new test method plus the `cusip.py` `text=code` fix together; discard quote/mode churn.

### lexnlp/extract/en/tests/test_ratios_plain.py  (batch 16, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Adds `test_ratios_do_not_require_nltk_unit_expansion` which patches `lexnlp.extract.en.amounts.get_np` and `nltk.word_tokenize` to raise and asserts `get_ratio_annotations("Ratio of 3.0:1.5.")` still yields ratio 2 without calling them. This guards the companion PR fix in `ratios.py` passing `extended_sources=False` to `get_amounts`; on master `ratios.py` uses the default `extended_sources=True`, which does call `get_np`/`word_tokenize` via `amounts.py`, so the test fails without that source fix.
- **Notes:** Remainder is quote/import-order churn (`"en"`→`'en'`). Keep the new test + `ratios.py` `extended_sources=False` change together; drop formatting churn.

### lexnlp/extract/en/utils.py  (batch 16, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Good micro-cleanups: deletes dead `words = re.findall(r'[a-zA-Z]+', text)` (unused, confirmed on master line 47) and renames cryptic `l` to `leaf_group` in `cleanup_leaves`. No behavior change. The rest is style downgrade: `from collections.abc import Generator` + builtin `list[...]`/`tuple[...]` on master reverted to `from typing import Generator, List, Tuple`, plus double→single quote churn.
- **Notes:** Keep the dead-code deletion and variable rename only; restore master typing (`collections.abc.Generator`, `list`/`tuple` generics for requires-python `>=3.13,<3.15`) and quote style.

### lexnlp/extract/es/dates.py  (batch 16, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Real functional improvements over master: `SEQUENTIAL_DATES_RE` becomes whitespace-tolerant (`\s+` instead of literal single spaces, handling `15   de   febrero` and linebreaks), adds overlap filtering that discards spurious dateparser candidates like `28 de abril y 17 de` → 2017-04-28 while keeping canonical `sequential_texts`, adds `keys_to_replace` prefix-eviction and per-call `_build_parser()`/`_coerce_locale()` replacing the shared module singleton (`get_dates = parser.get_dates`), fixing locale/state leakage. Adds missing `DateAnnotation` import.
- **Notes:** Style is a downgrade: master `str | None`, `dict[str, Any]`, double quotes reverted to `Optional/Dict/Generator`, single quotes. The pasted `-import datetime` is merge-base only (absent on master). Keep logic, restore modern `X | Y` hints and formatting.

### lexnlp/extract/es/tests/test_dates.py  (batch 16, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Adds two valuable tests for the `es/dates.py` fix: `test_sequential_dates_accept_legal_document_whitespace` (multiple spaces + `\n`) and `test_sequential_date_split_across_linebreak` (`17 de\nnoviembre de 1995` with exact coords/dates). These directly cover the new `\s+` regex and overlap logic and should be cherry-picked with the source fix.
- **Notes:** Remainder is double→single quote and line-join churn in pre-existing asserts. Keep the two new test methods; discard formatting churn.

### lexnlp/extract/es/tests/test_definitions.py  (batch 16, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** True net diff vs master (`git diff ec55727..pr30`) is formatting-only: import re-wrap, double→single quotes, comment spacing. The pasted diff's `get_definition_list` import removal is a merge-base artifact — master already imports only `get_definition_annotation_list, get_definition_annotations, make_es_definitions_parser`, and `get_definition_list` still exists in `es/definitions.py` (line 121), so nothing functional changes here.
- **Notes:** PR additionally deletes master's detailed `test_grab_just_quoted_words` docstring. No value to cherry-pick; reject to avoid docstring/formatting downgrade.

### lexnlp/extract/ml/en/data/definition_model_layered.pickle.gzip  (batch 16, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 95
- **Summary:** PR adds a ~3.7MB `definition_model_layered.pickle.gzip` blob while master tracks only `definition_model_layered.skops.zip` (secure `.skops` format per confirmed master state). Reintroducing the legacy `.pickle.gzip` bundle reverts the skops security migration and drops the `skops>=0.11` trust path.
- **Notes:** `git diff --stat ec55727..pr30` shows `Bin 0 -> 3690751 bytes` for the pickle; do not accept this file under cherry-pick.

### lexnlp/extract/ml/en/definitions/layered_definition_detector.py  (batch 16, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Genuine security/robustness fix over master's `os.mkdir` + `extractall` + `os.listdir` flow: validates `set(archive.namelist()) == {"definition.pickle", "term.pickle"}` with explicit `missing=/unexpected=` error (replacing buggy `file_name.filename` check), loads via `archive.open(...)` + existing `load_from_stream()` (confirmed present on master `artifact_detector.py:35`) so nothing is extracted to disk (fixes ZipSlip, e.g. `../outside.pickle`), and trains via `tempfile.TemporaryDirectory` + atomic `archive_path.replace(destination)` with `mkdir(parents=True)`.
- **Notes:** Accompanying churn downgrades `list[DefinitionAnnotation]`→`List[...]`, double→single quotes, and signature re-wrapping. Keep the `load_compressed`/`train_on_formatted_data` logic; restore master typing/quotes. Do not take the `.pickle.gzip` blob with it.

### lexnlp/extract/ml/en/definitions/tests/test_layered_definition_detector.py  (batch 16, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Adds three high-value tests guarding the detector fix: in-memory load with no disk extraction, parametrized rejection of missing/unexpected members (including `../outside.pickle` ZipSlip and `notes.txt`), and atomic-train verifying exact `{"definition.pickle","term.pickle"}` contents and a clean destination dir. These are new coverage master lacks and are compatible since `load_from_stream` exists on master.
- **Notes:** Remainder is quote/whitespace churn in `TRAINED_MODEL_PATH` and `test_parse_trivial`. Keep the `RecordingDetector`/`TrainingDetector` tests; discard formatting churn.

### lexnlp/extract/en/constraints.py  (batch 14, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 80
- **Summary:** Replaces the single `RE_CONSTRAINT` pattern (with `pre`/`post` captures via `capturesdict()`) with two simpler delimiter-anchored patterns plus a cursor loop, motivated as a fix for quadratic `.*?` backtracking on trigger-free sentences. The rewrite drops the `strict` gate entirely (`strict` is accepted but never read; master skips matches with empty pre+post when `strict=True`), changes `coords` from `match.span()` to `(cursor, match.end())`, and forces `post=""` on all loop matches, so spans and pre/post fields are not equivalent to master despite the "keeps the legacy spans intact" comment. It also reverts modern type hints (`collections.abc Generator`, `list[...]`, `X | None`, f-strings) to `typing.Generator/List/Optional/Tuple` and `.format()`.
- **Notes:** True diff is `git diff ec55727..pr30 -- lexnlp/extract/en/constraints.py`. Perf idea may be worth re-doing on top of master, but only with `strict` restored, span parity proven against `test_constraints.py` fixtures, and modern hints kept.

### lexnlp/extract/en/contracts/contract_type_detector.py  (batch 14, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** Makes two functional changes: `joblib.load` becomes `load_joblib_model` from `lexnlp.utils.unpickler` (a scoped compat loader that handles legacy sklearn tree-dtype renames master’s raw `joblib.load` lacks), and positional `type_vector[0]` / `type_vector[1]` become `type_vector.iloc[0]` / `type_vector.iloc[1]`. The `.iloc` fix is unambiguously good — with string-labeled Series, `Series.__getitem__` positional fallback is deprecated/fragile. The loader swap is also forward, not a skops regression (this RF+Doc2Vec artifact is joblib, outside the `.skops` pipeline path). The file also reverts style (`typing.List`, `''` quotes, reformatting) away from master’s modern form.
- **Notes:** Cherry-pick the `.iloc[0]`/`.iloc[1]` fix and evaluate `load_joblib_model` vs master’s `renamed_load` compat path; drop the `List`/quote churn.

### lexnlp/extract/en/contracts/predictors.py  (batch 14, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Three functional improvements over master: legacy is-contract fallback uses scoped `load_sklearn_model` instead of raw `cloudpickle.load`; both fallback `RuntimeError` messages gain actionable recovery steps (`download_github_release('pipeline/is-contract/0.1', prompt_user=False)` and `ensure_runtime_contract_type_model(force=True)` alongside the bootstrap scripts); and `predictions[0]`/`predictions[1]` become `predictions.iloc[0]`/`predictions.iloc[1]` in `infer_classification` (same pandas positional-indexing fix as the detector). Counterweight is pure style regression: `collections.abc Iterable`, `bool | tuple[...]`, `str | Iterable[str]`, double quotes reverted to `typing.Iterable/Tuple/Union` and single quotes.
- **Notes:** Take the loader, error-message, and `.iloc` changes; restore master’s `X | Y` hints and `collections.abc` imports before merging.

### lexnlp/extract/en/contracts/runtime_model.py  (batch 14, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** Reverts master’s secure serialization stack — canonical `CONTRACT_TYPE_MODEL_FILENAME = "pipeline_contract_type_classifier.skops"` via `lexnlp.ml.model_io.dump_model`/`load_model(path, trusted=True)` — to `pickle.dump` plus `load_sklearn_model` under the legacy `.cloudpickle` name, i.e. the known SECURITY REGRESSION. It also changes the training recipe: `max_features=120000` becomes a `75_000` parameter default, default `max_docs_per_label` flips from `120` (with `<= 0` rejected) to `0`-means-all-corpus, sorted-member sampling is replaced by raw archive order, `write_pipeline_to_catalog` changes return type from `tuple[Path, bool]` to bare `Path`, and the module docstring downgrades Python `3.13+` to `3.11`. Good ideas are bundled in (single-thread `threadpool_limits` fit, atomic writes, manifest/checksum verification, local-candidate tag split), but they are wired to PR-only catalog APIs (`get_exact_path_from_catalog`, `get_local_candidate_tag`, `download_github_release_to_path`, `lexnlp.ml.artifact_io`).
- **Notes:** Do not merge; re-implement desirable bits (thread pinning, atomic write, trust checks) on top of the `.skops`/`model_io` path instead.

### lexnlp/extract/en/contracts/tests/test_contract_type_quality_gate.py  (batch 14, part 5/10)
- **Regression:** NO
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** New 137-line test for `scripts.contract_type_quality_gate.evaluate_duplicate_group_holdout` and `verify_fixture_sha256` (holdout accept/regress/missing-evidence/split-mismatch cases). The file does not exist on master, so it removes nothing, but it imports PR-only script APIs — master’s `scripts/contract_type_quality_gate.py` has no `evaluate_duplicate_group_holdout` (it gates on fixture metrics, not duplicate-group holdout). Merging this file without the PR’s script rewrite (a separate batch decision, coupled to the rejected training-recipe change) breaks collection.
- **Notes:** Defer to the `scripts/train_contract_type_model.py` / `scripts/contract_type_quality_gate.py` verdict; only merge together with that API if it is ever adopted.

### lexnlp/extract/en/contracts/tests/test_contract_type_training.py  (batch 14, part 6/10)
- **Regression:** NO
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** New 91-line test for `scripts.train_contract_type_model.build_duplicate_group_holdout` and `split_assignment_sha256` (determinism, leak-freedom, validation-size, fingerprint sensitivity). Like part 5/10 it regresses nothing on master (file absent there) but depends entirely on PR-only training-script functions — master trains with `train_test_split`, not a global duplicate-group holdout. Standalone merge fails at import.
- **Notes:** Same coupling as part 5/10; judge jointly with the training-script change, not here.

### lexnlp/extract/en/contracts/tests/test_model_tag_overrides.py  (batch 14, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 80
- **Summary:** True diff (`git diff ec55727..pr30`) deletes master’s `test_probability_predictor_uses_model_io_loader_for_default_pipeline`, the test that pins the secure `load_model(path, trusted=True)` default-pipeline path — that deletion removes skops coverage and is a regression. It swaps the legacy-fallback mock from `cloudpickle.load` to `unpickler.load_sklearn_model` (consistent with the PR’s insecure-loader direction, not master’s) and adds two error-message tests (`download_github_release` / `ensure_runtime_contract_type_model(force=True)` guidance, `__cause__` chaining) that are themselves fine.
- **Notes:** As presented, reject; a future split could keep the two new guidance tests on top of master while restoring the deleted `model_io` pinning test and the `cloudpickle` mock appropriate to master’s loader.

### lexnlp/extract/en/contracts/tests/test_runtime_model.py  (batch 14, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** Rewrites master’s `test_runtime_model.py` (+229/−224) to match the PR’s rejected runtime API: it drops expectations tied to `write_pipeline_to_catalog` returning `(Path, bool)`, `max_docs_per_label=120` force-train flow, and sorted sampling, replacing them with tests for cap-`0`-means-all, atomic corrupt-artifact repair, manifest-pinned write rejection (`ChecksumError`), and private local-candidate tags. Every added test imports PR-only behavior (`get_local_candidate_tag`, `download_github_release_to_path`, bare-`Path` returns) absent on master, while the deleted tests pin the correct `.skops` behavior.
- **Notes:** Reject wholesale with `runtime_model.py` (part 4/10); master’s existing tests must stay until any future skops-native rework lands.

### lexnlp/extract/en/contracts/tests/test_series_positioning.py  (batch 14, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** New 27-line test asserting both `ProbabilityPredictorContractType.infer_classification` and `ContractTypeDetector.detect_contract_type` resolve the top label from a string-indexed `Series([0.9, 0.1], index=["EMPLOYMENT AGREEMENT", "SERVICES AGREEMENT"])`. It depends only on modules that exist on master, imports nothing PR-only, and guards the `.iloc` migration in parts 2–3/10 against pandas positional-getitem deprecation. Safe standalone addition.
- **Notes:** Merge as-is; pairs with the `.iloc[0]`/`.iloc[1]` fixes.

### lexnlp/extract/en/cusip.py  (batch 14, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** The one functional line, `text=code` in the `CusipAnnotation(...)` construction, is a safe improvement: master’s annotation accepts `text` but the extractor never passed it (leaving `.text` empty/`None`), while `to_dictionary_legacy()` already reports `'text': self.code`, so populating the attribute aligns the object with its own legacy dict and harms no existing `test_cusip.py` assertions. Everything else in the true diff is style regression — `typing.Dict/List/Generator`, single-quote churn, and import reordering — reverting master’s `dict`/`list`/`collections.abc` modern form.
- **Notes:** Cherry-pick only `text=code` onto `ec55727:lexnlp/extract/en/cusip.py`; discard the quoting/typing churn.

### lexnlp/extract/common/text_beautifier.py  (batch 12, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Pasted hunk (`except:` → `except Exception` in `unify_quotes_braces` / `unify_quotes_braces_coords`) is a genuine improvement over master's bare `except:  # pylint:disable=bare-except`. True diff `git diff ec55727..pr30` shows it bundled with a broad style downgrade: `str | tuple[str,int,int]` → `Union[str, Tuple[str,int,int]]`, `list[int]` → `List[int]`, `str | None` → `Optional[str]`, plus double→single quote churn. Take only the two `except Exception` lines; reject the typing/quote reverts.
- **Notes:** Master: `except:  # pylint:disable=bare-except`; PR: `except Exception:`. Regression lines e.g. `strip_pair_symbols(term_coords: str | tuple[str, int, int])` → `Union[...]`, `find_transformed_word(...) -> tuple[str,int] | None` → `Optional[Tuple[str,int]]`.

### lexnlp/extract/de/citations.py  (batch 12, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** The one-line `except:` → `except Exception` around `get_dates(date,'de')` is good. True diff shows it bundled with a modernisation revert: `from collections.abc import Generator` → `from typing import Dict, Generator, List`, `list[CitationAnnotation]` → `List[CitationAnnotation]`, `Generator[CitationAnnotation]` → `Generator[...,None,None]`, `dict` → `Dict`, plus quote-only churn. No logic change otherwise. Cherry-pick the `except Exception` hunk only.
- **Notes:** Master line: `date = str(list(get_dates(date, "de"))[0]["value"])` + bare `except:`; PR same logic with `except Exception:` — keep that, drop `List/Dict/Generator` reverts.

### lexnlp/extract/de/court_citations.py  (batch 12, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Functional core is an improvement: master shares a module-global `parser = CourtCitationsParser()` across `get_court_citation_annotations/list/citations` (`yield from parser.parse(...)` mutates `self.items`/`self.language`), while PR constructs a fresh `CourtCitationsParser()` per call, fixing shared-mutable/thread-safety state. However true diff bundles it with typing downgrades (`collections.abc Generator` → `typing Generator`, `list[]` → `List[]`, `tuple[]` → `Tuple[]`, `str`→single quotes) and deletes the `split_text_by_keywords` docstring. The pasted `- from ... Locale` removal is a merge-base artifact — master already lacks that import. Keep fresh-instance logic re-expressed with master's modern typing; reject the rest.
- **Notes:** Master: `parser = CourtCitationsParser()` + `yield from parser.parse(text, language)`; PR: `yield from CourtCitationsParser().parse(text, language)` (x4 functions, `parser` global left dead). Keep latter behavior, keep `list[]/tuple[]/Generator` from `collections.abc`.

### lexnlp/extract/de/date_model.pickle  (batch 12, part 4/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Binary re-add is a security regression. Master has no `lexnlp/extract/de/date_model.pickle` — it ships `date_model.skops` (+ `model.skops`) loaded via `lexnlp.ml.model_io.load_bundled_model` with `skops>=0.11`. PR reintroduces a 54794-byte `.pickle` and loads it via pickle/joblib (`lexnlp.utils.unpickler.load_joblib_model`), reinstating arbitrary-code-execution on load. No-op at best if cherry-picked; wholesale merge would resurrect deleted legacy artifacts.
- **Notes:** `git ls-tree ec55727 -- lexnlp/extract/de/` shows only `.skops`; `git cat-file -s pr30:lexnlp/extract/de/date_model.pickle` = 54794. See also `dates.py` loader revert.

### lexnlp/extract/de/dates.py  (batch 12, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Two changes in one file. Good: replaces shared-global `parser = DeDateParser(...)` + `get_dates = parser.get_dates` aliases with `_coerce_locale`/`_build_parser` fresh-parser-per-call wrappers plus `base_date`/`threshold` parameters, fixing concurrent-call mutable `text/locale` sharing. Bad (security regression): swaps `from lexnlp.ml.model_io import load_bundled_model` (skops) for `from lexnlp.utils.unpickler import load_joblib_model` (pickle/joblib). Merge the fresh-parser structure rewired to `load_bundled_model`/`.skops`; reject the unpickler import.
- **Notes:** Master: `MODEL_DATE = load_bundled_model(...date_model.pickle)` + `get_dates = parser.get_dates`; PR: `MODEL_DATE = load_joblib_model(...)` + `def get_dates(text=None, locale=None)` → `_build_parser(locale=locale).get_dates(...)`. Keep second shape, restore first loader.

### lexnlp/extract/de/de_date_parser.py  (batch 12, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Real bug fixes mixed with style reverts. PR fixes document-relative coordinates by replacing length-changing `re.sub(CUSTOM_DATES_SEPARATOR,'\n')` + `split('\n')` with length-preserving `text_part_spans` offsets (`location_start = text_part_start + match.start()`), hoists `positions = []` outside the part loop so cross-part overlap is tracked, copies locale via `Locale(locale.get_locale())` instead of mutating `self.locale.language`, and replaces noisy `except Exception as e: print(str(e))` with `except (TypeError, ValueError): self.dates = []`. All worth keeping. Bundled regressions: `str | None` → `Optional`, `list[DatePart]` → `List[DatePart]`, `except:` → `except Exception` (good in isolation), deleted `get_word_parts`/`get_date_annotations` docstrings, quote churn. Keep functional hunks, restore master's typing/docstrings.
- **Notes:** Master bug: coords relative to `text_part` after `und`-substitution drift; PR test `text[slice(*coords)] == ['15. Februar 1972','29. Dezember 1972']` proves fix. Narrow `except (TypeError, ValueError)` is acceptable here (dateparser failures); do not restore `print`.

### lexnlp/extract/de/definitions.py  (batch 12, part 7/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** No functional change; pure modernisation revert versus master HEAD. True diff: `from collections.abc import Generator` → `from typing import List, Generator`, `list[PatternFound]` → `List[PatternFound]`, `list[DefinitionAnnotation]` → `List[...]`, double→single quotes, and `yield from dfs` → `for d in dfs: yield d`. The pasted `Optional` removal (`List, Generator, Optional` → `List, Generator`) is a merge-base artifact — master has no `typing` import at all. Reject wholesale; re-do any unused-import cleanup on master if still needed.
- **Notes:** Master: `def match_im_sinne(phrase: str) -> list[PatternFound]`, `def get_definition_annotations(...) -> Generator[DefinitionAnnotation]`; PR reverts both to `typing.List/Generator`.

### lexnlp/extract/de/model.pickle  (batch 12, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 100
- **Summary:** Same pattern as `date_model.pickle`: master deleted `lexnlp/extract/de/model.pickle` and ships `model.skops` via `skops>=0.11`; PR re-adds a 29064-byte legacy `.pickle`. Merging reintroduces an insecure pickle artifact the project deliberately migrated away from. Reject.
- **Notes:** `git ls-tree ec55727` = `model.skops`, no `.pickle`; `git cat-file -s pr30:lexnlp/extract/de/model.pickle` = 29064.

### lexnlp/extract/de/percents.py  (batch 12, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** Contains one genuine one-line bug fix: master computes `real_amount = PERCENT_UNITS_MAP.get(...) * amount` then discards it with `real_amount = round(amount, float_digits)`; PR correctly rounds the scaled value with `real_amount = round(real_amount, float_digits)`. The rest is a style downgrade (`collections.abc Generator` → `typing Generator`, `list[]` → `List[]`, `rf"...{amounts_parser.NUM_PTN}..."` → `r"...{num_ptn}...".format(...)`, quote churn). Cherry-pick only the `round(real_amount, ...)` line.
- **Notes:** Exact lines — master `lexnlp/extract/de/percents.py:72`: `real_amount = round(amount, float_digits)` (bug); PR `:73`: `real_amount = round(real_amount, float_digits)` (fix). Verify with e.g. `50%` with `float_digits=4` → `0.5` vs `50.0`.

### lexnlp/extract/de/tests/test_dates.py  (batch 12, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** The 4-line added assertion in `test_point_inside_with_two_dates` (`assertEqual(['15. Februar 1972','29. Dezember 1972'], [text[slice(*a.coords)] for a in dates])`) is valuable — it locks in the `de_date_parser.py` span-preserving coordinate fix. Everything else in the true diff is churn to reject: double→single quote flips, deletion of docstrings master added (`test_dates`, `test_date_reverse_order`, `test_negative_jahr`, `test_point_inside`), and a gratuitous mode change `100644` → `100755`. Cherry-pick the assertion hunk only.
- **Notes:** Keep hunk at `test_point_inside_with_two_dates` tail; discard `old mode 100644 / new mode 100755` and docstring deletions.
