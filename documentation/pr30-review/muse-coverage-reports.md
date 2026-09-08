# PR30 test-generation report (Muse, batch 1)

### scripts/train_contract_model.py
- **Tests added:** 43 in `scripts/tests/test_train_contract_model_coverage.py`
- **Lines now covered:** all 205 listed missing lines (my file alone measures 100%: 205 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** `--positive-tags`/`--negative-tags` empty-list guards (lines 169-172) cannot be triggered via CLI (`nargs="+"` never yields an empty list), so they are tested by emptying the module-level `DEFAULT_*_TAGS` with monkeypatch. `main()` trains real sklearn estimators on 12 synthetic texts (CountVectorizer + densifier + GaussianNB/LogReg/RandomForest-300); only catalog download I/O and the `run_quality_gate` subprocess are mocked.

### lexnlp/extract/ml/classifier/spacy_token_sequence_model.py
- **Tests added:** 31 in `lexnlp/extract/ml/classifier/tests/test_spacy_token_sequence_model_coverage.py`
- **Lines now covered:** all 191 listed missing lines (my file alone measures 100%: 191 stmts, 0 miss)
- **Unreachable:** none (`if TYPE_CHECKING:` import is excluded by coverage; `__main__`-style guards do not exist in this module)
- **Notes:** `spacy` is not installed here, so only the `spacy` import / model `load` is faked (recording stub + hand-built `FakeToken`/`FakeDoc`); every feature-list/feature-data assertion runs the real code. Observation for the maintainer (no source change made): the post-window loop uses `j_end = i + post_window` (exclusive), so interior tokens never receive the `+1` neighbor copy — only near-tail tokens do (pinned by `test_window_range_excludes_self`). Also `NUM` is a member of `SPACY_POS_LIST`, and the `_lchar_*` lookup requires lowercase entries in `letter_set` (real callers pass `string.ascii_letters`).

### scripts/bootstrap_assets.py
- **Tests added:** 50 in `scripts/tests/test_bootstrap_assets_coverage.py`
- **Lines now covered:** all 141 listed missing lines (my file alone: 98%; the remaining 6 lines — 199-200, 210, 250-251, 261 — were already covered by `scripts/tests/test_bootstrap_assets.py`; combined run measures 100%: 241 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** network (`urlopen`), `nltk.download`, catalog downloads and the contract-type builder are mocked; tar/zip handling, Stanford verification, the 404→legacy re-export fallback (incl. its inner-failure path) and task wiring all run for real against tmp dirs.

### scripts/model_quality_gate.py
- **Tests added:** 38 in `scripts/tests/test_model_quality_gate_coverage.py`
- **Lines now covered:** all 130 listed missing lines (my file alone measures 100%: 130 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** `score_pipeline` is tested against a real fitted CountVectorizer+LogisticRegression pipeline wrapped by the real `ProbabilityPredictorIsContract`; `main()` is tested end-to-end (real fixture CSV + real scoring, incl. write-then-reuse of the baseline-metrics JSON) plus canned-metrics cases for every violation/mismatch branch. Only catalog downloads are mocked.

### Cross-cutting
- All 162 new tests pass (`uv run --python 3.13 pytest <4 files> -q`); `ruff check` and `ruff format --check` are clean on all four files.
- Pre-existing, unrelated failure left untouched: `scripts/tests/test_reexport_bundled_sklearn_models_coverage.py::TestSanitizePandasIndices::test_tuple_and_set_recurse_into_mutables` (another batch's file; fails standalone).

### lexnlp/extract/ml/classifier/token_sequence_model.py
- **Tests added:** 29 in `lexnlp/extract/ml/classifier/tests/test_token_sequence_model_coverage.py`
- **Lines now covered:** all 74 listed missing lines (with `test_feature_mask_optional.py`: 100%, 162 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** real model construction throughout (no mocks at all). Pins the `get_feature_list` None-vs-explicit default branches, every char class (letter/digit/punc/symbol/other incl. single-char first+last tokens), `feature_mask` max-per-token, `string_checks` flags, unicode category counts, consecutive-separator tokenization (`"ab  cd"` -> `[(0, 2), (4, 6)]`), and the window-copy quirk (first token gets no neighbor copies; middle token's `-1_*` mirrors token 0's `0_*`).

### lexnlp/nlp/train/en/train_section_segmanizer.py
- **Tests added:** 20 in `lexnlp/nlp/train/en/tests/test_train_section_segmanizer_coverage.py`
- **Lines now covered:** all listed missing lines except 648-651 (with `test_section_segmanizer_normalization.py`: 97%, 127 stmts, 4 miss)
- **Unreachable:** 648-651 (`train_logistic_regression` post-fit report/return) — dead code: the hardcoded `LogisticRegression(penalty="l1", solver="lbfgs")` always raises `ValueError: Solver lbfgs supports only 'l2' or None penalties` at `fit` (line 645), so lines after it cannot execute. Pinned by `test_train_logistic_regression_rejects_l1_with_lbfgs`.
- **Notes:** sample-repo I/O mocked (`ensure_documents_in_folder`); sklearn fitting real (ExtraTrees/DecisionTree train and predict on tiny frames; `joblib.dump`/`getsize` mocked only to avoid writing into the source tree). Two real source quirks pinned, not fixed: (1) the `section`/`sw_*` keyword flags read the *last window line*, not `lines[line_id]` (tests place keyword lines accordingly); (2) `build_features` on a <4-line document raises `UnboundLocalError: 'line'` because the window range is empty (`test_short_document_leaves_window_loop_empty`). Window pre/post mutation is instance-stateful, so each feature test uses a fresh manager.

### scripts/contract_type_quality_gate.py
- **Tests added:** 20 in `scripts/tests/test_contract_type_quality_gate_coverage.py`
- **Lines now covered:** all 71 listed missing lines except line 351 (with `test_contract_type_quality_gate.py`: 99%, 153 stmts, 1 miss)
- **Unreachable:** 351 (`raise SystemExit(main(sys.argv[1:]))` under `if __name__ == "__main__":`) — import-only reachability, not contorted.
- **Notes:** only catalog downloads/model loading mocked (`get_path_from_catalog`, `download_github_release`, `model_io.load_model`, and `score_pipeline`/`load_pipeline_for_tag` inside `main()` tests). Fixture parsing, `load_baseline_metrics` (metrics/baseline-legacy/missing-section/non-object/missing-file), all three mismatch errors, regression pass/fail deltas, tolerance flags, `--min-candidate-accuracy-top1`, and both JSON writers run for real against tmp files; `main()` stdout is parsed as JSON and asserted.

### lexnlp/nlp/en/segments/chunks.py
- **Tests added:** 71 in `lexnlp/nlp/en/tests/test_chunks_coverage.py`
- **Lines now covered:** all 70 listed missing lines except 626, 850, 921, 1530, 1564-1566, 1574, 1656 (with `test_lossless_segment_core.py`: 98%, 848 stmts, 15 miss; the other 6 miss — 61, 63, 1359, 1436, 1444, 1459 — were never listed and are covered elsewhere in the full suite)
- **Unreachable:** all defensive, with reasons: 626 (`_preserved_units.emit` merge) — parent-scope gap emits are always separated by `visit()` on a protected child, which appends >=1 unit under a different (deeper) scope, so same-scope adjacency never occurs; 850 (`_raw_token_end` candidate bump) — `candidate = min(limit, fresh + 2*distance)` with `distance = last - fresh >= 1` always exceeds `last_feasible` while the loop runs; 921 (`_token_end` infeasible `None`) — every candidate reaching it was pre-checked feasible (loop condition or `boundary_limit` checks), so only a nondeterministic counter could trigger it; 1530, 1564-1566, 1574 (char-mode overlap/end fallbacks) — `overlap < budget` is enforced, hence `context_start + budget > fresh_start` and `end > fresh_start` always hold; 1656 (progress-invariant `RuntimeError`) — char path always has `end > fresh_start` with `unit_count <= budget`, token path returns only feasible plans or raises `ValueError` first.
- **Notes:** no mocks at all; hierarchies built via real `segment_document` (whole-paragraph/sentence test backends, structural spans). Covers every `_enum`/`_attributes`/`_digest_value` guard, all manifest/reference/provenance/`DocumentChunk` validators (via `dataclasses.replace`, recomputing metadata digests where required), `_TokenCountCache` eviction, `_token_context_start` unit-start reach, `_raw_token_end` past-limit `None`, both cannot-fit `ValueError`s (monotonic + arbitrary), `_normalise_document`/`iter_chunks` option guards, and all `reconstruct_chunks` guards.

### lexnlp/extract/common/copyrights/copyright_pattern_found.py
- **Tests added:** 16 in `lexnlp/extract/common/copyrights/tests/test_copyright_pattern_found_coverage.py`
- **Lines now covered:** all 51 listed missing lines (with existing suite: 100%, 51 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** no mocks at all. Covers default init, init-from-`PatternFound` field copy, `__repr__`/`get_length`, every `get_detalization_level` signal (uppercase/company/start-year/end-year, stacked to 4), and all `pattern_worse_than_target` outcomes (no-overlap, lower/higher/equal level, shorter/longer target, end-inside-only overlap).

### lexnlp/extract/common/dates_classifier_model.py
- **Tests added:** 3 in `lexnlp/extract/common/tests/test_dates_classifier_model_coverage.py`
- **Lines now covered:** all 57 listed missing lines (with existing suite: 100%, 130 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** `build_date_model` runs for real (real `Pipeline.fit`, `SelectKBest`/`VarianceThreshold`, `cross_val_score`, `joblib.dump` + reload + predict); only `RandomForestClassifier(n_estimators=10)` is narrowed for speed and `cross_val_score` is real. A text-conditional fake `parse_dates` yields both fully-matched examples (covers `correct += 1`) and mismatched ones (covers the verbose diff print), with both target classes present for stratified cv=5. The `except` re-raise branch is pinned with an unhashable parse result (`TypeError`). The empty-word `continue` is pinned via a leading separator (`",,Al 12"` splits to `["", "Al", "12"]`; note interior `",,"` collapses and does NOT produce `""`).

### lexnlp/extract/en/contracts/runtime_model.py
- **Tests added:** 14 in `lexnlp/extract/en/contracts/tests/test_runtime_model_coverage.py`
- **Lines now covered:** all 51 listed missing lines (with existing suite: 100%, 104 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** only catalog/download/tarfile boundaries mocked. Real `.tar.gz` archives exercise caps, head truncation, non-`.txt`/directory skipping, blank-text skipping, empty-archive and single-label `RuntimeError`s; a stub archive pins the `extractfile is None` skip. Notable real behavior: a top-level `.txt` member (fewer than 3 path parts) raises `ValueError` from `_extract_label` instead of being skipped.

### lexnlp/extract/en/dates.py
- **Tests added:** 18 in `lexnlp/extract/en/tests/test_dates_coverage.py`
- **Lines now covered:** all 57 listed missing lines except 262-264 (with existing suite: 98%, 244 stmts, 5 miss: 216-217 covered by `test_dates_plain.py`/`test_dates_syntax_regression.py`, plus 262-264 below)
- **Unreachable:** 262-264 (`except TypeError` around the cutter loop) — defensive dead code: the only fallible call in the guarded block is `parse_date_string`, already wrapped by the inner `except Exception` (which catches `TypeError` first); every other statement (`split`/`range`/`len`/slicing/`join`/truthiness of a date) is infallible for the `str`/`dict` inputs the extractor yields.
- **Notes:** `DateFinder.extract_date_strings` is stubbed per-test to feed crafted candidates while `parse_date_string` stays real (except two tests that stub it to raise `RuntimeError`, and to return a `tzoffset(None, -104400)` datetime whose `isoformat()` raises `ValueError`). Natural inputs that yield zero candidates (`"1st"`, `"Monday"`, `"12. Marquee"`) cannot reach the filter branches, hence the stubs. `train_default_model` is tested with mocked `build_date_model` + stub `random` namespace (save=False asserts the probe-file build+unlink round-trip under `chdir(tmp_path)`; save=True asserts `MODULE_PATH` output, `DATE_MODEL_CHARS`, and random-example generation incl. `ValueError` skips for impossible dates). `ruff check` and `ruff format --check` clean on all four files.

### scripts/segmentation_benchmark.py
- **Tests added:** 49 in `scripts/tests/test_segmentation_benchmark_coverage.py`
- **Lines now covered:** all 25 listed missing lines except 455 (with `test_segmentation_benchmark.py`: 99%, 201 stmts, 1 miss)
- **Unreachable:** 455 (`if __name__ == "__main__":` guard) — import-only reachability, not contorted
- **Notes:** real segmentation/chunking throughout at small sizes (500 chars, 5-80 headings); only `_chunk_signature` (determinism-failure test), `segment_document` (lossy-hierarchy / heading-mismatch tests) and `chunk_document` (lossy-repeat test) are stubbed per-test. Line 30 (`sys.path` insert) is pinned by removing the root from `sys.path` and `importlib.reload`-ing the module. `main()` benchmark lane tested via stdout JSON (pass and `inf`-throughput fail) plus `--output` file write.

### scripts/asset_drift_check.py
- **Tests added:** 14 in `scripts/tests/test_asset_drift_check_coverage.py`
- **Lines now covered:** all 24 listed missing lines except 145 (with `test_asset_drift_check.py`: 99%, 95 stmts, 1 miss)
- **Unreachable:** 145 (`if __name__ == "__main__":` guard) — import-only reachability, not contorted
- **Notes:** only catalog/download boundaries mocked (`get_path_from_catalog`, `download_github_release`, `ensure_tag_downloaded`); manifest parsing, sha/size/filename checks and stdout/stderr messages all run for real against tmp files. `ensure_tag_downloaded` hit, miss-then-download, and unexpected-error-propagates paths all pinned.

### lexnlp/utils/amount_delimiting.py
- **Tests added:** 31 in `lexnlp/utils/tests/test_amount_delimiting_coverage.py`
- **Lines now covered:** all 23 listed missing lines except 222, 223, 225 (with `test_amount_delimiting.py`: 97%, 99 stmts, 3 miss: 222-225)
- **Unreachable:** 222, 223, 225 (len==2 fallthrough past `if decimal_delimiter and group_delimiter: return`) — when the text has exactly 2 unique delimiters, one always equals `blocks[-1].delimiter` (becomes decimal) and the other becomes group, so both are always truthy and the `check_block_grouping` fallback below line 220 cannot execute with real inputs
- **Notes:** locale layer mocked per-test (`LocaleContextManager` + `localeconv`); all delimiter inference runs the real code. Two initial expectations were corrected against real behavior: the en_US canonical-override rewrites a mocked `group="."`, so the group-collision-None test uses `fr_FR`; and `"10.000.000"`/de_DE keeps `decimal=","`, so the decimal-None test uses `"1.000.000"`/en_US (grouping-valid, decimal present in text).

### lexnlp/extract/en/contracts/predictors.py
- **Tests added:** 16 in `lexnlp/extract/en/contracts/tests/test_predictors_coverage.py`
- **Lines now covered:** all 21 listed missing lines (with `test_model_tag_overrides.py`: 100%, 80 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** no model loads; `__new__`-constructed predictors with stub pipelines exercise the real `is_contract` (incl. `return_probability`), `make_predictions` (sort + `top_n`), every `infer_classification` branch (empty / below-min / close-runner-up / single / confident) and `detect_contract_type` end to end. Both double-failure `RuntimeError` paths (is-contract legacy fallback, contract-type runtime fallback) pinned. `classification` compares with `==` not `is` because `predict_proba` yields `numpy.bool_`.

### Cross-cutting (batch 9)
- All 110 new tests pass (`uv run --python 3.13 pytest <4 files> -q`); `ruff check` and `ruff format --check` clean on all four files.
- Heads-up: `test_predictors_coverage.py` and `test_amount_delimiting_coverage.py` already existed in the tree (commit `6b4896b`, batches 35-37) with near-identical tests — same names/counts, no tests lost. The worktree keeps that content plus two fixes: the committed amount file's en_US group-collision and de_DE decimal-None tests fail against the real canonical-override logic (fixed as described above), and the predictors file is `ruff format`-normalized. A mixed `("x", 10, 20)` sizes input raises `TypeError` at the sorted-check (line 257), so the non-int `.5`-float case `(10.5, 20, 30)` is used to reach line 260.

### lexnlp/extract/common/annotations/company_annotation.py
- **Tests added:** 15 in `lexnlp/extract/common/annotations/tests/test_company_annotation_coverage.py`
- **Lines now covered:** all 12 listed missing lines (my file alone: 100%, 37 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** no mocks. Pins `__repr__` name/abbr/text/empty fallback chain with and without company type, `company_type` full-over-abbr-over-label priority, `get_cite_value_parts`, `get_cite` end to end (`/en/company/Acme/LLC`), and `get_dictionary_values` incl. the text-or-name fallback and absent-name/absent-type key omissions.

### lexnlp/extract/common/annotations/cusip_annotation.py
- **Tests added:** 14 in `lexnlp/extract/common/annotations/tests/test_cusip_annotation_coverage.py`
- **Lines now covered:** all 12 listed missing lines (my file alone covers 85-100; only line 103, a `to_dictionary_legacy` body line outside the listed set, is left to the existing suite)
- **Unreachable:** none
- **Notes:** no mocks. Pins every `get_dictionary_values` guard: tba, string vs bool ppn, checksum `0` vs string vs None (zero digit must be kept), issuer/issue ids, plus the full-combination tags dict and `get_cite_value_parts` bool-ppn coercion to `""`.

### lexnlp/extract/common/definitions/definition_match.py
- **Tests added:** 5 in `lexnlp/extract/common/definitions/tests/test_definition_match_coverage.py`
- **Lines now covered:** all 12 listed missing lines (my file alone: 100%, 12 stmts, 0 miss)
- **Unreachable:** none
- **Notes:** no mocks. The module is a plain data holder, so tests pin defaults (`name=None`, `start/end/probability=0`), attribute assignment, instance independence, and tuple projection of results.

### lexnlp/extract/de/amounts.py
- **Tests added:** 12 in `lexnlp/extract/de/tests/test_amounts_coverage.py`
- **Lines now covered:** all 12 listed missing lines (233, 255, 256, 258, 277, 278, 279, 281, 300, 301, 317, 321; remaining misses in a solo run — 62-67, 209-211, 218, 222, 225-230, 274, 291-295 — are outside the listed set)
- **Unreachable:** none (line 281, `if amount is None: continue`, is unreachable with real inputs since `text2num` always returns a Decimal or raises; it is covered via a `monkeypatch` stub forcing `None`, disclosed here)
- **Notes:** only mock is that one `text2num -> None` stub. Everything else is real: `text2num("blah")` raises `RuntimeError("Unknown number: blah")` (line 233); `parse("dreißig")` pins all three return-shape branches; `parse_annotations("1.2.3")` yields nothing while printing the `ConversionSyntax` error (lines 277-279); `"€ 100"` vs `"€100"` pin both `sep` branches of the preceding-currency-unit logic (lines 300-301); `get_amount_list` / `get_amount_annotation_list` wrappers asserted on values, text, and coords.

### Cross-cutting (batch 13)
- All 46 new tests pass (`uv run --python 3.13 pytest <4 files> -q`); neighboring suites (`annotations/tests`, `definitions/tests`, `de/test_amounts.py`) pass unmodified — 226 passed. `ruff check` and `ruff format --check` clean on all four files.
