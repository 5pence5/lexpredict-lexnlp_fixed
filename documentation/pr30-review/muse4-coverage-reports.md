# PR30 Test Generation Report (Muse batches)

### lexnlp/extract/en/addresses/addresses.py
- **Tests added:** 16 in lexnlp/extract/en/addresses/tests/test_addresses_coverage.py
- **Lines now covered:** 27, 28, 29, 30, 31, 32, 33, 36, 39, 42, 43, 44, 47, 65, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 105, 106, 148, 149, 158, 159 (module now 100% in combined run with existing tests)
- **Unreachable:** none
- **Notes:** `get_address_annotations` (lines 148-149) is reached but raises `TypeError: AddressAnnotation.__init__() got an unexpected keyword argument 'name'` — the wrapper passes `name=""` but `AddressAnnotation.__init__` accepts only `(coords, locale, text)`. The new test asserts this raised error as observed behaviour; the wrapper likely needs a source fix (drop the `name` kwarg). `get_addresses`/`get_address_spans` verified against real classifier output on a real address sentence.

### lexnlp/extract/ml/detector/artifact_detector.py
- **Tests added:** 17 in lexnlp/extract/ml/detector/tests/test_artifact_detector_coverage.py
- **Lines now covered:** 33, 36, 39, 40, 41, 42, 43, 49, 88, 91, 94, 107, 120, 122, 123, 124, 125, 127, 128, 129, 131, 134, 137, 140, 141, 142, 143, 146, 147, 148, 158, 159 (module now 100% in combined run with existing tests)
- **Unreachable:** none
- **Notes:** `process_sample` is abstract, so a minimal concrete `DummyDetector` is used; `load*` paths mock only the `BaseTokenSequenceClassifierModel` loaders; `train_and_save_on_tokens` trains real `ExtraTrees`/`RandomForest` models on a 4-sample fixture and really writes (gzip-detected) model files to `tmp_path`. `build_amount_tokens` asserts the exact 607-token shape (100 days x 6 variants + 7 quantity words).

### lexnlp/nlp/en/segments/titles.py
- **Tests added:** 3 in lexnlp/nlp/en/segments/tests/test_titles_coverage.py
- **Lines now covered:** 165, 166, 167, 170, 171, 172, 175, 177, 180, 181, 184, 185, 188, 189, 190, 192, 194, 195, 197, 198, 199, 202, 203, 204, 207, 208, 211, 212, 215, 256 (all listed lines; remaining misses 70-71, 87, 249, 252-253 belong to other tests' scope)
- **Unreachable:** none
- **Notes:** `build_model` is tested with mocked `requests.get`/`pandas.read_csv`/`joblib.dump` but a real `ExtraTreesClassifier.fit` on real `build_document_title_features` output. Fixture covers all branches: null line number filtered, `1-3` range skipped (`continue`), blank target line ignored, GitHub→raw URL rewriting asserted. Trailing-title yield (line 256) uses a stubbed `SECTION_SEGMENTER_MODEL`.

### scripts/segmentation_quality_gate.py
- **Tests added:** 42 in scripts/tests/test_segmentation_quality_gate_coverage.py
- **Lines now covered:** 22, 68, 69, 75, 76, 152, 155, 160, 165, 167, 175, 180, 182, 187, 199, 201, 205, 216, 220, 226, 287, 291, 293, 324, 597, 624, 628, 634, 640, 645 (module now 100% in combined run with existing tests)
- **Unreachable:** none (line 644-645 `__main__` guard covered via `runpy.run_path(..., run_name="__main__")` with `--help` argv)
- **Notes:** Line 22 (`sys.path` insert) covered by stripping the root from `sys.path` and `importlib.reload`-ing the module. `main([])` full-gate run and `--output` file run included; existing tests untouched and still passing.

### lexnlp/nlp/en/segments/payloads.py
- **Tests added:** 24 in lexnlp/nlp/en/segments/tests/test_payloads_coverage.py
- **Lines now covered:** 31, 37, 73, 84, 86, 92, 94, 98, 109, 114, 201, 211, 258, 260, 264, 280, 282 (module now 100% in combined run with existing tests)
- **Unreachable:** none
- **Notes:** Valid baselines are built with real `chunk_document`/`render_embedding_payload` output, then mutated via `dataclasses.replace` to hit each `EmbeddingPayload.__post_init__` guard. Heading-owner check (line 211) uses a real `PARAGRAPH` provenance reference. `PayloadBudgetExceeded` evidence (`actual`/`maximum`/`chunk_id`) asserted.

### lexnlp/nlp/train/train_data_manager.py
- **Tests added:** 7 in lexnlp/nlp/train/tests/test_train_data_manager_coverage.py (new `lexnlp/nlp/train/tests/__init__.py` created)
- **Lines now covered:** 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51 (module now 100%)
- **Unreachable:** none
- **Notes:** All tests use real files under `tmp_path`. Covers target-hit passthrough, alias copy with byte/content equality, non-matching alias skip, missing-source skip, first-usable-alias ordering with nested prefixes, and a mixed batch. One iteration subtlety: `path.replace(alias, ...)` rewrites every occurrence, so alias prefixes must nest cleanly.

### lexnlp/extract/all_locales/geoentities.py
- **Tests added:** 11 in lexnlp/extract/all_locales/tests/test_geoentities_coverage.py
- **Lines now covered:** 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 17, 20, 48, 49 (module now 100%)
- **Unreachable:** none
- **Notes:** No mocks for extraction: uses the real `test_geoentities` CSV fixtures via `DictionaryEntry.load_entities_from_files`; "I live in Berlin." yields one `GeoAnnotation` at (10, 16) through `en`, `en_US`, `de`, and unknown-locale (`fr_FR`/`xx`) fallback paths. A `monkeypatch` spy test proves dispatch-table routing matches the direct `en`/`de` routines. ("Paris" does not match this fixture due to alias-length/ban-list filtering, so "Berlin" is the probe entity.)

### lexnlp/extract/en/contracts/contract_type_detector.py
- **Tests added:** 14 in lexnlp/extract/en/contracts/tests/test_contract_type_detector_coverage.py
- **Lines now covered:** 29, 30, 31, 49, 50, 52, 53, 55, 56, 57, 59, 63, 64, 65, 66, 71 (module now 100%)
- **Unreachable:** none
- **Notes:** `__init__` (lines 29-31) mocks only `joblib.load`/`Doc2Vec.load` (large model loads) against real temp files and asserts loaded sentinels plus paths. `detect_contract_type_vector` uses tiny fake RF/D2V models and asserts the descending-sorted `Series` and the forwarded token list. `process_document` asserts real output (`['hello', 'world', 'testing', 'foobar']`; stopwords/numbers dropped). Pre-existing `FutureWarning` from the source's positional `Series.__getitem__` (lines 52/55/56) surfaces in test output; source untouched per instructions.
