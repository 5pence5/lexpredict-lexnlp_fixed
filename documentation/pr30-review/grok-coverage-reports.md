# PR30 testgen (Grok batch 2)

### scripts/reexport_bundled_sklearn_models.py
- **Tests added:** 37 in scripts/tests/test_reexport_bundled_sklearn_models_coverage.py
- **Lines now covered:** 49, 52, 57, 63, 72, 81, 85–87, 89, 100, 102–105, 108, 114–117, 122, 126, 128–133, 162, 164–182, 186–192, 197–198, 201–202, 204–205, 221–222, 226–231, 233–236, 238–243, 245–247, 251–253, 255–257, 260–263, 265–266, 271–274, 278–279, 281–284, 286, 288–293, 298–302, 304–305, 310–312, 314–316, 318–322 (all listed statements except 326). Measured 165/166 statements (99%).
- **Unreachable:** 326 (`raise SystemExit(main(...))` under `if __name__ == "__main__":`). Importing the module evaluates the `if` as false; covering the body would require a runpy/`__main__` harness. Lines 223–224 (`except ImportError` after `import pandas`) and 258–259 (`except TypeError` on `vars(obj)`) are `pragma: no cover` (pandas is a hard runtime dep; `vars()` TypeError is not reachable after the `__dict__` guard).
- **Notes:** Existing `test_reexport_bundled_sklearn_models.py` only exercised `load_model` (imported from `lexnlp.ml.model_io`) and the pickle layered-zip rewrite. New tests drive CLI parse/`iter_paths`, addresses/joblib fallbacks, skops layered zip + `.part` cleanup, pandas Index sanitization, and `main()` pickle/skops/`--remove-legacy` paths with tiny fitted `LogisticRegression` artifacts (no bundled model files rewritten).

### scripts/reexport_contract_model.py
- **Tests added:** 20 in scripts/tests/test_reexport_contract_model_coverage.py
- **Lines now covered:** 4, 6–14, 16, 18–20, 23–24, 31–32, 37, 42, 47, 52, 57, 63, 69, 75, 81, 87, 95, 101, 104–106, 108–112, 115, 125–126, 143–144, 146–155, 158–159, 185, 191, 194, 208–210, 212, 214–216, 218–220, 222–223, 225, 227–228, 231, 234–235, 237–240, 254, 259–260, 262–266, 271–273, 277–281, 283–284, 286, 295, 297, 300 (all listed statements except 301). Measured 102/103 statements (99%).
- **Unreachable:** 301 (`raise SystemExit(main(sys.argv[1:]))` under `if __name__ == "__main__":`). Line 300 (`if __name__ == "__main__":`) is executed on import (condition false).
- **Notes:** `download_github_release` is mocked (network). `run_quality_gate` mocks `subprocess.run`. `ProbabilityPredictorIsContract` is stubbed so `main()` does not load the catalog classifier. `get_legacy_warning_messages` is exercised with a real local `python -c` probe on a tiny pickle (empty warning list) and a corrupt file (`CalledProcessError`).

### lexnlp/extract/en/entities/stanford_ner.py
- **Tests added:** 19 in lexnlp/extract/en/entities/tests/test_stanford_ner_coverage.py
- **Lines now covered:** 11–16, 20–22, 24, 26–29, 34–35, 41, 47, 50–53, 56, 66, 68, 71–73, 75–77, 79–80, 82–85, 88–91, 93, 96, 106, 108, 111–113, 115–117, 119–120, 122–125, 128–131, 133, 136, 146, 148, 151–153, 155–157, 159–160, 162–164, 167, 170–173, 175 (all 82 listed statements). Measured 82/82 statements (100%).
- **Unreachable:** none as statements. Branch coverage still reports 83→73 and 163→153 as partial (the `else` of `token[0] in [".", ","]` jumping back to the `for`); those lines themselves execute. The `" "` arm of the punctuation join (`token[0] not in string.punctuation`) is dead: it sits inside `if token[0] in [".", ","]`, and both characters are in `string.punctuation`.
- **Notes:** `StanfordNERTagger` and `get_tokens_list` are mocked (Java/NER model load). `get_sentence_list` is real. A module-scoped autouse fixture reloads the module after this file so later Stanford tests see the real import-time tagger init. The extractor tags `get_tokens_list(text)` rather than the current sentence (existing source behaviour); tests assert that.

### lexnlp/extract/common/copyrights/copyright_parsing_methods.py
- **Tests added:** 17 in lexnlp/extract/common/copyrights/tests/test_copyright_parsing_methods_coverage.py (plus tests/__init__.py)
- **Lines now covered:** 3–8, 12, 14–17, 20–27, 29, 31, 33–36, 39, 43, 48–49, 51–52, 60–61, 63, 68–69, 71–72, 76–77, 79, 82–84, 87–92, 95–98, 100, 102, 105–115, 119–123, 125–127, 129 (all 76 listed statements). Measured 76/76 statements (100%, including branches).
- **Unreachable:** none
- **Notes:** `CopyrightParsingMethods.init_trigger_words` is a no-op; tests use a subclass with `trigger_words = r"copyright|©"`. Company strings are asserted as the parser actually returns them (`"Siemen"`, `"Acm"`, `"Siemens A"`, trailing spaces) — `get_company_name_from_match` truncates via `text[:-1]` on the `end == 0` / `start == len(text) - 1` reset and via `[\p{L}\s]+` letter runs.

# PR30 testgen (Grok batch 4)

### lexnlp/ml/catalog/download.py
- **Tests added:** 28 in lexnlp/ml/catalog/tests/test_download_coverage.py (plus tests/__init__.py)
- **Lines now covered:** 44–47, 85–87, 101, 110–113, 117–118, 127–128, 132, 135–136, 158, 219–221, 239–240, 242–244, 273–278, 280, 282–284, 286–288, 290–295, 298–305, 307–308, 310–312, 314, 317–318 (all listed statements). Isolated new-test run: 168/301 statements (56%); the remaining misses are download/network helpers already exercised by `test_download_security.py` / `test_download_path_normalization.py`.
- **Unreachable:** none of the listed statements. 133–134 and 297 are blank lines, not statements.
- **Notes:** Network is not used. Tests construct a real `requests.Session` via `build_retry_session` / `_session`, load JSON manifests from temp files, and verify local bytes with `_verify_file` / `verify_trusted_asset_file`. Legacy `MODELS_REPO` without a trailing slash is the 158 branch the older security test did not hit.

### lexnlp/extract/ml/en/definitions/layered_definition_detector.py
- **Tests added:** 17 in lexnlp/extract/ml/en/definitions/tests/test_layered_definition_detector_coverage.py
- **Lines now covered:** 59, 68, 78, 101, 103, 106, 108, 140–142, 144–145, 147–155, 163, 167–168, 170–172, 182–185, 187–189, 191–192, 194, 198, 202–206, 208–209, 213–216, 218–226, 229–235, 237 (all 65 listed statements). Measured 145/145 statements (100%).
- **Unreachable:** none
- **Notes:** Phrase/term detectors are stubbed (`load`, `predict_text`, `train_and_save_on_dataframe`); zip load/save and pandas/jsonl splitting are real. `get_annotations` pairs a term with `defs[0] == (definition_index, distance)` and then uses that pair as coords (`min`/`max`) — tests assert that existing behaviour, not the intended start/end of the definition span. Training writes dummy member bytes and checks the zip namelist.

### lexnlp/nlp/en/segments/hierarchy.py
- **Tests added:** 44 in lexnlp/nlp/en/segments/tests/test_hierarchy_coverage.py (plus tests/__init__.py)
- **Lines now covered:** 54–56, 61, 63, 69, 72–73, 77, 80, 82, 111, 128, 154, 159–160, 162, 171, 179, 181, 223, 240–241, 249, 280, 282, 284, 291, 303, 307, 310, 325, 327, 329, 336–337, 350, 542–544, 570, 573, 936, 939, 989, 992, 1012, 1030, 1106, 1111, 1115, 1132, 1149, 1218, 1322, 1345, 1432, 1443
- **Unreachable:** 295 (`node.end > len(source)` cannot fire after the root-span check at 282 and the child-exceeds-parent check at 307). 339 (`reconstruct() != source` cannot fire after `_validate_hierarchy` requires an exact partition). 447–448 (leftover after `splitlines(keepends=True)`; CPython 3.13 always consumes the whole string). 817 (`active_sections.pop()` while admitting a new section — the line-based pop at 808–809 already removed any section with `end <= section.start`, because `section.start <= line.start`). 934 (same pattern in `_table_spans`: the table-start pop already removed containers with `end <= start` before the candidate-start pop).
- **Notes:** Custom paragraph/sentence backends avoid the NLTK sentence model. Depth-limit coverage needs distinct `(kind, start, end)` identities; a same-span wrapper chain hits duplicate-id before `MAX_HIERARCHY_DEPTH`. `get_annotations`-style pairing is not involved here. `iter_document_segments` was previously unused by the suite.

### lexnlp/extract/de/dates_de_classifier.py
- **Tests added:** 7 in lexnlp/extract/de/tests/test_dates_de_classifier_coverage.py
- **Lines now covered:** 50, 65, 67–69, 71–72, 75, 78–80, 82–83, 94–95, 99, 522, 526–544, 547–548, 551–552, 554–563, 565–568, 577–578, 580, 583 (all listed statements except 582). Measured 83/84 statements (99%).
- **Unreachable:** 581–582 (`return date_str + "ten"` under `if num < 20`). `WRITTEN_DATE_NUMS` already handles 1–19, so `get_written_date_num` only reaches `num2words` for `num >= 20`, which always takes the `"sten"` suffix.
- **Notes:** `build_date_model` is mocked (sklearn training / large model write). `make_date_samples`, `add_numeric_date_samples`, `setup_date_parser`, and the `parse_dates` lambda inside `train_default_model` are real. `save=True` is redirected at `MODULE_PATH` so the packaged `date_model.pickle` is not overwritten. `random.random` is patched to 0.0 when generating numeric samples so Feb 30 hits the `ValueError` skip.

# PR30 testgen (Grok batch 6)

### scripts/unify_py_file_structure.py
- **Tests added:** 10 in scripts/tests/test_unify_py_file_structure_coverage.py
- **Lines now covered:** 9, 10, 11, 13, 23, 25, 32, 34, 36, 37, 40, 41, 42, 44, 45, 46, 51, 53, 54, 55, 57, 59, 60, 62, 63, 64, 65, 69, 85, 86, 89, 90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 105 (all listed statements except 67). Measured 42/43 statements (95%).
- **Unreachable:** 67 (`service = imports = code = docstr = ""` in the `fullmatch` else). Every other group is optional and `(?P<code>.*)` with `re.S` always consumes the rest of the file, so `fullmatch` cannot fail.
- **Notes:** `__file__` / `parse_paths` / `exclude_paths` are pointed at a tmp tree so the real package is never rewritten. `__main__` is exercised via `runpy.run_path` with `os.walk` stubbed to an empty iterator. The module-level `author` string is restored after each test.

### lexnlp/extract/ml/detector/sample_processor.py
- **Tests added:** 12 in lexnlp/extract/ml/detector/tests/test_sample_processor_coverage.py
- **Lines now covered:** 19, 20, 21, 22, 26, 57, 58, 59, 60, 61, 64, 66, 67, 69, 72, 73, 74, 77, 78, 81, 82, 83, 84, 85, 86, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 104, 105, 106 (all 40 listed statements). Measured 54/54 statements (100%, including branches).
- **Unreachable:** none
- **Notes:** Classifier is a tiny duck-typed double (`get_feature_data` + `feature_list`) so tests can pin token spans for start/inner/end/outer classes and force the preallocation resize (`pre_alloc_multiple=1` with a 1-token row then a 3-token row). Pandas/numpy are real. Resize on row index 0 would divide by zero in the production formula; tests use the default RangeIndex so overflow happens on row 1.

### lexnlp/nlp/en/stanford.py
- **Tests added:** 19 in lexnlp/nlp/en/tests/test_stanford_coverage.py
- **Lines now covered:** 34, 35, 43, 44, 49, 50, 78, 79, 80, 81, 82, 83, 85, 87, 88, 89, 91, 103, 106, 107, 109, 110, 111, 112, 113, 115, 116, 128, 131, 132, 134, 135, 136, 137, 138, 140, 141 (all 37 listed statements). Measured 64/64 statements (100%, including branches).
- **Unreachable:** none
- **Notes:** `StanfordTokenizer` / `StanfordPOSTagger` are faked (Java jar load). `is_stanford_enabled` is patched per test. `get_lemma_list` is stubbed only on the lemmatize paths so lemma indices match the fake tokenizer. A module-scoped autouse fixture reloads the module after this file so later Stanford tests see the real import-time init.

### lexnlp/extract/en/dict_entities.py
- **Tests added:** 23 in lexnlp/extract/en/tests/test_dict_entities_coverage.py
- **Lines now covered:** 64, 65, 66, 84, 85, 102, 104, 107, 108, 109, 110, 111, 112, 144, 236, 237, 264, 265, 266, 344, 345, 346, 452, 455, 482, 511, 512, 513, 514, 515, 516, 517, 518, 519 (all 34 listed statements). Isolated new-test run still misses loaders / `find_dict_entities` already covered by `test_dict_entities.py`.
- **Unreachable:** none of the listed statements
- **Notes:** `_find_entity_positions` is called directly for the empty-alias / `context is None` / banlist-skip branches (the function does not return the local context it allocates). `has_closer_locale` empty-language vs ordered-list cases are unit-tested on `DictionaryEntryAlias`. Simple-tokenization `normalize_text_with_map` uses already-lowercase input so `PhrasePositionFinder` can locate the split tokens.
