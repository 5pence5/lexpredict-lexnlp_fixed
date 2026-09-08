# PR30 testgen (Muse batch 42)

### lexnlp/extract/pt/ratios.py
- **Tests added:** 3 in lexnlp/extract/pt/tests/test_ratios_coverage.py
- **Lines now covered:** [77] (`yield ant.left, ant.right, ant.ratio` — the `return_sources=False` branch of `get_ratios`, exercised via `get_ratios(...)`/`get_ratio_list(...)` defaults asserting exact 3-tuples, e.g. `(Decimal("3"), Decimal("1"), Decimal("3"))` for `"3:1"`)
- **Unreachable:** none
- **Notes:** pytest-cov (`--cov=lexnlp.extract.pt.ratios`) cannot collect the pt package in this env (ImportError: `numpy: cannot load module more than once per process` via `pt/__init__` → `courts` → `pandas`); line 77 is the sole statement of the `else` branch so the passing default-branch tests necessarily execute it.

### lexnlp/extract/pt/trademarks.py
- **Tests added:** 0 (no file created — line genuinely unreachable, see below)
- **Lines now covered:** none
- **Unreachable:** [60] (`coords = (coords[0], len(text))`). `regex`/`re` `Match.span()` guarantees `0 <= start <= end <= len(string)`, so `coords[1] > len(text)` is always False on real `finditer` output (verified: spans `(0, 10)/27`, `(0, 6)/26`, `(0, 4)/8`, `(119, 201)/201`). Reaching it would require monkeypatching the compiled pattern to return a fake match — contortion, not real behaviour.
- **Notes:** Existing `test_trademarks.py` covers both sides of the surrounding guard (`search` hit/miss) and the normal `yield` path.

### lexnlp/utils/caching.py
- **Tests added:** 2 in lexnlp/utils/tests/test_caching_coverage.py
- **Lines now covered:** [42] (`return Path.home() / ".cache" / "lexnlp"` — `_default_cache_dir` fallback with `LEXNLP_CACHE_DIR` unset/empty; asserts equality with `Path.home() / ".cache" / "lexnlp"`)
- **Unreachable:** none
- **Notes:** Verified 100% (`lexnlp/utils/caching.py 28 0 100%`) with existing + new tests. New tests target `_default_cache_dir` directly to avoid interference from the `get_memory` `lru_cache` and the existing autouse env-override fixture.

### scripts/train_contract_type_model.py
- **Tests added:** 0 (no file created — line genuinely unreachable, see below)
- **Lines now covered:** none
- **Unreachable:** [177] (`raise SystemExit(main(sys.argv[1:]))` under `if __name__ == "__main__":`). Only executes when the file is run as a script; importing it (what the test suite does) never reaches it. Covering it would require a subprocess/runpy-as-`__main__` harness — contortion per the task rules.
- **Notes:** All other statements (incl. the stratify-fallback, `wrote_artifact` report, exit-code branches) are covered by `scripts/tests/test_train_contract_type_model.py`.

### lexnlp/extract/de/copyrights.py
- **Tests added:** 4 in lexnlp/extract/de/tests/test_copyrights_coverage.py
- **Lines now covered:** [96] (`yield ant.to_dictionary()` in `get_copyrights` — exercised via `get_copyrights("Copyright 2019, Siemens")` asserting the yielded dict's company/start/type tags, generator type, `get_copyright_list` equivalence, `return_sources=True`, and empty-text yields-nothing)
- **Unreachable:** none
- **Notes:** Existing tests only called `get_copyright_list("")` (empty loop body, line never runs) plus the annotation-level APIs, so the dict-generator body was the sole gap.

### lexnlp/extract/de/identifiers.py
- **Tests added:** 4 in lexnlp/extract/de/tests/test_identifiers_coverage.py
- **Lines now covered:** [141] (`sum_mod = 10` in `_ust_idnr_is_valid` when `(digit + product) % 10 == 0` — exercised with leading-zero body `037297049`, whose first iteration computes `(0 + 10) % 10 == 0`; asserts `_ust_idnr_is_valid` True, bad-check-digit variant False, and `get_ust_idnr_annotations` extraction incl. coords `(11, 22)` and spaced surface `DE 037 297 049`)
- **Unreachable:** none
- **Notes:** `DE037297049` is checksum-valid per the MOD 11 routine (found by brute-forcing the check digit over candidate leading-zero bodies), so the branch is hit on a real valid ID, not just an invalid one.

### lexnlp/extract/en/acts.py
- **Tests added:** 3 in lexnlp/extract/en/tests/test_acts_coverage.py
- **Lines now covered:** [62] (`return list(get_acts_annotations(text))` in `get_acts_annotations_list` — asserts `ActAnnotation` fields: act_name `VERY Important Act`, section `12`, year `1954`, coords `(5, 49)`; generator equivalence; empty/no-act inputs return `[]`)
- **Unreachable:** none
- **Notes:** Trivial one-line wrapper the existing tests never called (they use `get_act_list`/`get_acts_annotations`).

### lexnlp/extract/en/courts.py
- **Tests added:** 3 in lexnlp/extract/en/tests/test_courts_coverage.py
- **Lines now covered:** [97] (`return list(parser.parse(text, language))` in `get_court_annotation_list` — asserts `CourtAnnotation` type, court name `Supreme Court`, text containment, generator equivalence with `get_court_annotations`, and `[]` for no-court/empty inputs; uses the local parser only, no network)
- **Unreachable:** none
- **Notes:** Existing `test_courts.py` exercises only the deprecated `_get_courts` helper (which needs network CSVs) and never the module-level `get_court_annotation_list`.

### lexnlp/extract/all_locales/dates.py
- **Tests added:** 2 in lexnlp/extract/all_locales/tests/test_dates_coverage.py
- **Lines now covered:** [45] (English branch `yield from routine(...)` — real calls `get_date_annotations("en", "The meeting is on June 1, 2017.")` asserting date 2017-06-01, text, coords, plus an `en-US` call forwarding `strict`/`base_date`/`threshold`)
- **Unreachable:** none
- **Notes:** Coverage of `lexnlp.extract.all_locales.dates` is 100% with the new tests plus existing `test_dispatch.py`. No mocks; uses the bundled sklearn date model.

### lexnlp/extract/batch/async_extract.py
- **Tests added:** 2 in lexnlp/extract/batch/tests/test_async_extract_coverage.py
- **Lines now covered:** [130] (`raise` in the `except asyncio.CancelledError` handler of `_run_one` — extractor raising `CancelledError` must propagate rather than be captured, verified with both `raise_on_error=False` and `True`)
- **Unreachable:** none
- **Notes:** `except Exception` would not catch `CancelledError` (it is `BaseException` since 3.8), so the handler exists to keep TaskGroup structured concurrency intact. Note: `pytest --cov=lexnlp.extract.batch.async_extract` on this machine fails at collection with `ImportError: cannot load module more than once per process` from numpy under coverage's importer; plain pytest runs are unaffected, and line 130 execution was additionally confirmed with `sys.settrace`.

### lexnlp/extract/common/annotations/citation_annotation.py
- **Tests added:** 2 in lexnlp/extract/common/annotations/tests/test_citation_annotation_coverage.py
- **Lines now covered:** [112] (`df.tags["Extracted Entity Page Range"]` — annotation with `page_range="113-115"` yields the tag; contrast test with `page_range=None` asserts the tag is absent while `Page` remains)
- **Unreachable:** none
- **Notes:** None.

### lexnlp/extract/common/annotations/phrase_position_finder.py
- **Tests added:** 2 in lexnlp/extract/common/annotations/tests/test_phrase_position_finder_coverage.py
- **Lines now covered:** [57] (`break` when `start >= len(stoc)` — `find_phrase_in_source_text("ab", ["ab", "cd"])` leaves the unreached phrase as the `("cd", 0, 0)` sentinel; second test with `"hello world"` and three phrases pins the same early exit)
- **Unreachable:** none
- **Notes:** None.

### lexnlp/extract/en/entities/nltk_tokenizer.py
- **Tests added:** 3 in lexnlp/extract/en/entities/tests/test_nltk_tokenizer_coverage.py
- **Lines now covered:** [38, 39] (`convert_parentheses=True` loop over `CONVERT_PARENTHESES` — `"hello (world)"` tokenizes to `["hello", "-LRB-", "world", "-RRB-"]` vs unchanged parens by default; `return_str=True` case pins all six bracket conversions and asserts no literal paren remains)
- **Unreachable:** none
- **Notes:** None.

### lexnlp/extract/en/geoentities.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_geoentities_coverage.py
- **Lines now covered:** [100] (`get_geoentity_list` wrapper), [186] (`get_geoentity_annotation_list` wrapper) — `"I live in France."` yields `[("France", "France")]` and a `GeoAnnotation` with `coords == (10, 16)`, `name/alias == "France"`; both assert equality with the generator versions plus empty-text cases
- **Unreachable:** none
- **Notes:** Config loaded from the same `test_geoentities/geoentities.csv` + `geoaliases.csv` fixtures as the existing tests. No mocks.

### lexnlp/extract/en/money.py
- **Tests added:** 5 in lexnlp/extract/en/tests/test_money_coverage.py
- **Lines now covered:** [83] (`get_money_list`), [117] (`get_money_annotation_list`) — `"The cost is $25.50."` yields `[(Decimal("25.50"), "USD")]` (with source `"cost is $25.50"` when `return_sources=True`); annotation case pins `amount/currency/coords/text`; both assert equality with the generator versions plus empty cases
- **Unreachable:** none
- **Notes:** None.

### lexnlp/extract/en/ratios.py
- **Tests added:** 5 in lexnlp/extract/en/tests/test_ratios_coverage.py
- **Lines now covered:** [54] (`get_ratio_list`), [80] (`get_ratio_annotation_list`) — `"The ratio is 3 to 1."` yields `[(Decimal("3.0"), Decimal("1.0"), Decimal("3"))]` (with source `"3 to 1."` when `return_sources=True`); annotation case pins `left/right/ratio/coords == (13, 20)`; both assert equality with the generator versions plus empty cases
- **Unreachable:** none
- **Notes:** Coverage-only run shows lines 66/70 (invalid-amount and zero-amount `continue` guards) uncovered, but those are exercised by the existing data-driven `test_ratios.py` suite, not this batch's listed lines.

### lexnlp/extract/common/annotations/url_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_url_annotation_coverage.py
- **Lines now covered:** [37, 38] (`get_dictionary_values` body — exact `{"tags": {"Extracted Entity URL", "Extracted Entity Text"}}` dict asserted, incl. `None`-url case, plus `to_dictionary()` merge and `get_cite_value_parts`)
- **Unreachable:** none
- **Notes:** New-tests-only coverage run is 100% (17/17 stmts).

### lexnlp/extract/common/durations/durations_parser.py
- **Tests added:** 3 in lexnlp/extract/common/durations/tests/test_durations_parser_coverage.py (new sibling `tests/` dir with `__init__.py`)
- **Lines now covered:** [75] (`value_dict[type] += float(amount)` — `sum_annotations` over two same-type (`year`/`year`) annotations asserting accumulated `{"year": 8.0}`, summed days `2920`, spanning coords, `is_complex`; distinct-type contrast test pins the `else` branch), [109] (`raise NotImplementedError()` in base `get_all_annotations`, asserted via `assertRaises`)
- **Unreachable:** none
- **Notes:** Same-type grouping cannot arise via `EnDurationParser.get_annotations` (grouping requires strictly decreasing timeframes), so line 75 is reached by calling `sum_annotations` directly with two same-type annotations — real behaviour of that method, no mocks.

### lexnlp/extract/common/ocr_rating/lang_vector_distribution_builder.py
- **Tests added:** 5 in lexnlp/extract/common/ocr_rating/tests/test_lang_vector_distribution_builder_coverage.py (new sibling `tests/` dir with `__init__.py`)
- **Lines now covered:** [27] (`continue` for empty/short texts — mixed short+long build equals long-only build key-by-key), [32] (`return None` — empty iterable, all-short (`""`, `"short"`, 99-char) inputs, and a short-only temp file via `build_files_reference_distribution` all assert `None`)
- **Unreachable:** none
- **Notes:** New-tests-only coverage run is 100% (33/33 stmts). Long-text case additionally asserts the distribution is normalized (`sum ~= 1.0`, all non-negative). No mocks; real n-gram computation.

### lexnlp/extract/de/citations.py
- **Tests added:** 5 in lexnlp/extract/de/tests/test_citations_coverage.py
- **Lines now covered:** [103] (`ant.volume = volume` — `"Artikel 2 Nr. 1 des Gesetzes vom 2. Januar 2002 (BGBl. I S. 2477)"` yields `volume == 1`, `volume_str is None`, dict `number == "1"`; `Nummer 5` variant pins `volume == 5`), [122] (`get_citation_annotation_list` wrapper — asserts equality with `list(get_citation_annotations(...))` on coords/text/article/volume plus `[]` for no-citation/empty inputs)
- **Unreachable:** none
- **Notes:** Combined existing + new run is 100% (59/59 stmts). Existing `test_parts` uses number `"1 bis 3"` which fails `int()` and takes the `volume_str` branch, so the single-int `Nr.` form is the real input that hits line 103.

### lexnlp/extract/common/annotations/condition_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_condition_annotation_coverage.py
- **Lines now covered:** [50, 57] (`get_dictionary_values` def + `return df` — exercised via exact-dict assertions for full and all-None fields, `to_dictionary` tag-merge, and `get_cite_value_parts`)
- **Unreachable:** none
- **Notes:** Combined annotations-tests run with the new files is 100% on this module.

### lexnlp/extract/common/annotations/constraint_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_constraint_annotation_coverage.py
- **Lines now covered:** [50, 57] (`get_dictionary_values` def + `return df` — exact-dict assertions for full and all-None fields, `to_dictionary` tag-merge, `get_cite_value_parts`)
- **Unreachable:** none
- **Notes:** Combined annotations-tests run with the new files is 100% on this module.

### lexnlp/extract/common/annotations/date_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_date_annotation_coverage.py
- **Lines now covered:** [48, 49] (`get_dictionary_values` def + `return df` — with-date, no-date (empty-string date), and default (None text) variants plus `to_dictionary` merge)
- **Unreachable:** none
- **Notes:** Combined annotations-tests run with the new files is 100% on this module.

### lexnlp/extract/common/annotations/distance_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_distance_annotation_coverage.py
- **Lines now covered:** [54, 55] (`get_dictionary_values` def + `return df` — full amount/text, default empty/None variants, `to_dictionary` merge, `get_cite_value_parts`)
- **Unreachable:** none
- **Notes:** Combined annotations-tests run with the new files is 100% on this module.

### lexnlp/extract/pt/identifiers.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_identifiers_coverage.py
- **Lines now covered:** [80] (`to_dictionary` return — exact-dict assertions for CPF `(4, 18)` and OAB `SP/123456` `(34, 48)` incl. record_type/coords/text/value/locale), [136] (CPF first-check-digit `return False` — `"52998224735"` fails first check where existing `"...724"` test fails the second; validator False + extractor yields `[]` with valid-contrast `529.982.247-25`), [171] (CNPJ first-check-digit `return False` — `"11222333000171"` vs valid `"11222333000181"`, same validator + extractor contrast)
- **Unreachable:** none
- **Notes:** pytest-cov cannot collect the pt package in this env (ImportError: `numpy: cannot load module more than once per process` via `pt/__init__` → `courts` → `pandas`, same as the earlier pt batch); line execution was verified with `sys.settrace` — new-tests-only run hits all of [80, 136, 171].

### lexnlp/extract/pt/pii.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_pii_coverage.py
- **Lines now covered:** [100] (`continue` length filter — `"+55 011 91234-5678"` and `"032 11 91234-5678"` are proven regex hits (14 digits via `PHONE_PTN_RE.search`) that yield no annotations), [112] (`yield ant.phone` in `get_phones` — `"Contato: (11) 98765-4321 ..."` yields `["11987654321"]`, pinned equal to `get_phone_list`), [144] (`yield from get_phone_annotations` — `get_pii_annotations` equals `get_phone_annotation_list` on phone/text/coords/locale, plus empty-text `[]`)
- **Unreachable:** none
- **Notes:** Existing phone tests only used `get_phone_annotation_list`/`get_phone_list` on empty text for the `get_phones` path, so the non-empty `get_phones` body and `get_pii_annotations` were never entered. Same pytest-cov/pt collection caveat as above; `sys.settrace` confirms new-tests-only hits [100, 112, 144].

### lexnlp/ml/normalizers.py
- **Tests added:** 5 in lexnlp/ml/tests/test_normalizers_coverage.py
- **Lines now covered:** [47] (`self.normalizations = normalizations` — identity assertion on the stored iterable), [73, 74] (`except Exception: yield text` fallback — raising extractor returns the original text via both `__call__` and direct `_find_replace`; contrast tests pin the happy-path replacement `" __X__  world"` and the no-annotation passthrough)
- **Unreachable:** none
- **Notes:** Verified 100% (`lexnlp/ml/normalizers.py 36 0 100%`) with new tests alone. No mocks; uses a minimal `TextAnnotation` extractor.

### lexnlp/utils/lines_processing/parsed_text_corrector.py
- **Tests added:** 2 in lexnlp/utils/lines_processing/tests/test_parsed_text_corrector_coverage.py
- **Lines now covered:** [53, 54, 55] (`if extra_line_breaks_prob > 50` + correction + return — docstring corrupted text estimates corrupted 66/extra 66; asserts output differs, is shorter, `"\n\n"` count drops 3 → 1 with the header break kept, and equality with `correct_line_breaks(text)`; clean-text contrast test pins the early-return branch)
- **Unreachable:** none
- **Notes:** Verified 100% (`parsed_text_corrector.py 34 0 100%`) with new + existing `test_parsed_text_corrector.py`. Note `corrupted_prob` always equals `extra_line_breaks_prob` (assigned in `estimate_text`), so the `53-False/51-False` combination only occurs at exactly 50.

### lexnlp/extract/de/percents.py
- **Tests added:** 5 in lexnlp/extract/de/tests/test_percents_coverage.py
- **Lines now covered:** [65] (`continue` when the percent-pattern match yields no single amount — `"1,2,3 Prozent"` matches but `get_amounts("1,2,3 ")` returns `[]`, asserting both annotation and dict APIs yield `[]` with a `"15 Prozent"` control), [85] (`get_percent_list` wrapper — exact dict incl. `amount == Decimal("15.0")`, `real_amount == Decimal("15.0000")`, coords `(0, 10)`), [92] (`get_percent_annotation_list` wrapper — `PercentAnnotation` sign/coords/locale plus generator equivalence)
- **Unreachable:** none
- **Notes:** Verified 100% (`de/percents.py 38 0 100%`) with existing + new tests. Note `real_amount` is `round(amount)` rather than `round(amount * 0.01)` (line 72 requantizes `amount`, not the scaled value) — tests pin the actual behaviour.

### lexnlp/extract/en/citation_variations.py
- **Tests added:** 3 in lexnlp/extract/en/tests/test_citation_variations_coverage.py
- **Lines now covered:** [59, 60] (`elif isinstance(value, str)` branch of `variation_map` — all shipped `VARIATIONS_ONLY` values are lists, so the test patches `_VARIATIONS_RAW` to `{"FakeVariant": "FakeCanon"}` with both `lru_cache`s cleared/restored, asserting `== {"FakeVariant": ("FakeCanon",)}` in both `variation_map()` and the normalized index), [110] (`return name` for `name in EDITIONS` in `normalize_reporter` — first `EDITIONS` key round-trips unchanged)
- **Unreachable:** none
- **Notes:** Verified 100% (`citation_variations.py 42 0 100%`) with existing + new tests. The str-branch mock is the minimal possible (one-entry raw map, caches restored in `finally`); no network.

### lexnlp/extract/en/cusip.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_cusip_coverage.py
- **Lines now covered:** [68] (`return False` for a char that is neither digit nor in `CHECKSUM_BASE` — `is_cusip_valid("12a456781") is False` with a valid-CUSIP control), [74] (`return checksum` — `is_cusip_valid("837649128", return_checksum=True) == 8`, plus `"392690QT3" -> 3`), [129] (`get_cusip_annotation_list` wrapper — full field pin incl. `coords == (8, 17)` plus generator equivalence and empty-text case)
- **Unreachable:** none
- **Notes:** Verified 100% (`cusip.py 65 0 100%`) with existing + new tests.

### lexnlp/extract/en/percents.py
- **Tests added:** 7 in lexnlp/extract/en/tests/test_percents_coverage.py
- **Lines now covered:** [81] (`get_percent_list` wrapper — exact `("percent", Decimal("5.0"), Decimal("0.050"))` tuples with and without `return_sources`, plus generator equivalence), [104] (`continue` when the match parses to neither a single amount nor a single ratio — `"1,2,3 percent"` yields `[]` with a `"5 percent"` control), [121] (`get_percent_annotation_list` wrapper — amount/fraction/sign/coords pin plus generator equivalence)
- **Unreachable:** none
- **Notes:** Verified 100% (`en/percents.py 44 0 100%`) with existing + new tests. Also added a ratio-fallback test (`"1/2 percent"` → amount `0.5`, fraction `0.005`) which covers the adjacent line 102 in subset runs (line 102 is covered by the full data-driven suite, but the subset used for verification needed it).

### lexnlp/extract/en/citations.py
- **Tests added:** 5 in lexnlp/extract/en/tests/test_citations_coverage.py
- **Lines now covered:** [92] (`get_citation_list` wrapper — exact `(410, "U.S.", "United States Supreme Court Reports", 113, None, None, 1973)` tuple plus `return_source`/`as_dict` variants), [140, 141] (`except KeyError: pass` — lowercase `"410 u.s. 113"` matches the case-insensitive pattern but `EDITIONS["u.s."]` raises KeyError since dict lookup is case-sensitive; assert it yields `[]` while the uppercase form yields one citation, plus a mixed line where only the valid citation survives), [154] (`get_citation_annotation_list` wrapper — full `CitationAnnotation` field pin plus generator equivalence and empty-text case)
- **Unreachable:** none
- **Notes:** Verified 100% (`citations.py 43 0 100%`) with existing + new tests. `CitationAnnotation` has no `__eq__`, so generator/wrapper equivalence is asserted via field tuples, not list equality. Note the year-paren group `(.+?)?(\d{4})` is DOTALL-greedy across citations, so mixed-case tests must separate the bad citation without a year paren (`"X 410 u.s. 113. Y 410 U.S. 113 (1973) Z"`); two adjacent `(1973)` citations merge into one `u.s.` match and yield `[]`.

### lexnlp/extract/es/courts.py
- **Tests added:** 4 in lexnlp/extract/es/tests/test_courts_coverage.py
- **Lines now covered:** [60, 61] (`warnings.warn(...)` + `find_dict_entities` loop head in deprecated `_get_courts` — asserted via `pytest.warns(DeprecationWarning, match="removed in a future version")` on a real `DictionaryEntry(id=1, name="Tribunal Superior de Justicia de Madrid")` match), [69] (`yield ent.entity` — asserts the yielded `(entry, alias)` tuple fields), [91] (`get_court_annotation_list` wrapper — asserts `CourtAnnotation` name `Tribunal Superior de Justicia de Madrid`, locale `es`, coords `(9, 49)`, generator equivalence, empty/no-court inputs)
- **Unreachable:** none
- **Notes:** New-file-only coverage leaves just `get_courts`/`get_court_list` bodies (lines 95-96, 100), which the existing `test_courts.py` covers. pytest-cov (`--cov=...`) cannot collect this module in this env (`numpy: cannot load module more than once per process`); verified with `coverage run -m pytest <new-file>` instead, which shows only the pre-existing-test lines missing.

### lexnlp/extract/pt/courts.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_courts_coverage.py
- **Lines now covered:** [58] (`warnings.warn(..., stacklevel=2)` in deprecated `_get_courts` — asserted via `pytest.warns(DeprecationWarning, match="removed in a future version")` on a real `DictionaryEntry(id=1, name="Supremo Tribunal Federal")` match), [63, 71] (`find_dict_entities` loop + `yield ent.entity` — asserts yielded `(entry, alias)` fields, plus `priority`/`text_languages`/`simplified_normalization` flags and no-match-`[]`), [124] (`get_court_annotation_list` wrapper — asserts `CourtAnnotation` name `Supremo Tribunal Federal`, locale `pt`, coords `(12, 37)`, generator equivalence, empty/no-court inputs)
- **Unreachable:** none
- **Notes:** New-file-only coverage leaves just `get_courts`/`get_court_list` bodies (lines 134-135, 149), covered by existing `test_courts.py`. Same pytest-cov collection caveat as es/courts — verified via `coverage run -m pytest <new-file>`.

### lexnlp/utils/map.py
- **Tests added:** 6 in lexnlp/utils/tests/test_map_coverage.py
- **Lines now covered:** [34] (`__getattr__` fallback `return self.get(attr)` — missing attr returns `None` and stays out of `__dict__`; existing `test_map.py` never hit it because `__setitem__` mirrors keys into `__dict__`, so normal attribute reads bypass `__getattr__`), [44] (`__delattr__` → `__delitem__` — `del m.b` removes key and `__dict__` entry, subsequent `m.b is None`; missing attr raises `KeyError`), [47, 48] (`__delitem__` — `del m["a"]` removes key and `__dict__` entry; missing key raises `KeyError`)
- **Unreachable:** none
- **Notes:** Verified 100% (`map.py 33 0 100%`) with existing + new tests. Also added a `Map(a=1, b=2)` kwargs-init test (lines 23-24) so the new file is robust standalone.

### lexnlp/nlp/en/tokens.py
- **Tests added:** 9 in lexnlp/nlp/en/tests/test_tokens_coverage.py
- **Lines now covered:** [68] (`text = text.lower()` in `get_tokens_by_regex` — asserted via `get_tokens_by_regex("Hello WORLD Foo", lowercase=True)` returning `["hello", "world", "foo"]` vs case-preserving default), [75, 76, 77] (`wrd.strip(".")` + drop-if-empty under `preserve_line=False` — asserted via `"Hello . World"` yielding `["Hello", "World"]` vs `["Hello", ".", "World"]` with `preserve_line=True`, and `"..."` yielding `[]` vs `[".", ".", "."]`), [95] (`yield token.lower()` in `get_tokens` with `stopword=True, lowercase=True` — asserted via `"This is a Test."` returning `["test", "."]` vs `["Test", "."]` without lowercase)
- **Unreachable:** none
- **Notes:** Verified 100% (`lexnlp/nlp/en/tokens.py 122 0 100%`) with existing `test_tokens.py` + new file.

### lexnlp/extract/batch/pandas_output.py
- **Tests added:** 4 in lexnlp/extract/batch/tests/test_pandas_output_coverage.py
- **Lines now covered:** [135, 136] (`return frame.convert_dtypes(dtype_backend="pyarrow")` — asserted via stubbed `pyarrow` in `sys.modules` plus monkeypatched `DataFrame.convert_dtypes` returning a sentinel and recording `dtype_backend="pyarrow"`, both directly and end-to-end through `annotations_to_dataframe`), [137, 138] (`except TypeError: return frame` old-pandas fallback — asserted via `convert_dtypes` raising `TypeError`, returning the identical original frame with values intact, plus an end-to-end `annotations_to_dataframe(..., prefer_arrow=True)` row-preservation test)
- **Unreachable:** none
- **Notes:** `pyarrow` is not installed in this env, so the import is stubbed with `types.ModuleType("pyarrow")` in `sys.modules` (env limitation, not app logic) and `convert_dtypes` is monkeypatched to emulate new/old pandas. Verified 100% (`pandas_output.py 42 0 100%`) with existing `test_pandas_output.py` + `test_pandas_extras.py` + new file via `coverage run -m pytest` (pytest-cov `--cov` cannot collect pandas tests here: `ImportError: numpy: cannot load module more than once per process`).

### lexnlp/extract/common/annotations/geo_annotation.py
- **Tests added:** 3 in lexnlp/extract/common/annotations/tests/test_geo_annotation_coverage.py
- **Lines now covered:** [71, 72, 73, 74] (`get_dictionary_values` — asserted via `GeoAnnotation(..., name="Berlin", text="Berlin", year=2020)` tags equal to `{"Extracted Entity Name": "Berlin", "Extracted Entity Text": "Berlin", "year": 2020}`, no-year variant having exactly the two base keys, and falsy `year=0` omitting the `year` key per the `if self.year:` guard)
- **Unreachable:** none
- **Notes:** Remaining misses in the combined annotations run are lines 67-68 (`get_cite_value_parts`), outside this batch's scope.

### lexnlp/extract/common/annotations/percent_annotation.py
- **Tests added:** 3 in lexnlp/extract/common/annotations/tests/test_percent_annotation_coverage.py
- **Lines now covered:** [52, 53, 54, 55] (`get_dictionary_values` — asserted via `PercentAnnotation(..., amount=Decimal("5"), sign="+")` tags equal to `{"Extracted Entity Value": "5", "Extracted Entity Text": "5%", "sign": "+"}`, no-sign variant having exactly the two base keys, and amount-`None` rendering `"Extracted Entity Value": ""`)
- **Unreachable:** none
- **Notes:** Remaining miss in the combined annotations run is line 49 (`get_cite_value_parts`), outside this batch's scope.

### lexnlp/extract/en/definitions.py
- **Tests added:** 8 in lexnlp/extract/en/tests/test_definitions_coverage.py
- **Lines now covered:** [53] (`raise` in `get_definition_annotations` ML-uninitialized branch, via toggling `parser_ml_classifier.initialized=False` and asserting `pytest.raises(Exception, match="should be initialized")`), [83, 84] (same guard in `get_definitions`, same technique), [85] (`parser_ml_classifier.get_annotations(text)` ML success path, asserted on `"Consolidated EBITDA"` text yielding `'"Consolidated EBITDA"'` names), [93] (`elif return_sources` yield, asserted via `get_definitions(TEXT, return_sources=True) == [("Consolidated EBITDA", <sentence>)]` plus `return_coords`-precedence check `(name, text, (1, 20))`)
- **Unreachable:** none
- **Notes:** Verified 100% (`definitions.py 52 0 100%`) with existing `test_definitions.py` + new file. ML success tests assert structural properties (name contains `Consolidated EBITDA`, annotation coords slice equals annotation text) rather than exact model scores to stay robust across sklearn unpickle versions. Global `parser_ml_classifier` flag is toggled with try/finally so no state leaks.

### lexnlp/extract/ml/detector/detecting_settings.py
- **Tests added:** 4 in lexnlp/extract/ml/detector/tests/test_detecting_settings_coverage.py
- **Lines now covered:** [19, 20, 21, 22] (`__init__` assignments, asserted via default `DetectingSettings()` equals `(False, 0, 0, "random_forest")` and explicit `(True, 1, 2, "extra_trees")` round-trip), [25] (`__repr__`, asserted exactly: `"use_spacy=False, pre_window=0, post_window=0, model_type=random_forest"` and the explicit-args variant)
- **Unreachable:** none
- **Notes:** Verified 100% (`detecting_settings.py 14 0 100%`) standalone.

### lexnlp/extract/ml/en/definitions/definition_phrase_detector.py
- **Tests added:** 5 in lexnlp/extract/ml/en/definitions/tests/test_definition_phrase_detector_coverage.py
- **Lines now covered:** [30] (`process_sample` delegation, asserted via real `TokenSequenceClassifierModel` + 1-row frame yielding `(6, N)` features and start/inner/end targets `{1.0, 2.0, 3.0}` plus a feature-mask propagation check on the `mask` column), [47, 49] (`train_and_save` CSV read + delegation, asserted via real temp CSV with `patch.object(detector, "train_and_save_on_dataframe")` capturing the 2-row frame and `train_size=1` head-limit variant), [59, 84] (`train_and_save_on_dataframe` def-tokens + `train_and_save_on_tokens` call, asserted via real 2-row training to `tmp_path` with `RandomForest`, checking `punc_set == ':"()"'`, `shall`/`mean` in `feature_list`, saved-file nonzero, and `predict()` shape agreement)
- **Unreachable:** none
- **Notes:** Verified 100% for both definition detectors together (`definition_phrase_detector.py 20 0 100%`) with the layered-detector test + new files. `train_and_save` downstream training is mocked (CSV string columns cannot carry list-valued `labels`/`feature_mask`); real training is covered separately in `train_and_save_on_dataframe`.

### lexnlp/extract/ml/en/definitions/definition_term_detector.py
- **Tests added:** 5 in lexnlp/extract/ml/en/definitions/tests/test_definition_term_detector_coverage.py
- **Lines now covered:** [21] (`process_sample` delegation, asserted via real classifier + `{"sentence": "The Employment Period shall mean foo", "labels": [(4, 21)]}` yielding 6 tokens with targets `[0.0, 1.0, 3.0, 0.0, ...]` plus an empty-labels all-outer check), [37, 39] (`train_and_save` CSV read + delegation, same temp-CSV + `patch.object` technique incl. `train_size=1` variant), [49, 74] (`train_and_save_on_dataframe` def-tokens + `train_and_save_on_tokens`, asserted via real 2-row `RandomForest` training to `tmp_path` with `punc_set`/`shall`/`mean` checks, saved-file nonzero, and `predict()` shape agreement)
- **Unreachable:** none
- **Notes:** Same 100% verification and CSV-mocking rationale as the phrase detector.

### lexnlp/ml/predictor.py
- **Tests added:** 7 in lexnlp/ml/tests/test_predictor_coverage.py
- **Lines now covered:** [66] (`__init_subclass__` NotImplementedError via defining a subclass without `_DEFAULT_PIPELINE`), [79, 80] (unfitted `LogisticRegression` final estimator raises `ValueError("is not fitted")`), [85] (fitted `LinearRegression` final estimator without `predict_proba` raises `ValueError` matching `ScikitLearnHasPredictProba`), [112, 114] (`sigma_` legacy patch assigns `var_`/`variance_`, incl. preserve-existing-`var_` variant), [121] (base `_sanity_check` raises `NotImplementedError`)
- **Unreachable:** none
- **Notes:** All tests use tiny real sklearn `Pipeline` objects (no model downloads, no network). Verified the 7 listed lines no longer appear as missing (remaining misses in a narrow run are `get_default_pipeline_tag`/`get_default_pipeline` env branches covered by contract-predictor tests).

### lexnlp/nlp/en/segments/sections.py
- **Tests added:** 6 in lexnlp/nlp/en/segments/tests/test_sections_coverage.py (new `tests/__init__.py` created)
- **Lines now covered:** [69] (`DocumentSection.__str__` asserted exactly `"Hello [1: 5]"`), [73] (`__eq__` against `str` returns `NotImplemented`, plus `==`/`!=` fallback), [425, 426, 427] (`get_document_sections_with_titles` via real `"SECTION 1./2."` text + `get_sentence_span_list`, asserting 2 sections, titles, spans, and title-slice equality), [441] (`find_section_titles` with `[]` returns `None`), [453] (`section.end < sentence.start` break via crafted sections `[(0,5),(50,55)]`/sentences `[(0,5),(100,110)]`, asserting first title updated to `"xxxxx"` and second left as `"B"`)
- **Unreachable:** none
- **Notes:** Verified 100% (`sections.py 238 0 100%`) with existing `test_sections.py` + new file. `use_ml=False` regex path used to avoid ML flakiness.

### lexnlp/extract/common/annotations/ratio_annotation.py
- **Tests added:** 5 in lexnlp/extract/common/annotations/tests/test_ratio_annotation_coverage.py
- **Lines now covered:** [53, 54, 55, 56, 57, 58] (entire `get_dictionary_values`: all-fields tags `{"Extracted Entity Ratio": "0.5", "left": "1", "right": "2"}`, left-only/right-only branch presence/absence, empty-optionals exact-dict, plus `to_dictionary` merge incl. `attrs`/`Extracted Entity Type`)
- **Unreachable:** none
- **Notes:** Remaining miss in a narrow run is lines 49-50 (`get_cite_value_parts`), outside this batch's scope and covered by ratio-extractor tests.

### lexnlp/extract/common/countries.py
- **Tests added:** 4 in lexnlp/extract/common/tests/test_countries_coverage.py
- **Lines now covered:** [73, 74] (`official_name` fallback via `"United States of America"`/`"united states of america"` resolving to US and `"Federal Republic of Germany"` variants to DE, with `pycountry.countries.get(name=...) is None` asserted to prove the fallback path)
- **Unreachable:** [59, 60] (`except LookupError` around `pycountry.countries.get`): `Database.get` only raises `LookupError` for non-`str` values but `lookup_country` always passes `text.strip()` (`str`), so no real `str` input reaches it (verified: `get(alpha_2="")`/`get(name="")` return `None`, bad-field raises `KeyError`); [69, 70] (fallback `c.name` match): `pycountry` 24.6.1 indexes are case-insensitive (`get(name="germany")` already returns Germany), so any exact name succeeds in the direct loop and never reaches the fallback — the fallback comment assumes case-sensitive indexes. Reaching either would require monkeypatching `pycountry` to raise/return `None`, which is contortion, not real behaviour.
- **Notes:** Combined existing `test_countries.py` + new file leaves exactly `59-60, 69-70` missing, matching the unreachable judgment above.

### lexnlp/extract/common/fact_extracting.py
- **Tests added:** 7 in lexnlp/extract/common/tests/test_fact_extracting_coverage.py
- **Lines now covered:** [149, 150, 155, 161, 169, 172, 220, 221] (all 8; existing + new files together measure 100%, 165/165 statements)
- **Unreachable:** none
- **Notes:** Unsupported-format branch (155) is not reachable through the public registry as-is (en/de/es register both fmt_class and fmt_object, and fmt_dict aliases fmt_class), so the test temporarily narrows the en registry and restores it in `finally`. Exclude-all test uses `exclude_types=set(AnnotationType)` with `extract_all=True`. Global `parser_extra_arguments`/`func_by_lang` state is saved and restored so sibling tests are unaffected. One transient `ModuleNotFoundError: No module named 'dateparser'` collection error appeared once under `--cov` and did not reproduce on immediate re-run (15 passed, 100%).

### lexnlp/extract/en/addresses/address_features.py
- **Tests added:** 7 in lexnlp/extract/en/addresses/tests/test_address_features_coverage.py
- **Lines now covered:** [36, 53, 73, 89, 149, 187, 188] (combined with existing test_addresses.py: 99%, only line 192 missing)
- **Unreachable:** [192] (`prepare_pos_tagset_index_file()` call under `if __name__ == "__main__":` — runs only when the module is executed as a script, not on import)
- **Notes:** LATENT BUG (source not modified per task rules): `prepare_pos_tagset_index_file` calls `nltk.help.load(...)`, but the installed nltk's `nltk.help` module has no `load` attribute (verified: raises `AttributeError` before line 187; the working call is `nltk.data.load`). The test shims `nltk.help.load` (`create=True`) purely as the external-resource load and intercepts the output-file `open`, so lines 187-188 (sorted-index build + `json.dump`) execute for real and the written JSON is asserted as `{"NN": 1, "VB": 2}`. Maintainer should fix the call to `nltk.data.load`. No repo data files were touched (write path was captured in memory).

### lexnlp/ml/vectorizers.py
- **Tests added:** 8 in lexnlp/ml/tests/test_vectorizers_coverage.py
- **Lines now covered:** [25, 50, 69, 85, 86, 87, 88, 90] (all 8; new file alone measures 97%, only line 93 `infer_vector` outside this batch remains)
- **Unreachable:** none
- **Notes:** `Doc2Vec(vector_size=5, min_count=1)` constructs an untrained model cheaply with no corpus, so no mocks were needed for the instance path; `Doc2Vec.load` is mocked only for the PathLike branch (explicitly allowed model-load mock). Line 93 (`vectorize` via `infer_vector`) was already covered by the existing suite and is out of scope.

### lexnlp/nlp/en/segments/backends.py
- **Tests added:** 7 in lexnlp/nlp/en/segments/tests/test_backends_coverage.py
- **Lines now covered:** [31, 39, 50, 71, 80, 88, 95, 97] (all 8; with existing test_sota_backends.py: 100%, 52/52 statements)
- **Unreachable:** none
- **Notes:** `SaTSentenceSegmenter.__call__` is a generator function, so the non-string-text (80) and string-split-result (88) `TypeError`s only raise on iteration — tests use `list(...)`. Legacy test asserts span-level equality with `get_sentence_span` plus per-span `(start, end, text[start:end])` invariants.

### lexnlp/utils/parse_df.py
- **Tests added:** 5 in lexnlp/utils/tests/test_parse_df_coverage.py
- **Lines now covered:** [88, 101, 123, 124, 125, 126, 127, 128, 182, 222] (all 10; with existing tests only lines 136-140, the `line_processor` branch, remain)
- **Unreachable:** none
- **Notes:** Priority test uses two rows sharing `name="Peppa"` with `priority_sort_column="prio"` and asserts the `prio=1` row's title wins; non-unique test asserts the full `entities` list `[{"label": "Low"}, {"label": "High"}]`.

### lexnlp/extract/common/text_beautifier.py
- **Tests added:** 6 in lexnlp/extract/common/tests/test_text_beautifier_coverage.py
- **Lines now covered:** [43, 54, 75, 126, 127, 135, 136, 255] (exception fallbacks via `None` input; whitespace-only coords `("   ", 0, 3) -> ("", 3, 3)`; far-offset `find_transformed_word` returning `None` contrasted with a near hit `('\"', 0)`)
- **Unreachable:** [99] (`stack = 1 if not stack else 0` inside `elif c in open_set`). When `flip_stack` is True, `open_set is close_set` (both are `QUOTES`), so the preceding `if c in close_set` catches every matching char and the `elif` never fires; when `flip_stack` is False the `else` branch runs instead. Dead code on both paths.
- **Notes:** nothing else changed; lines 164-165/226/228 seen missing in a two-file subset run are covered by other suite files and were not in this batch's list.

### lexnlp/extract/en/amounts.py
- **Tests added:** 9 in lexnlp/extract/en/tests/test_amounts_coverage.py
- **Lines now covered:** [276, 277, 295, 326, 328, 333, 372, 435, 450] (all 9; subset run with existing amount tests shows 99% with only line 304 remaining)
- **Unreachable:** none
- **Notes:** 276-277 use a minimal monkeypatch of `FRACTION_EXTRACT_PTN_RE.search` returning denominator `"zeroth"` (which really evaluates to `Decimal(0)`), so the caught `ZeroDivisionError` is genuine `decimal.DivisionByZero` and the fallback `Decimal(0)` is asserted against the unpatched `1/3` value. 333 is hit with `Decimal("1E+30")`, whose `quantize(Decimal("0.0"))` raises `InvalidOperation` under the default 28-digit context, returning the amount unchanged. 435 asserts exact `(coords, value, text)` triples for `extended_sources=False`.

### lexnlp/ml/catalog/__init__.py
- **Tests added:** 5 in lexnlp/ml/catalog/tests/test_init_coverage.py
- **Lines now covered:** [38, 49, 117, 118, 119, 123, 124, 125, 128] (all 9; `coverage report` on `lexnlp/ml/catalog/__init__.py` with existing + new tests: 100%, 55/55 statements)
- **Unreachable:** none
- **Notes:** 38 via an existing regular file as the `nltk.data.path` candidate (skipped, falls through to `~/nltk_data`); 49 via empty `nltk.data.path` plus `Path.home` pointed at a missing tree (nearest existing parent `/` is not writable as non-root, so it reaches the `cwd` fallback). 117-119 via a tag added after cache warmup; 123-125 via deleting the cached file; 128 via both `FileNotFoundError` paths. Tests set `catalog.CATALOG` directly with try/finally restore (no module reload), so they do not disturb the existing reload-based path tests.

### lexnlp/extract/pt/dates.py
- **Tests added:** 15 in lexnlp/extract/pt/tests/test_dates_coverage.py
- **Lines now covered:** [205, 213, 228, 231, 232, 234, 250, 254, 255] (205 via `""`/`"ab"`; 213 via `"terça feira"` rejected vs `"terça feira 2020"` accepted; 228/231-232 via `_coerce_year("")`/`"abcd"` → None; 234 via `"05"→2005`, `"20"→2020`, `"49"→2049`, `"50"→1950`, `"95"→1995`; 250 via day 0/month 13 → None; 254-255 via Apr-31/Feb-30 → None vs real `datetime(2020,2,15)`; plus end-to-end `Brasília, 12 de março de 2024` coords `(10,29)` and 2-digit `15/02/20` → `2020-02-15`)
- **Unreachable:** [316, 317, 319] (defensive `try/except` + `if not month: continue` around the locality-date build). `LOCALITY_DATE_RE` only matches known `PT_MONTHS` names and `\d` groups, and `_MONTH_NUMBER` is built from the same `info` table (lookup is lowercased on both sides), so `int()` cannot fail and the month always resolves on real `finditer` output.
- **Notes:** pytest-cov (`--cov=...`) cannot collect the pt package in this env (ImportError `numpy: cannot load module more than once per process` via `pt/__init__` → courts → pandas); coverage measured with `python -m coverage run -m pytest -p no:cov` instead (96%, remainder `203, 208-209` covered by other suite files outside the two-file subset run).

### lexnlp/extract/common/entities/entity_banlist.py
- **Tests added:** 6 in lexnlp/extract/common/entities/tests/test_entity_banlist_coverage.py (new sibling `tests/` dir + `__init__.py`; existing `lexnlp/extract/common/tests/test_entity_banlist.py` untouched)
- **Lines now covered:** [32, 33, 34, 35, 36, 37, 38, 39, 40, 107, 108] (all 11; `__repr__` flag combos `"company" I,T` / `"x" Re` / `"AbC"` / `"Acme.*" I,Re,T`, and `BanListUsage` reprs `0 items provided, default=Trueappend=False` / `2 items provided, default=Falseappend=True`)
- **Unreachable:** none
- **Notes:** subset run (existing + new banlist tests) shows 93% with only `52-55, 87` (`check_list`, 4-col CSV branch) remaining — covered by other suite files (e.g. company-detector tests), not in this batch's list.

### lexnlp/extract/de/de_date_parser.py
- **Tests added:** 11 in lexnlp/extract/de/tests/test_de_date_parser_coverage.py
- **Lines now covered:** [35, 38, 99, 118, 127, 161, 165, 170, 171, 187, 192] (all 11; 35/38 `str`/`repr` `"Juli [month]"`; 99 leading-separator empty token; 118 negative numeral via `w2n.convert` wrapper returning `-4` for one token (real `minus eins` cannot survive whitespace splitting as a single token); 127 `§5` non-alpha token; 161 trailing `" und "` empty segment; 165 `Locale("")` → `RuntimeError("Define text and language.")`; 170-171 monkeypatched `get_dateparser_dates` raising, `capsys` asserts `"boom"` printed and `[]` yielded; 187 overlapping `Oktober 2011`/`5. Oktober 2011` pair → single `(0,15)`-style annotation; 192 stub classifier (`columns=[]`, numpy proba) reject → `[]`, accept → one `2011-10-05` annotation)
- **Unreachable:** none
- **Notes:** subset run (existing de date tests + new) shows 98% with only lines 47/153 remaining, covered elsewhere in the full suite. Heavyweight sklearn model load avoided via the stub; real `dateparser` output used everywhere except the two tests that deterministically need a failure/overlap shape.

### lexnlp/extract/common/copyrights/copyright_en_style_parser.py
- **Tests added:** 7 in lexnlp/extract/common/copyrights/tests/test_copyright_en_style_parser_coverage.py
- **Lines now covered:** [38, 39, 40, 42, 46, 94, 95, 96, 97, 111] (all 10; 46 base `extract_phrases_with_coords` raises `NotImplementedError`; 38-42 `get_copyrights` both `return_sources` branches yielding exact `("Copyright", "2019", "Siemens AG")` / 4-tuples; 94-96 preset valid company `"Acme Corp"` kept; 97 invalid `"123"` reset then rederived `"Siemens AG"`; 111 `["abc","def"]` → `"abc"`)
- **Unreachable:** none (but see note)
- **Notes:** `CopyrightEnStyleParser.get_copyrights` hardcodes the base class (`CopyrightEnStyleParser.get_copyright_annotations`), whose phrase splitter is the abstract `raise NotImplementedError` — so the `for`/`yield` body (38-42) is unreachable without stubbing that seam. Tests monkeypatch it with the production DE `LineProcessor` splitter (real phrase-splitting code from a subclass), so all downstream regex/annotation logic exercised is real. Subset run shows 93% with only `68-71, 75, 116` (year-at-end, span clamp, single-year) remaining — covered by other suites.
### lexnlp/extract/all_locales/citations.py
- **Tests added:** 7 in lexnlp/extract/all_locales/tests/test_citations_coverage.py
- **Lines now covered:** [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 19, 20, 21] (all 15; 100% per coverage run)
- **Unreachable:** none
- **Notes:** thin en/de dispatcher. Tests assert real routing: en `410 U.S. 113 (1973)` yields volume 410/page 113/reporter `U.S.` at coords (3, 24); de BGBl text yields article 2/page 2477; unknown locales (`fr-FR`, `es-ES`, `pt-PT`) fall back to the English routine; generator type and empty/citation-free inputs covered.

### lexnlp/extract/all_locales/durations.py
- **Tests added:** 9 in lexnlp/extract/all_locales/tests/test_durations_coverage.py
- **Lines now covered:** [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 19, 20, 21] (all 15; 100% per coverage run)
- **Unreachable:** none
- **Notes:** en `30 days` / de `30 Tage` routing asserted against the direct en/de routines (text+coords+locale). `float_digits` forwarding proven with real values: default/4 gives amount 1.5679, 2 gives 1.57 on `1.56789 years`.

### lexnlp/extract/all_locales/money.py
- **Tests added:** 9 in lexnlp/extract/all_locales/tests/test_money_coverage.py
- **Lines now covered:** [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 19, 24, 25] (all 15; 100% per coverage run)
- **Unreachable:** none
- **Notes:** en `1000.56789 dollars` (amount Decimal, USD) vs de `1.000,50 Euro` routing asserted against direct routines. `float_digits` forwarding proven with real values: 4 gives 1000.5679, 2 gives 1000.57.

### lexnlp/extract/all_locales/percents.py
- **Tests added:** 9 in lexnlp/extract/all_locales/tests/test_percents_coverage.py
- **Lines now covered:** [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 19, 24, 25] (all 15; 100% per coverage run)
- **Unreachable:** none
- **Notes:** en `5.56789%` vs de `5,5 %` routing asserted against direct routines. `float_digits` forwarding proven with real values: 4 gives 5.5679, 2 gives 5.57.

### lexnlp/extract/common/copyrights/copyright_parser.py
- **Tests added:** 4 in lexnlp/extract/common/copyrights/tests/test_copyright_parser_coverage.py
- **Lines now covered:** [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 23, 26, 27, 28, 29] (all 18; 100% per coverage run)
- **Unreachable:** none
- **Notes:** `make_annotation_from_pattern` tested directly with a real `CopyrightPatternFound` (company `ContraxSuite, LLC`, years 2015/2021) plus an end-to-end `parse` test with an inline parsing function. Asserts exact company/years/coords/text-slice/locale, not just no-crash.

### lexnlp/ml/sklearn_transformers.py
- **Tests added:** 16 in lexnlp/ml/tests/test_sklearn_transformers_coverage.py
- **Lines now covered:** [33, 56, 57, 58, 59, 62, 63, 64, 81, 85, 118, 119, 120, 121, 136, 154, 155, 174] (all 18; module now 100% — also picked up 133-134 block-text-with-head and transform paths)
- **Unreachable:** none
- **Notes:** `parallel_estimator` exercised with real joblib fan-out for both branches (dense `concatenate` via doubling estimator, sparse `vstack` via `csr_matrix` estimator). `VectorizerKeywordSearch` used as the real `Vectorizer`. `preprocess` head-limiting proven with real `get_sentences`/`get_lemmas`: iterable+head drops the second sentence (`bark` absent), block+head keeps only the first sentence.

### lexnlp/extract/common/ocr_rating/ocr_rating_calculator.py
- **Tests added:** 15 in lexnlp/extract/common/ocr_rating/tests/test_ocr_rating_calculator_coverage.py
- **Lines now covered:** [27, 55, 56, 57, 58, 65, 71, 72, 73, 74, 75, 76, 77, 78, 79, 93, 116] (all 17; module now 100% — also covered quadratic `get_rating` 105-109 and `build_cs_quad_rating_calculator` 123)
- **Unreachable:** none
- **Notes:** `init_language_data` tested with real `to_pickle`/`read_pickle` round-trips in tmp dirs, including the no-overwrite `continue` (explicit path wins over folder file). Unknown-language fallback (`xx-unknown` == `en`) proven against the real bundled `reference_vectors/en.pickle` via `build_cs_rating_calculator`. `break_chars` covers all five branches.

### lexnlp/extract/ner/__init__.py
- **Tests added:** 13 in lexnlp/extract/ner/tests/test_ner_coverage.py
- **Lines now covered:** [93, 99, 106, 110, 111, 112, 113, 114, 123, 159, 175, 176, 215, 216, 217, 221, 231] (all 17; 100% per coverage run)
- **Unreachable:** none
- **Notes:** spaCy is not installed here, so the spaCy-backend lines use mocks at the documented seams only: fake `spacy` module in `sys.modules` for `spacy_is_available() is True` (line 93), stubbed `_load_spacy_pipeline` returning fake `doc.ents` for `_spacy_extract` (106-123) with exact `HybridNERMatch` equality asserts, and `prefer_spacy=True` success + `OSError`-degrade-to-NLTK paths for 214-221. NLTK lines use real code: `_nltk_extract("")` for 159, real `pos_tag` + stubbed `ne_chunk` returning an empty `Tree("PERSON", [])` and an oversized 5-leaf tree for 175-176. No skip markers added.

### lexnlp/extract/ml/classifier/base_token_sequence_classifier_model.py
- **Tests added:** 21 in lexnlp/extract/ml/classifier/tests/test_base_token_sequence_classifier_model_coverage.py
- **Lines now covered:** [53, 54, 55, 56, 58, 101, 102, 105, 106, 121, 122, 123, 127, 128, 129, 135, 158, 177, 188, 222, 225, 226, 227, 230, 231, 233, 235, 236, 237] (all 29; module now 100% with existing tests)
- **Unreachable:** none
- **Notes:** `get_classifier(use_spacy=True)` constructs (no spaCy import at init, so no mock needed). run_model branches driven by a fake estimator plus stubbed `get_feature_data` with real span assertions for all 10 strict/non-strict start/inner/end/outer combinations. skops-dispatch load covered via a real `dump_model` round-trip.

### lexnlp/extract/common/date_parsing/datefinder.py
- **Tests added:** 32 in lexnlp/extract/common/tests/test_datefinder_coverage.py
- **Lines now covered:** [35, 36, 39, 42, 371, 372, 373, 376, 378, 379, 380, 381, 382, 384, 385, 386, 426, 427, 445, 455, 465, 466, 468, 469, 480, 499, 530, 531] (all 28; module now 100% — also covered get_token_group empty-group line 278 and the dateparser/dateutil mismatch line 405)
- **Unreachable:** none
- **Notes:** All fallback paths use real inputs, no mocks: "zz" reaches the short-string guard (445); "due January 5, 2024 EST" with timezones=["EST"] runs the full dateutil fallback + `_add_tzinfo` (455, 465-469, 480); `Locale("xx-YY")` makes dateparser raise ValueError on both the locales and languages calls (426-427); "99/99/9999" extracts a fragment but parses to None, covering the find_dates skip (376).

### lexnlp/ml/model_io.py
- **Tests added:** 17 in lexnlp/ml/tests/test_model_io_coverage.py
- **Lines now covered:** [118, 215, 216, 219, 221, 223, 224, 225, 226, 227, 235, 236, 239, 241, 242, 243, 257, 258, 259, 264, 265, 266, 272, 273, 274, 275, 276, 277] (all 28; module now 100% with existing tests)
- **Unreachable:** none (lines 237-238 carry their own `# pragma: no cover` for the `vars()` TypeError guard and were not in the listed set)
- **Notes:** skops-sibling preference proven by dumping `{"v": "skops"}` next to a legacy pickle holding `{"v": "legacy"}` and asserting the skops value wins. Tree-compat rewrite tested against the real `_tree.NODE_DTYPE` (old dtype minus `missing_go_to_left`, values preserved, new field zero-filled, passthrough for new dtype, original validator restored). ImportError path via `sys.modules["numpy"] = None`.

### lexnlp/utils/unicode/unicode_lookup.py
- **Tests added:** 6 in lexnlp/utils/unicode/tests/test_unicode_lookup_coverage.py
- **Lines now covered:** [24, 25, 26, 27, 28, 61, 63, 88, 97, 106, 115, 124, 138, 145, 152, 159, 174, 177, 180, 183, 186, 189, 192] (all listed lines except the `__main__` body below)
- **Unreachable:** [196, 197, 198, 199, 200, 201] — the `if __name__ == "__main__":` body (build + prints). Only runs when the module is executed as a script; per task rules, not contorted to reach. (Line 195, the guard itself, evaluates False on import and is covered.)
- **Notes:** `_load_table` error paths use a real missing path (ignore_error True returns None with the "Unable to load" message asserted via capsys; False re-raises OSError). `build_lookup_tables` runs for real against `test_data/lexnlp/utils/unicode_data.txt` with assertions on every category group, the `a`->`Ll`/`L` mappings, and non-empty outputs.
