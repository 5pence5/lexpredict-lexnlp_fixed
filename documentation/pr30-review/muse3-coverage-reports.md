# PR30 testgen (Muse batch 41)

### lexnlp/extract/en/durations.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_durations_coverage.py
- **Lines now covered:** [92] (`return list(get_durations(...))` in `get_duration_list` — `"The lease is for 30 days."` yields `[("day", Decimal("30.0"), Decimal("30.0"))]`, `"The term is 2 years."` yields `[("year", Decimal("2.0"), Decimal("730.0"))]`, plus `return_sources=True` source-text and empty-text cases)
- **Unreachable:** none
- **Notes:** New-tests-only coverage run shows line 92 covered (remaining misses 58, 65, 96, 100 belong to other paths exercised by the existing data-driven suite).

### lexnlp/extract/es/regulations.py
- **Tests added:** 3 in lexnlp/extract/es/tests/test_regulations_coverage.py
- **Lines now covered:** [231] (`return list(parser.parse(text, language))` in `get_regulation_annotation_list` — `"Ley Orgánica 4/2015, de 30 de marzo"` pins name/text/coords `(14, 49)`/locale `es`/country `Spain`; field-wise generator-equivalence test since `RegulationAnnotation` defines no `__eq__`; empty-text cases)
- **Unreachable:** none
- **Notes:** `pytest --cov=lexnlp.extract.es.regulations` cannot collect in this env (`ImportError: numpy: cannot load module more than once per process` via `es/__init__` → pandas); line 231 execution verified with `sys.settrace`.

### lexnlp/extract/pt/distances.py
- **Tests added:** 5 in lexnlp/extract/pt/tests/test_distances_coverage.py
- **Lines now covered:** [103] (`yield ant.amount, ant.distance_type` — the `return_sources=False` branch of `get_distances`, via `get_distances(...)`/`get_distance_list(...)` defaults asserting exact 2-tuples e.g. `(Decimal("12.5000"), "kilometer")` for `"12,5 quilômetros"` and `(Decimal("100.0000"), "kilometer")` for `"100 km"`, plus a `return_sources=True` 3-tuple contrast and empty cases)
- **Unreachable:** none
- **Notes:** Same pytest-cov/pt collection caveat as above; line 103 execution verified with `sys.settrace`.

### lexnlp/extract/pt/durations.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_durations_coverage.py
- **Lines now covered:** [138] (`return False` when `curr_days >= prev_days` in `_is_continuation` — `"Prazo de 6 meses e 2 anos."` yields 2 ungrouped annotations (`month` 180 days + `year` 730 days); equal-unit `"2 dias e 30 dias."` hits the `==` half of the guard; descending `"2 anos e 6 meses"` contrast merges to one complex annotation with 910 days; direct `_is_continuation(...) is False` pin)
- **Unreachable:** none
- **Notes:** None.

### lexnlp/extract/common/annotations/regulation_annotation.py
- **Tests added:** 3 in lexnlp/extract/common/annotations/tests/test_regulation_annotation_coverage.py
- **Lines now covered:** [60] (`dic.tags["External Reference Source"] = self.source` — source="SEC" case asserts full 4-key tags dict; new-tests-only cov run no longer lists line 60 as missing)
- **Unreachable:** none
- **Notes:** Also pins the no-source branch (tag absent) and the `text or name` fallback when text is None.

### lexnlp/extract/common/language_dictionary_reader.py
- **Tests added:** 0 (none — line judged unreachable, no file created)
- **Lines now covered:** none
- **Unreachable:** [26] (`continue` under `if not line:` in `read_str_set`). `fr.readlines()` never yields `""` (verified: `["a\n", "\n", "b\n"]` for a file with a blank line, `[]` for an empty file), and only `""` is falsy among str, so the guard can never fire on a real file. Reaching it would require mocking `codecs.open` to yield `""`, which would contort the test rather than exercise real behaviour.
- **Notes:** None.

### lexnlp/extract/common/money_detector.py
- **Tests added:** 0 (none — line judged unreachable, no file created)
- **Lines now covered:** none
- **Unreachable:** [68] (`continue` when a regex match has no prefix, postfix, or trigger_word). Every one of the four pattern alternatives requires at least one of those groups (prefix-amount / amount-postfix / amount-prefix / trigger-word-amount), and `capturesdict()` returns `[]` (falsy) vs non-empty lists, so a real match always carries one. Probed en + de detectors over 15 diverse samples (12/10 matches): zero matches with all three groups empty.
- **Notes:** Reaching it would require stubbing `currency_ptn_re` with a fake match, which mocks the very unit under test.

### lexnlp/extract/common/preprocessing/html_cleaner.py
- **Tests added:** 2 in lexnlp/extract/common/preprocessing/tests/test_html_cleaner_coverage.py
- **Lines now covered:** [104] (`return ""` empty-input guard in `clean_html` — `clean_html("") == ""`; new-tests-only cov run no longer lists line 104 as missing)
- **Unreachable:** none
- **Notes:** Second test pins that `clean_html` still strips `<script>` while keeping `<p>hi</p>`.

### lexnlp/extract/en/regulations.py
- **Tests added:** 6 in lexnlp/extract/en/tests/test_regulations_coverage.py
- **Lines now covered:** [74] (`return list(get_regulations(...))` in `get_regulation_list` — `"test 123 U.S.C § 456, code"` yields `[("United States Code", "123 USC § 456")]`, plus `return_source` 3-tuple and `as_dict` dict cases with empty-text cases), [113] (`return list(get_regulation_annotations(...))` in `get_regulation_annotation_list` — same text pins `RegulationAnnotation` source/name/text/coords `(5, 20)`/locale `en` plus generator-equivalence and empty cases)
- **Unreachable:** none
- **Notes:** New-tests-only cov run no longer lists 74/113 as missing (remaining misses 91, 99-108 are other paths covered by the existing suite).

### lexnlp/extract/en/urls.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_urls_coverage.py
- **Lines now covered:** [59] (`return list(get_urls(...))` in `get_url_list` — `"Visit https://example.com/page and www.google.com now"` yields `["https://example.com/page", "www.google.com"]` plus empty-text cases), [76] (`return list(get_url_annotations(...))` in `get_url_annotation_list` — same text pins both `UrlAnnotation` url/coords `(6, 30)`/`(35, 49)`/locale `en` plus generator-equivalence and empty cases)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% for this module.

### lexnlp/extract/pt/money.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_money_coverage.py
- **Lines now covered:** [165] (`return list(get_money(...))` in `get_money_list` — `"Valor: R$ 50,00."` yields `[{"source_text": "R$ 50,00", "amount": Decimal("50.0000"), "currency": "BRL", "location_start": 7, "location_end": 15}]`, plus suffix-form, `float_digits=0` rounding (`Decimal("13")`), and empty cases; generator-equivalence with `get_money`)
- **Unreachable:** [112] (`continue` when a prefix-match span repeats). `_MONEY_PREFIX_RE.finditer` yields strictly advancing, non-overlapping spans, so `span in seen_spans` can never be true in a single pass (verified over 5 diverse samples: all spans unique). Reaching it would require stubbing the regex with a fake duplicate match, which mocks the very unit under test.
- **Notes:** `pytest --cov=lexnlp.extract.pt.money` cannot collect in this env (`ImportError: numpy: cannot load module more than once per process` via `pt/__init__` → pandas, same as prior pt batches); line 165 execution verified with `sys.settrace` (hit set includes 165, excludes 112).

### lexnlp/nlp/en/segments/pages.py
- **Tests added:** 4 in lexnlp/nlp/en/tests/test_pages_coverage.py
- **Lines now covered:** [118] (`line_window_pre = lines_count - 1` clamp — `get_page_break_feature_names(3, 5, 1)` keeps `line_len_-2`/`line_len_1` while `line_len_-3` is absent), [122] (`line_window_post = lines_count - line_window_post - 1` clamp — `get_page_break_feature_names(10, 2, 10)` keeps `line_len_-2`/`line_len_-1` while `line_len_0`/`line_len_10` are absent)
- **Unreachable:** none
- **Notes:** Also pins the both-clamped tiny-doc case `(1, 3, 3)` with zero `line_len_*` keys and the unclamped `(10, 3, 3)` full `-3..3` range; new-tests-only cov run no longer lists 118/122 as missing.

### lexnlp/extract/de/durations.py
- **Tests added:** 3 in lexnlp/extract/de/tests/test_durations_coverage.py
- **Lines now covered:** [138] (`return DeDurationParser.get_annotations(...)` in `get_duration_annotations_list` — "seit fünfundzwanzig Jahren" pins coords `(4, 26)`/`jahren`/`year`/`Decimal(25)`/`Decimal(9125)`/locale `de` plus generator-equivalence, "Vier Wochen, 3 Tage und 151 Sekunden." pins complex `Decimal("31.0017")`/`second`/`sekunden`, plus empty-text cases)
- **Unreachable:** [94] (`continue` when `DURATION_MAP_RE.findall` returns empty). `DURATION_PTN_RE` and `DURATION_MAP_RE` are built from the same `DURATION_TRANSLATION_MAP` key list, the capture is lowercased before the findall, and every key lowercased matches at least itself (verified over all 18 keys x lower/upper/capitalize). A real match therefore always yields a non-empty findall; reaching it would require stubbing the class regex, which mocks the unit under test.
- **Notes:** Combined run (`test_durations.py` + new file) shows only 94 missing (98%); new-tests-only run already excludes 138 from missing.

### lexnlp/extract/en/conditions.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_conditions_coverage.py
- **Lines now covered:** [117] (`return list(get_conditions(...))` in `get_condition_list` — "Hello if you come, we will go." yields `[("if", "Hello", "")]` with generator-equivalence), [178] (`return list(get_condition_annotations(...))` in `get_condition_annotation_list` — same text pins `condition`/`pre`/`post`/`coords (0, 9)`/locale `en` plus generator-equivalence)
- **Unreachable:** none
- **Notes:** Also pins `unless and until` / `subject to` variants, `strict=False == strict=True` (flag has no effect per code comment), and empty-text cases; combined run shows 100% for this module.

### lexnlp/extract/en/copyright.py
- **Tests added:** 5 in lexnlp/extract/en/tests/test_copyright_coverage.py
- **Lines now covered:** [91] (`return list(get_copyrights(...))` in `get_copyright_list` — "Copyright 2020 Maverick International, Inc." yields `[("Copyright", "2020", "Maverick International, Inc")]` plus `return_sources=True` 4-tuple with source text), [130] (`return list(get_copyright_annotations(...))` in `get_copyright_annotation_list` — same text pins `sign`/`date`/`name`/`coords (0, 42)`/locale `en` plus generator-equivalence)
- **Unreachable:** none
- **Notes:** Also pins `text == ""` without sources vs populated `text` with `return_sources=True`, and empty-text cases; combined run shows 100% for this module.

### lexnlp/extract/en/distances.py
- **Tests added:** 5 in lexnlp/extract/en/tests/test_distances_coverage.py
- **Lines now covered:** [61] (`return list(get_distances(...))` in `get_distance_list` — "Today I ran 8 miles." yields `[(Decimal("8.0"), "mile")]`, "The road is 5 km long." yields `[(Decimal("5.0"), "kilometer")]` plus `return_sources=True` 3-tuple `[(Decimal("5.0"), "kilometer", "5 km")]`), [81] (`return list(get_distance_annotations(...))` in `get_distance_annotation_list` — same text pins `coords (12, 17)`/`amount`/`kilometer`/`"5 km"`/locale `en` plus generator-equivalence)
- **Unreachable:** none
- **Notes:** Also pins word-number ("eight kilometers"), `float_digits=0` rounding (`Decimal("5")`), and empty/bare-unit ("", "km") cases; combined run shows 100% for this module.

### lexnlp/extract/common/annotations/duration_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_duration_annotation_coverage.py
- **Lines now covered:** [64, 65] (`get_dictionary_values` — populated amount/text, empty-value fallback, `to_dictionary` merge, plus `get_cite_value_parts` pin)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (25/25) for this module.

### lexnlp/extract/common/annotations/law_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_law_annotation_coverage.py
- **Lines now covered:** [27, 28] (`get_dictionary_values` — with text, `text or name` fallback on default `""`, `to_dictionary` merge, plus `get_cite_value_parts` pin)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (16/16) for this module.

### lexnlp/extract/common/annotations/phone_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_phone_annotation_coverage.py
- **Lines now covered:** [37, 38] (`get_dictionary_values` — populated phone/text, `phone or ""` fallback when phone is None, `to_dictionary` merge, plus `get_cite_value_parts` pin)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (17/17) for this module.

### lexnlp/extract/common/annotations/trademark_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_trademark_annotation_coverage.py
- **Lines now covered:** [37, 38] (`get_dictionary_values` — populated trademark/text, default `""` trademark, `to_dictionary` merge, plus `get_cite_value_parts` pin)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (17/17) for this module.

### lexnlp/utils/lines_processing/parsed_text_quality_estimator.py
- **Tests added:** 7 in lexnlp/utils/lines_processing/tests/test_parsed_text_quality_estimator_coverage.py
- **Lines now covered:** [57] (`TypedLineOrPhrase.__repr__` — `wrap_line` roundtrip plus header-type repr asserting `"[type] text->ending"`), [136] (`return` when `lines_total == 0` in `estimate_extra_line_breaks` — `estimate_text("")` yields `lines == []` with zero `extra_line_breaks_prob`/`corrupted_prob`, plus direct no-op call), [185] (`return 0` for empty line in `estimate_line_is_header_prob` — `""`, `"   "`, `" \t "` all `== 0` with a 65-prob contrast)
- **Unreachable:** none
- **Notes:** New-tests-only cov run no longer lists 57/136/185 as missing (remaining misses 131, 138-153, 156-166, 169-175, 178-180, 187, 189, 194 belong to other paths exercised by the existing suite).

### lexnlp/config/en/company_types.py
- **Tests added:** 3 in lexnlp/config/en/tests/test_company_types_coverage.py (new `lexnlp/config/en/tests/` package with `__init__.py`)
- **Lines now covered:** [84] (`CompanyDescriptor.__str__` — `"inc: Inc. (Corporation)"`), [87] (`__repr__` delegates to `__str__` — `repr == str`, plus loaded `get_company_types()["inc"]` roundtrip)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% for this module.

### lexnlp/extract/all_locales/courts.py
- **Tests added:** 2 in lexnlp/extract/all_locales/tests/test_courts_coverage.py
- **Lines now covered:** [59, 60] (`setattr` loop over `toponym.extra_columns` — offline `DictionaryEntry(id=7, ..., extra_columns={"jurisdiction": "Testland", "court_type": "supreme"})` matched against `"The Supreme Court of Testland decided the case."` pins `entity_id`/`category`/`priority`/`name_en`/`name`/`alias`/`locale == "en"`/`coords == (4, 29)` plus `jurisdiction`/`court_type` set from extra columns; second test repeats with `priority=True`)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% for this module. Avoided the network-dependent `test_universal_courts_parser.py` CSV fixtures; used a synthetic `DictionaryEntry` instead.

### lexnlp/extract/common/annotations/amount_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_amount_annotation_coverage.py
- **Lines now covered:** [32, 33] (`get_dictionary_values` — populated value/text, zero-value `""` fallback via default `Decimal("0.0")`, `to_dictionary` merge, plus `get_cite_value_parts` pin for both branches)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (18/18) for this module.

### lexnlp/extract/en/trademarks.py
- **Tests added:** 9 in lexnlp/extract/en/tests/test_trademarks_coverage.py
- **Lines now covered:** [45] (`return list(get_trademarks(text))` in `get_trademark_list` — `"... OASyS(R) product."` yields `["OASyS (R)"]` plus generator-equivalence and empty cases), [64] (`coords = (coords[0], len(text) - 1)` end-of-text clip — `"Licensee's OASyS(R)"` pins `(11, 18)`, `"BetLUCK(TM)"` pins `(0, 10)`, `"Buy Acme™"` pins `(0, 8)`, plus non-clipped `(11, 20)` contrast), [71] (`return list(get_trademark_annotations(text))` in `get_trademark_annotation_list` — same text pins trademark/coords/locale `en` plus generator-equivalence and empty cases)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (33/33) for this module. Clip triggers because NPExtractor normalizes `"OASyS(R)"` to `"OASyS (R)"`, so an in-phrase match ending at the text end overshoots `len(text)`.

### lexnlp/extract/es/definitions.py
- **Tests added:** 4 in lexnlp/extract/es/tests/test_definitions_coverage.py
- **Lines now covered:** [117, 118] (`for annotation in parser.parse(...): yield annotation.to_dictionary()` in `get_definitions` — `"El término \"Software\" se refiere a..."` yields one dict with name `'"Software"'`/type `definition`, plus `Generator` type pin and empty cases), [122] (`return list(get_definitions(...))` in `get_definition_list` — same text plus generator-equivalence and `"Hola."`/`""`/`"Ella está muerta. "` empty cases)
- **Unreachable:** none
- **Notes:** `pytest --cov=lexnlp.extract.es.definitions` cannot collect in this env (`ImportError: numpy: cannot load module more than once per process` via `es/__init__` → pandas, same as prior batches); lines 117/118/122 execution verified with `sys.settrace`.

### lexnlp/extract/ml/detector/phrase_constructor.py
- **Tests added:** 4 in lexnlp/extract/ml/detector/tests/test_phrase_constructor_coverage.py
- **Lines now covered:** [40, 41] (`return f"by class, strict={self.strict}"` — default `"by class, strict=False"` and strict `"by class, strict=True"` pins), [42] (`return f"by score, ..."` — defaults `"by score, min_score=2, max_zeros=2"` and custom `"by score, min_score=7, max_zeros=5"` pins)
- **Unreachable:** none
- **Notes:** New-tests-only cov run no longer lists 40/41/42 as missing (remaining misses 76-83, 99-124, 149-196 belong to `join_tokens*` paths covered by the existing suite).

### lexnlp/extract/pt/definitions.py
- **Tests added:** 4 in lexnlp/extract/pt/tests/test_definitions_coverage.py
- **Lines now covered:** [251, 252] (`for annotation in parser.parse(...): yield annotation.to_dictionary()` in `get_definitions` — `'o termo "Software" refere-se a...'` yields one dict with name `'"Software"'`/type `definition`, plus `Generator` type pin and empty cases), [266] (`return list(get_definitions(...))` in `get_definition_list` — same text plus generator-equivalence and `"Olá."`/`""`/`"Sem rótulo aqui."` empty cases)
- **Unreachable:** none
- **Notes:** Same pytest-cov/pt collection caveat as es above; lines 251/252/266 execution verified with `sys.settrace` (hit sets include 251/252 and 266 respectively).

### lexnlp/utils/unpickler.py
- **Tests added:** 7 in lexnlp/utils/tests/test_unpickler_coverage.py
- **Lines now covered:** [38] (`find_class` remap — direct pins for passthrough `collections.OrderedDict` plus legacy renames `sklearn.tree.tree` → `DecisionTreeClassifier` and `sklearn.ensemble.forest` → `RandomForestClassifier`, and indirectly via every `renamed_load` roundtrip), [49, 54, 55] (`renamed_load` body — `OrderedDict`/`Decimal`/plain-dict roundtrips asserting real values and types, plus a pickled `_LegacyLike` object asserting the line-55 `_patch_legacy_sklearn_estimator` side effect `estimator == base_estimator == "sentinel"`)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (10/10) for this module.

### lexnlp/extract/common/annotations/copyright_annotation.py
- **Tests added:** 6 in lexnlp/extract/common/annotations/tests/test_copyright_annotation_coverage.py
- **Lines now covered:** [42, 43] (`__repr__` — company-preferred `"Acme, (0, 42)"`, name fallback `"nm, (0, 10)"`, text fallback `"some text, (1, 5)"`, all-empty `", (2, 20)"`), [60] (`df.tags["Extracted Entity End"] = self.year_end` — full 5-key tags dict pin with `year_end=2001`, plus key-absent contrast without `year_end`)
- **Unreachable:** none
- **Notes:** New-tests-only cov run leaves only 46-51 missing (`get_cite_value_parts`, covered by the existing `test_annotation.py` suite).

### lexnlp/extract/de/laws.py
- **Tests added:** 6 in lexnlp/extract/de/tests/test_laws_coverage.py
- **Lines now covered:** [112] (`yield from parser.parse(text, language)` — real CSV-backed `setup_parser()` patched in as module `parser`; `"Dies ist durch das AAÜG geschehen."` yields one `LawAnnotation` with name/text `AAÜG`, coords `(18, 24)`, locale `de`, plus `"x"` locale-passthrough and empty-text cases), [121] (`yield annotation.to_dictionary()` in `get_laws` — dict `attrs == {"start": 18, "end": 24}` with `AAÜG` name/text tags, plus `Generator` type pin), [125] (`return list(get_laws(...))` in `get_law_list` — list pin with self-equality and `""` empty cases)
- **Unreachable:** none
- **Notes:** Combined run (`test_laws.py` + `test_laws_coverage.py` + `test_laws_generator_contract.py`) shows 100% for this module.

### lexnlp/extract/de/money.py
- **Tests added:** 8 in lexnlp/extract/de/tests/test_money_coverage.py
- **Lines now covered:** [62] (`yield from money_detector.get_money(...)` in `get_money` — `"100 Pfunde, 45 Dollars"` yields `[(Decimal("100.0"), "GBP"), (Decimal("45.0"), "USD")]`, plus `return_sources=True` 3-tuples with source texts and `Generator` type pin), [70] (`return list(...)` in `get_money_list` — generator-equivalence plus empty cases), [85] (`return list(...)` in `get_money_annotation_list` — amounts/currencies/locale `de`/coords `(0, 11)`/`(12, 22)`/source-text pins, generator-equivalence, empty cases)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (22/22) for this module.

### lexnlp/extract/common/annotations/text_annotation.py
- **Tests added:** 8 in lexnlp/extract/common/annotations/tests/test_text_annotation_coverage.py
- **Lines now covered:** [52] (base `get_cite_value_parts` returns `[name]`, cite `"/en/Siemens"`), [68] (`to_dictionary` new-key branch `df[key] = extras[key]`), [76] (base `get_dictionary_values` returns `{}`), [88] (`get_int_value(None)` returns default), plus merge branch [70], `get_extracted_text` [56], int/str/bad-type `get_int_value`, `safe_cast`
- **Unreachable:** none
- **Notes:** New-tests-only cov run leaves only 38-41 missing (`__repr__`, covered by existing `test_annotation.py`).

### lexnlp/extract/common/text_pattern_collector.py
- **Tests added:** 8 in lexnlp/extract/common/tests/test_text_pattern_collector_coverage.py
- **Lines now covered:** [74] (base `make_annotation_from_pattern` raises `NotImplementedError`), [88] (duplicate-name dedup keeps highest-probability match, incl. quote/whitespace-stripped grouping), [99] (short-list early return for 0/1 matches), [114] (`estimate_match_quality` arithmetic pins 400/990)
- **Unreachable:** none
- **Notes:** Uses a minimal `DummyCollector` subclass; `parse` loop lines 50-65 remain covered by the existing definitions/parser suites, not these tests.

### lexnlp/extract/de/court_citations.py
- **Tests added:** 4 in lexnlp/extract/de/tests/test_court_citations_coverage.py
- **Lines now covered:** [27] (`PossibleToken.__repr__` exact string), [154, 155] (`except TypeError` fallback to year parsing — mocked `get_dates` raising `TypeError`, year `"2015"` token with prob 50, plus no-year empty case), [209] (`get_court_citation_annotation_list` returns BFH/BStBl annotations with locale `de`)
- **Unreachable:** none
- **Notes:** `get_dates` mock (raising `TypeError`) is the only mock; everything else uses real parser input.

### lexnlp/extract/de/geoentities.py
- **Tests added:** 3 in lexnlp/extract/de/tests/test_geoentities_coverage.py
- **Lines now covered:** [193, 194] (`min_alias_len` falsy-to-2 default in `get_geoentities`), [205, 206] (`to_dictionary` yield loop — `"Georgien"` hit at (18, 26) with `Entity ID` 1)
- **Unreachable:** none
- **Notes:** Builds a real single-entry `DictionaryEntry` list (`Georgien`/`de` alias); explicit `min_alias_len=5` and no-match cases pin behavior.

### lexnlp/extract/pt/citations.py
- **Tests added:** 5 in lexnlp/extract/pt/tests/test_citations_coverage.py
- **Lines now covered:** [86] (`_parse_int(None)` returns `None`), [100] (`_normalise_year` fall-through for 3/4-digit years — `"RE 123.456/2019"` yields `year=2019`, `"RE 123.456/202"` yields `year=202`), [155, 156] (`get_citations` yield loop — `["REsp 12.345/SP", "1234567-56.2020.5.04.0001"]` plus empty case), [166] (`get_citation_list` — list/generator equivalence plus empty case)
- **Unreachable:** none
- **Notes:** New-tests-only cov run shows 100% (48/48) for this module.

### lexnlp/extract/pt/regulations.py
- **Tests added:** 3 in lexnlp/extract/pt/tests/test_regulations_coverage.py
- **Lines now covered:** [353] (`get_regulation_annotation_list` — trigger `(0, 13)` + formal `(0, 40)` annotations with `pt`/`Brazil` pins), [367, 368] (`get_regulations` yield loop — `attrs`/`tags` dict pins), [382] (`get_regulation_list` — explicit/`None`/default language equivalence)
- **Unreachable:** [264] (`continue` when a paragraph-leading span duplicates an article span). `ARTICLE_REFERENCE_RE` matches must start with `art`/`artigo` while `PARAGRAPH_LEADING_REFERENCE_RE` matches must start with `§`/`parágrafo`/`inciso`/`alínea`, so identical spans are impossible; verified empirically that paragraph-leading spans always strictly extend the inner `art.` span.
- **Notes:** New-tests-only cov run leaves only line 264 missing for this module.

### lexnlp/nlp/en/segments/paragraphs.py
- **Tests added:** 5 in lexnlp/nlp/en/segments/tests/test_paragraphs_coverage.py
- **Lines now covered:** [157] (`splitlines_with_spans(None)` returns `([], [])`, plus offset pins), [267, 268, 271] (non-matching `ValueError` re-raise — real input `get_paragraph_spans("")` raises `ValueError("Found array with 0 sample(s)...")`), [267, 268, 269] (legacy-message fallback yields `[(0, len(text), text)]` via stub model raising the old-sklearn `"Number of features of the model must match the input"` message; plus a stub raising an unrelated `ValueError` to pin the re-raise path)
- **Unreachable:** none
- **Notes:** The bundled `DecisionTreeClassifier` predates `n_features_in_`, so modern sklearn (1.8.0) never emits the legacy message naturally — the stub-model mock (an explicitly allowed large-model mock) is the only way to exercise line 269. `pytest-cov` is broken in this env (`numpy: cannot load module more than once per process`), so coverage was measured with `coverage run` instead.

### lexnlp/nlp/en/segments/sentences.py
- **Tests added:** 6 in lexnlp/nlp/en/segments/tests/test_sentences_coverage.py
- **Lines now covered:** [80] (`pre_process_document` falsy passthrough for `""`/`None`, plus page-number-line pin `"hello\n123\nworld"` → `"hello\n\nworld"`), [90] (`_trim_span` returns `None` for content-free spans, plus offset pins), [201, 202] (`build_sentence_model` `extra_abbrevs` loop — `"prof"` (absent from training text) in `abbrev_types` proves the loop ran; trained tokenizer keeps `"...Elm Blvd. It opened..."` unsplit vs. plain sentences split normally)
- **Unreachable:** [110] (`continue` when a `SENTENCE_SPLITTERS` match lower-fullmatches `\s*and\s*`). Blank-line matches are whitespace-only (cannot contain `"and"`), and the table-of-contents alternative requires ≥4 non-whitespace chars plus a 5+ `[ \t.]` run while the exclude pattern allows exactly the 3 letters of `"and"` plus whitespace — no match can satisfy both.
- **Notes:** Punkt training uses a seeded (`random.Random(7)`) 400-sentence corpus; tokenizer assertions pin observed deterministic output.

### lexnlp/extract/en/definition_parsing_methods.py
- **Tests added:** 7 in lexnlp/extract/en/tests/test_definition_parsing_methods_coverage.py
- **Lines now covered:** [49] (`__repr__` pin `"Acme [3, 9]"`), [65] (`does_consume_target` returns 0 for overlapping but unrelated names, plus 1/-1/disjoint pins), [383] (`The word "..." includes all things.` → `[]`, with `trim_defined_term("...", ...)` returning `""`), [389] (`The word ";" includes all things.` → `[]`; `";"` survives trimming but fails the punctuation-strip guard), [400] (`"called" means something.` → `[]`; `"called"` tags VBN and is fully consumed as an introduction)
- **Unreachable:** [454] (fallback `return [(term, term_start, term_end)]` when the phrase finder returns fewer coords than matches). `PhrasePositionFinder.find_phrase_in_source_text` builds its result as `[(p, 0, 0) for p in phrases]` and only ever updates entries in place, so it always returns exactly `len(matches)` entries — the `<` comparison can never be true.
- **Notes:** New-tests-only cov run leaves only the 454 range missing for this module. Reachability of 383/389/400 was verified empirically with `sys.settrace` before writing the tests.

### lexnlp/extract/en/pii.py
- **Tests added:** 6 in lexnlp/extract/en/tests/test_pii_coverage.py
- **Lines now covered:** [61] (`get_ssn_list`, plain + `return_sources`), [79] (`get_ssn_annotation_list`, number/text/coords `(9, 20)` pins), [101] (`get_us_phone_list`, plain + `return_sources`), [127] (`get_us_phone_annotation_list`, phone/text/coords `(44, 56)` pins), [161] (`get_pii_list`, plain + `return_sources` 3-tuples), [177] (`get_pii_annotation_list`, record-type and coords pins)
- **Unreachable:** none
- **Notes:** New-tests-only cov run leaves only line 70 (all-zero SSN `continue`, covered by the existing suite) missing for this module.

### lexnlp/utils/pandas_config.py
- **Tests added:** 4 in lexnlp/utils/tests/test_pandas_config_coverage.py
- **Lines now covered:** [65, 66] (`convert_to_arrow` forwards `dtype_backend="pyarrow"` via a recording frame), [67, 68] (`TypeError` from `convert_dtypes` returns the frame unchanged, identity-pinned), [87, 88] (`read_csv_arrow` forwards `dtype_backend="pyarrow"` and preserves an explicit `dtype_backend="numpy_nullable"` via `setdefault`, with `pandas.read_csv` spied)
- **Unreachable:** none
- **Notes:** pyarrow is not installed in this environment (and installs/network are off-limits), so the `import pyarrow` availability probes are satisfied with a stub `sys.modules["pyarrow"]` entry — the only mocking used, limited to exactly what the missing dependency requires. The pre-existing `test_uses_pyarrow_backend_when_available` test skips here, which is why these lines were uncovered.

### lexnlp/extract/en/constraints.py
- **Tests added:** 4 in lexnlp/extract/en/tests/test_constraints_coverage.py
- **Lines now covered:** [130] (`get_constraint_list` pins `[("within", "the value is", "")]`), [166, 167] (strict mode drops the bare-trigger sentence `"Maximum"` while relaxed mode keeps one `maximum` annotation with empty pre/post), [188] (`get_constraint_annotation_list` field/coords `(0, 20)` pins)
- **Unreachable:** [174] (`constraint = combined` when `pre + constraint` forms a longer phrase). The regex alternation is sorted longest-first, so at any position the longest phrase wins; a shorter suffix (e.g. `"less than"`) can only match where the longer phrase (e.g. `"no less than"`) failed its leading boundary, which forces `pre` to contain extra text before the prefix — so `f"{pre} {constraint}".strip()` can never equal a bare phrase. A 200k-sentence fuzz over the constraint vocabulary found zero hits.
- **Notes:** New-tests-only cov run leaves only line 174 missing for this module.

### lexnlp/utils/decorators.py
- **Tests added:** 8 in lexnlp/utils/tests/test_decorators_coverage.py
- **Lines now covered:** [36, 37, 38, 39, 40, 41, 42, 68] (inner `except` swallow via generator yielding `[1, 2]` before a mid-iteration `RuntimeError`; re-raise via `safe_failure=False`; outer `except` swallow (`[]`) and re-raise (`ValueError`) via immediately-raising plain function; direct `handle_invalid_text(func)` form returning `None`/`"EMPTY"` for `""` and `"HI"` for `"hi"`)
- **Unreachable:** none
- **Notes:** The wrapper is always a generator function, so a wrapped non-generator success yields zero items (`list(add(1, 2)) == []`) — pinned as documented real behaviour. New-tests-only cov run leaves only line 67 missing (the `@handle_invalid_text(...)` with-parens branch, covered by the existing suite).

### lexnlp/extract/common/universal_court_parser.py
- **Tests added:** 5 in lexnlp/extract/common/tests/test_universal_court_parser_coverage.py
- **Lines now covered:** [51, 52, 53, 54, 55] (`MatchFound.__repr__` with fields and with `subset = None` → `"[nil]"`), [174] (`load_courts([])` returns an empty `DataFrame`), [215] (`find_court_by_key_column` returns `None` when a `PhraseFinder` hit has no matching court row, via `PhraseFinder(["Imaginary Court of Nowhere"])`)
- **Unreachable:** none
- **Notes:** No network used — parser is built from a local temp CSV. `MatchFound(None, ...)` raises `TypeError` in `__init__`, so the `"nil"` repr branch is reached by setting `match.subset = None` post-construction. New-tests-only cov run no longer lists any of the 7 lines as missing.

### lexnlp/extract/es/identifiers.py
- **Tests added:** 10 in lexnlp/extract/es/tests/test_identifiers_coverage.py
- **Lines now covered:** [119] (`_nie_is_valid("X123")`/`""` is `False`), [125] (`_nie_is_valid("X123A567L")` is `False` — folded body `0123A567` non-digit), [132] (`_cif_is_valid("A1234")`/`""` is `False`), [150] (`P1234567D` valid / `P12345674` rejected, validator + `get_cif_annotations` text pins with coords `(12, 21)`), [155, 156, 157] (`C12345674` + `C1234567D` both valid with 2-annotation text extraction; `C12345670`/`C1234567A` rejected via digit/letter paths)
- **Unreachable:** none
- **Notes:** Short/non-digit validator inputs are unreachable via the public regexes (which only match well-formed surfaces), so those branches are tested via direct validator calls; the CIF head branches are additionally pinned through real `get_cif_annotations` text. Verified with `sys.settrace`: all 7 targets hit, none missing.

### lexnlp/extract/pt/amounts.py
- **Tests added:** 8 in lexnlp/extract/pt/tests/test_amounts_coverage.py
- **Lines now covered:** [174] (`text_to_number("!!!")`/`"   "` is `None` — non-empty text with no word tokens), [206] (`text_to_number("meio e milhão")`/`"meio e milhao"` == `Decimal("500000.0")` — fraction + `"e"` + multiplier lookahead), [272] (`get_amount_annotations("e") == []` single-`"e"` skip), [275] (`get_amount_annotations("e e") == []` with `text_to_number("e e") is None`), [294, 295] (`get_amounts("pagou 100 reais") == [Decimal("100.0000")]` + empty case), [307] (`get_amount_list("pagou 100 reais") == [Decimal("100.0000")]` + empty case)
- **Unreachable:** none
- **Notes:** `pytest --cov` cannot collect pt tests in this env (`ImportError: numpy: cannot load module more than once per process` via `pt/__init__` → pandas); all 7 lines verified hit with `sys.settrace`.

> Batch-19 overlap note: while this batch ran, commit b1be3f9 (batches 31-32)
> landed in the shared checkout containing semantically identical versions of
> all four files above (this worker's files predate that commit by seconds;
> the only remaining tree-vs-HEAD diffs are `ruff format` line-joins in the
> court-parser and pt-amounts files). No test was weakened or deleted; the
> combined run of new + pre-existing sibling tests is green (75 passed).

### lexnlp/nlp/en/transforms/tokens.py
- **Tests added:** 7 in lexnlp/nlp/en/transforms/tests/test_tokens_coverage.py (new dir, plus `tests/__init__.py`)
- **Lines now covered:** [51, 54, 55, 56, 59] (bigram/trigram counts incl. repeated-token counts and `lowercase=True`), [70] (`get_bigram_distribution` equals `get_ngram_distribution(text, 2)`), [81] (`get_trigram_distribution` equals `get_ngram_distribution(text, 3)`), [96, 99] (`get_skipgram_distribution` raises `AttributeError: ... no attribute 'skipgrams'`)
- **Unreachable:** none
- **Notes:** `get_skipgram_distribution` is genuinely broken at runtime: `nltk.util` resolves to `nltk.stem.util` in the pinned NLTK (even after explicit `import nltk.util`; `from nltk.util import skipgrams` works fine), so line 99 always raises `AttributeError`. The test pins this real behaviour; the maintainer likely wants `from nltk.util import skipgrams` in `tokens.py`. New-tests-only cov run leaves only lines 26-30, 38-42 missing (token/stem distributions, covered by the existing suite).

### lexnlp/utils/lines_processing/line_processor.py
- **Tests added:** 9 in lexnlp/utils/lines_processing/tests/test_line_processor_coverage.py
- **Lines now covered:** [24] (`LineOrPhrase.__repr__` with default and set `ending`), [34] (`SingleWord.get_end`, incl. defaults), [37] (`SingleWord.__repr__`), [96, 97] (`determine_line_length` tab handling — `"xx"*20 + "\t" + "yy"*20` line measures 81 chars), [106] (empty and single-char text fall back to `default_length` 95), [194, 195, 196] (`words_to_lowercase` lowers words, keeps separators)
- **Unreachable:** none
- **Notes:** `tail_length` values pinned (48 for the tab case, 57 for default). No network or heavy deps used.

### lexnlp/extract/common/annotations/act_annotation.py
- **Tests added:** 4 in lexnlp/extract/common/annotations/tests/test_act_annotation_coverage.py
- **Lines now covered:** [55, 56, 57, 58, 59, 60, 61, 62] (full `get_dictionary_values`: minimal annotation with `ambiguous=None` omits all optional tags; full annotation with section/year/`ambiguous=True` emits all five tags with year stringified; `ambiguous=False` emits `"False"`; empty section/`year=None` omits those tags)
- **Unreachable:** none
- **Notes:** `ambiguous=None` (outside the declared `bool` type but accepted at runtime) is the only way to skip the `"Extracted Entity Ambiguous"` tag, since the guard is `is not None` — pinned by the minimal test.

### lexnlp/extract/common/dates.py
- **Tests added:** 9 in lexnlp/extract/common/tests/test_dates_coverage.py
- **Lines now covered:** [35] (`LocaleInfoImport(Locale("en-001")).date_order == "DMY"` via a `locale_specific` entry that carries `date_order`), [38, 39] (`Locale("xx-YY")` → `ModuleNotFoundError` → `"MDY"`), [185] (`RuntimeError("Define text and language.")` for missing text and for empty language), [192, 194] (patched `get_dateparser_dates` raising `ValueError("boom")` → yields `[]`, message printed), [215] (stub classifier below threshold filters `"on January 5, 2024"` → `[]`, while the same parser with the check disabled finds it), [245] (`get_date_annotation_list` returns one `DateAnnotation` with coords `(15, 33)`, text `"on January 5, 2024"`, date `2024-01-05`, locale `"en"`)
- **Unreachable:** none
- **Notes:** The classifier stub fakes only `columns`/`predict_proba` (a model load, explicitly allowed); spans, features, and filtering go through the real code. `Locale("de-AT")` was NOT usable for line 35: its `locale_specific` entry has no `date_order` key, so that path raises `KeyError` — `en-001` (verified `DMY` in dateparser data) is the pinned case. Neighboring suites green: `lexnlp/extract/common/tests/` 182 passed; transforms/lines-processing/annotations suites 198 passed.

### lexnlp/extract/common/definitions/common_definition_patterns.py
- **Tests added:** 6 in lexnlp/extract/common/definitions/tests/test_common_definition_patterns_coverage.py (new dir, plus `tests/__init__.py`)
- **Lines now covered:** [55] (`match_acronyms("(AB)")` / direct `get_acronym_words_start` with no preceding words → `-1`, match skipped), [184, 185, 186, 187, 188, 189, 190, 191, 193] (full `collect_regex_matches`: two-match case pins names/starts/ends/probability, no-match case → `[]`)
- **Unreachable:** none
- **Notes:** Positive acronym case pinned (`"Canal del Futbol (CDF)"` → `CDF @ (0, 16)`, prob 100). New-tests-only cov run leaves just 88-100, 118-130, 154-170 missing (other helpers, covered by the existing suite).

### lexnlp/extract/en/entities/company_detector.py
- **Tests added:** 7 in lexnlp/extract/en/entities/tests/test_company_detector_coverage.py
- **Lines now covered:** [180] (basic `get_company_annotations("Acme LLC is great.")` → `[("Acme", "LLC")]`), [309] (`get_persons("Tom and Jerry Adams went home.")` → `["Tom and Jerry Adams"]` via the CC-join branch), [322] (person filter comprehension over the same result), [342] (cleanup loop over the same result), [363, 367] (`get_companies_re(..., use_sentence_splitter=False)` sentence/match loops), [417] (`get_companies_re("Bank LLC")` and `("Trust Company")` → `[]`: surviving company_name is itself a description), [476] (`get_noun_phrases("John Smith went home.")` → `["John Smith"]`), [480] (`get_noun_phrases("John and Smith went home.")` → `["John and Smith"]`)
- **Unreachable:** [170] — docstring line (`:param banlist_usage:` inside `get_company_annotations`); not an executable statement, so no coverage tool can ever flag or clear it. The enclosing function is exercised by the new tests.
- **Notes:** All-uppercase input → `[]` also pinned. Neighboring suite green: `lexnlp/extract/en/entities/tests/` 68 passed (3 pre-existing Stanford-disabled skips).

### lexnlp/extract/en/entities/nltk_maxent.py
- **Tests added:** 5 in lexnlp/extract/en/entities/tests/test_nltk_maxent_coverage.py
- **Lines now covered:** [89] (strict `get_geopolitical("US and France signed the treaty.")` → `["France"]`, short `"US"` skipped), [94] (`get_geopolitical("France & really went to Paris.")` → `["France", "Paris"]`, trailing-`" &"` strip), [131, 133, 134, 135] (`get_parties_as` match + captures read), [138, 139] (`"Acme LLC as of March 1, 2020 shall pay."` → `[]`, date party_type skipped), [142, 143] (`get_parties_as("Acme LLC as the Seller shall pay.")` raises `AttributeError`)
- **Unreachable:** none
- **Notes:** `get_parties_as` is genuinely broken at runtime: `get_companies()` yields plain tuples but the loop dereferences `ant.name`/`ant.company_type`, so ANY party_string containing a company raises `AttributeError: 'tuple' object has no attribute 'name'`. The test pins this real behaviour with `pytest.raises`; the maintainer likely wants the loop rewritten against tuple positions or `CompanyAnnotation`s. New-tests-only cov run leaves only 46, 68, 74-75, 78, 92, 98, 120 missing (other GPE/company paths, covered by the existing suite).

### lexnlp/ml/gensim_utils.py
- **Tests added:** 7 in lexnlp/ml/tests/test_gensim_utils_coverage.py
- **Lines now covered:** [33] (`DummyGensimKeyedVectors(50).vector_size == 50`), [60, 61] (`TrainingCallback` counters start at 0), [67, 68] (`on_epoch_begin` increments epoch, prints `"Started epoch 1 / 5"`), [79, 80] (`on_epoch_end` prints train time, `completed_epochs == 1`), [86, 87] (`on_train_begin` prints version/config line), [95] (`on_train_end` prints `"Ended training."`)
- **Unreachable:** none
- **Notes:** Callback model faked with `types.SimpleNamespace` (no Gensim training run needed); `capsys` pins exact/partial stdout. Module now at 100% under the new tests alone. No network or heavy deps used.
