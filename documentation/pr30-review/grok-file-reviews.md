## Batch 2 (packaging/docs/legacy lockfiles, 5 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: the pasted diffs in this batch are three-dot `merge-base...pr30` (merge-base `51c07bc`, Python 3.11 / Pipfile-still-present era), **not** two-dot HEAD (`ec55727`) vs `pr30`. Two-dot (`git diff ec55727 1d4e48f`) shows **no** `Pipfile` / `Pipfile.lock` change because HEAD already deleted both. Judgments below compare against HEAD, the GOOD state (`requires-python >=3.13,<3.15`, `uv_build`, `.skops` bundled models). Arthur, the instruction that `-` lines are "OUR CURRENT master" is wrong for this batch: those minuses are merge-base bytes.

### MANIFEST.in  (batch 2, part 1/5)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Relative to HEAD the file grows from 18 to 23 lines. Additive/good: `include constraints/model-artifact-abi.txt`, a single `recursive-include lexnlp *.csv *.json *.pickle *.pickle.gzip *.txt *.xml` (HEAD split pickle/csv vs addresses-only json/txt/xml), `recursive-include test_data/lexnlp/nlp/en/sota_segmentation *.json` for the new segmentation fixtures, and `prune documentation/docs/build`. The regression is packaging-format rollback: the PR glob adds `*.pickle.gzip` to ship `lexnlp/extract/ml/en/data/definition_model_layered.pickle.gzip` and does **not** list `*.skops`, while HEAD already re-exported the sklearn artifacts to `date_model.skops`, `addresses_clf.skops`, `page/paragraph/section/sentence_segmenter.skops`, and `title_locator.skops`. `recursive-include lexnlp/nlp/en/tests *.py` plus the `test_data/**` include also fight HEAD `[tool.uv.build-backend] source-exclude` of `test_data/**` and `**/tests/**` (HEAD does not use setuptools `MANIFEST.in` as the sdist source of truth).
- **Notes:** Keep `prune documentation/docs/build` and the sota fixture include only if we decide those JSON files belong in the sdist. Add `*.skops` (or drop MANIFEST.in entirely under `uv_build`). Do not take `*.pickle.gzip` unless we also revert the skops migration, which we should not. `exclude Pipfile` / `Pipfile.lock` lines are unchanged from HEAD and remain correct.

### MIGRATION_RUNBOOK.md  (batch 2, part 2/5)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** A full rewrite (+301/−163 in the three-dot paste; two-dot vs HEAD is +304/−216). The PR text is a better *operational* runbook than merge-base (drops `/Users/jackeames/Downloads/LexNLP`, documents `uv lock --check`, asset-manifest trust, BLAS `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, byte-identical model publishes, `ruff` + `skip_audit` + `reexport_bundled_sklearn_models.py --check-current`, min-vs-latest dependency CI). Merging it as-is would still **overwrite HEAD's already-newer runbook**: HEAD documents Python `3.13` default and `>=3.13,<3.15` with `uv_build`; the PR documents Python 3.10–3.13 (`>=3.10,<3.14`) with default validation interpreter **3.12**, and explicitly excludes 3.14 because "Gensim 4.4.0 … has no CPython 3.14 wheels" even though HEAD `pyproject.toml` already classifies 3.13 and 3.14. Producer ABI table pins `joblib==1.5.0`, `numpy==1.26.4`, `pandas==2.2.0`, `scikit-learn==1.7.2`, `scipy==1.13.0` under Python 3.12.13; HEAD runtime is `numpy>=2.3,<3`, `scikit-learn>=1.5`, `skops>=0.11`. The PR runbook never mentions `uv_build`, `.skops`, or the `[arrow]` / `[hub]` / `[ner]` extras that HEAD documents.
- **Notes:** Pasted `-` toolchain (`Python 3.11`, `>=3.10,<3.13`, Pipfile retained) is merge-base, not HEAD. Fold in: `LEXNLP_ASSET_MANIFEST` fail-closed trust, Tika **3.3.2** loopback launcher (HEAD still ships `tika-app-1.16.jar` / `tika-server-1.16.jar` — that jar bump is an upgrade, keep it), quality-gate `0.0` thresholds, `numpy._core` ABI triage. Restore HEAD Python `>=3.13,<3.15` / default 3.13 / `uv_build` / extras table / skops re-export (`load_bundled_model`, `--format skops`, `--remove-legacy`). Do not document `scikit-learn==1.7.2` as "the newest scikit-learn line that still supports [3.10]" as our policy. Do not accept the PR claim that Pipenv manifests "were removed" as if that were new — HEAD already removed `Pipfile` / `Pipfile.lock`.

### Pipfile  (batch 2, part 3/5)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 98
- **Summary:** Three-dot diff deletes the 45-line deprecated Pipenv manifest (`python_version = "3.8"`, `gensim = "==4.1.2"`, `scikit-learn = "==0.24"`, unpinned `numpy = "*"`, `nose` in dev-packages). That deletion matches work HEAD already finished: `git cat-file -e ec55727:Pipfile` fails, two-dot `git diff ec55727 1d4e48f -- Pipfile` is empty, and HEAD `ci/check_dist_contents.py` already lists `Pipfile` in `BANNED_BASENAMES`. Merging does not reintroduce or downgrade anything; both sides end with the file absent.
- **Notes:** Arthur, `-` here is **not** current master. The deleted pins (`scikit-learn==0.24`, `gensim==4.1.2`, Python 3.8) are merge-base `51c07bc` bytes we already dropped. Keep the file deleted. Do not restore it.

### Pipfile.lock  (batch 2, part 4/5)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 98
- **Summary:** Part 1/2 of deleting the 1235-line Pipenv lock (`pipfile-spec: 6`, `requires.python_version: "3.8"`). Locked defaults in this hunk include `beautifulsoup4==4.11.1`, `certifi==2022.9.24`, `charset-normalizer==2.1.1`, `click==8.1.3`, `backports.zoneinfo==0.2.1` (marker `python_version < '3.9'`). HEAD already has no `Pipfile.lock` (same two-dot empty result as `Pipfile`); `ci/check_dist_contents.py` already bans the basename. Deleting this lock is aligned with HEAD, not a downgrade.
- **Notes:** Same three-dot caveat as `Pipfile`. The hashes/versions being removed are 2022-era merge-base pins, not HEAD's `uv.lock`. Continue using `uv.lock` as the only lockfile.

### Pipfile.lock  (batch 2, part 5/5)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 98
- **Summary:** Part 2/2 of the same `Pipfile.lock` deletion. The truncated tail shows further merge-base pins going away, including `numpy==1.14.1` (`markers: python_version < '3.11'`) and `zipp==3.10.0`. Removing those ancient pins is the correct direction and is a no-op versus HEAD, which already lacks the file.
- **Notes:** Do not treat disappearance of `numpy==1.14.1` as "the PR lowering numpy"; it is the PR deleting a lock HEAD already deleted. Net merge result: `Pipfile.lock` stays gone. Canonical pins remain HEAD `pyproject.toml` (`numpy>=2.3,<3`) plus `uv.lock`.

## Batch 4 (documentation guides + curated API rst, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: for this batch two-dot `git diff ec55727 1d4e48f` and three-dot `51c07bc...pr30` are **identical** on the three modified files (`index.rst`, `lexnlp.rst`, `license.rst`); HEAD vs merge-base is empty, so the pasted `-` lines really are current master. The seven `A` files do not exist on HEAD. Judgments compare against HEAD. PR `conf.py` (not in this batch) sets `exclude_patterns = ["api/**", "build/**"]`, so `modules/api/` becomes the canonical autodoc surface and omissions of HEAD-only modules (`lexnlp.extract.ner`, `lexnlp.ml.catalog.hub`, `lexnlp.ml.model_io`) matter.

### documentation/docs/source/guides/segmentation_current_api.rst  (batch 4, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 91
- **Summary:** New 164-line guide. It does not replace any HEAD page. It documents the real 2.3-compatible English surface (`get_sentence_span`, `get_paragraph_spans`, `get_section_spans` / `DocumentSection`, `get_pages`, `get_titles`) with an accurate capability matrix, half-open string offsets, `safe_failure` behaviour, and the feature-width fallback that yields a single paragraph span. Those functions exist on both HEAD and PR. The additive `SaTSentenceSegmenter` import path matches PR `lexnlp.nlp.en.segments.__init__` / `backends.py` (`backend_id`, default `split_on_input_newlines=False`, exact consecutive partition). Keep the page; it is the right place to freeze legacy contracts next to the 2.4.0a1 hierarchy.
- **Notes:** Example calls `make_pinned_sat_model()` but that name exists only in this file and `guides/lossless_segmentation_api.rst` — it is not a library symbol. Replace it with an explicit caller-constructed `SplitModel` (or a one-line comment that the application pins `wtpsplit.SaT` itself). The `assert sentence == text[start:end]` example is true by construction (`substring = text[start:end]`); the later “not a lossless partition” caveat is about omitted inter-sentence whitespace/OCR fragments, not that equality. Title underline / 2.4.0a1 versioning is PR-side and should stay consistent with whatever version we actually ship, not HEAD `2.3.0`.

### documentation/docs/source/index.rst  (batch 4, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Two-dot matches the paste: RST underline cleanup (`Table of Contents` underline lengthened from 12 to 17), toctree indent normalised, blank `|` spacers removed, and three new toctree entries (`guides/segmentation_current_api`, `guides/lossless_segmentation_api`, `development/segmentation_v2_design`). Those three files exist on PR. Mode `100755` → `100644` is a fix (HEAD docs sources are wrongly executable). Nothing HEAD documented is dropped.
- **Notes:** Overline for “Welcome to the LexNLP documentation!” is still short (36 `=` vs 38-character title; HEAD was 34). Harmless Sphinx warning, not a reason to reject. Do not drop the new guide entries if the corresponding rst files are kept.

### documentation/docs/source/lexnlp.rst  (batch 4, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Single additive toctree line `modules/api/index` under the existing `modules/extract/extract` and `modules/nlp/nlp` entries. HEAD file is otherwise unchanged (logo, maxdepth 4). This is how the new curated API reference is wired in. Mode stays `100755` (pre-existing on HEAD, not introduced here).
- **Notes:** Keep extract/nlp guides in this toctree; the new API pages explicitly say they cover modules those guides do not.

### documentation/docs/source/license.rst  (batch 4, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** RST heading cleanup only, plus one URL change: `https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE` → `.../blob/master/LICENSE`. AGPLv3 text and ContraxSuite licensing contact are unchanged. This is not a dependency or API downgrade. HEAD package version is still `2.3.0` and HEAD `__license__` points at the `2.3.0` blob; PR `__init__.__license__` points at `2.4.0a1` while this page points at moving `master`. Take the heading fixes; do not take an unversioned `master` URL as the canonical license pointer.
- **Notes:** Prefer a versioned blob (`2.3.0` until we actually cut a tag, or `2.4.0a1` if that tag exists on the license repo we intend to cite). A `2.4.0a1` path on `LexPredict/lexpredict-lexnlp` may 404. Mode remains `100755`.

### documentation/docs/source/modules/api/annotations.rst  (batch 4, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 93
- **Summary:** New curated autodoc page. Every `automodule` target exists on both HEAD and PR (`annotation_locator_type`, `annotation_type`, `text_annotation`, the fact annotation classes, `fact_extracting.ExtractingFunction` / `FactExtractor`, `datefinder`, `universal_definition_parser`, `durations_parser`, `ocr_rating_calculator`). The listed annotation types match `lexnlp/extract/common/annotations/` on both sides (act through url). Intentionally omits tests and `phrase_position_finder.py`, which matches the index blurb (“omits tests, private helpers”). Additive; does not delete HEAD docs.
- **Notes:** After merge, Sphinx autodoc will pull live signatures from whatever code we keep (HEAD type hints vs PR). No edit required unless we want `phrase_position_finder` documented.

### documentation/docs/source/modules/api/core.rst  (batch 4, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 92
- **Summary:** New 30-line autodoc of `lexnlp`, `lexnlp.config.en.company_types`, `lexnlp.config.en.geoentities_config`, and `lexnlp.config.stanford`. All four modules exist on HEAD and PR. Useful; Stanford remains an optional gated extra in runtime (`LEXNLP_USE_STANFORD`) and documenting the config module does not re-enable it.
- **Notes:** Does not mention HEAD `DEFAULT_MODELS_REPO` / `get_models_repo()` beyond whatever `automodule:: lexnlp` emits. Fine.

### documentation/docs/source/modules/api/extraction.rst  (batch 4, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** New 188-line autodoc for `lexnlp.extract.all_locales.*` (amounts through percents), English extras (`addresses`, `contracts.predictors`, selected `dict_entities` names including `AliasBanRecord` / `alias_is_banlisted` / `prepare_alias_banlist_dict` — those HEAD names, not the old `blacklist` sphinx-apidoc pages), `nltk_re`, `nltk_tokenizer`, `stanford_ner`, `span_tokenizer`, plus de/es extractors whose `get_*` symbols match the PR modules. Good additive reference for locales that lacked dedicated guides. The regression is coverage of **our** public surface: HEAD `lexnlp.extract.ner` (`extract_entities`, `prefer_spacy`, `[ner]` extra) is absent from PR and therefore from this page, and English entities also omit `nltk_maxent`, `company_detector`, and `company_np_extractor` which still exist on both trees. Because PR `conf.py` excludes the old `api/` dump, this page would become the public extraction index and would hide the hybrid NER fallback.
- **Notes:** After keeping HEAD `lexnlp/extract/ner/`, add `.. automodule:: lexnlp.extract.ner`. `dict_entities` symbols in the page match HEAD (`AliasBanList`, not `alias_is_blacklisted`). Do not restore the generated `api/lexnlp.extract.en.dict_entities.alias_is_blacklisted.rst` names.

### documentation/docs/source/modules/api/index.rst  (batch 4, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 94
- **Summary:** New toctree hub (`core`, `extraction`, `annotations`, `machine_learning`, `nlp`, `utilities`) with a correct statement that tests, private helpers, and one-page-per-constant output are omitted. All six children exist on PR (`utilities.rst` is not in this batch). This is the intended replacement for the obsolete `documentation/docs/source/api/` sphinx-apidoc snapshot.
- **Notes:** If we add hub/NER/skops pages, link them here. Do not re-include `api/**` in the toctree; PR `conf.py` excludes that tree for good reason (tests, removed modules, per-constant pages).

### documentation/docs/source/modules/api/machine_learning.rst  (batch 4, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 91
- **Summary:** New autodoc of modules that exist on both trees (`lexnlp.ml.catalog`, `catalog.download`, `predictor`, `normalizers`, `vectorizers`, `sklearn_transformers`, `gensim_utils`, and the extract.ml classifier/detector/definition stack). Useful. It does **not** document HEAD `lexnlp.ml.catalog.hub` (`get_path_from_hub`, `[hub]` extra, `huggingface_hub`) or HEAD `lexnlp.ml.model_io` / `model_card` / `sklearn_config` (skops load path). It also skips the PR’s own replacements `lexnlp.ml.artifact_io` and `lexnlp.ml.artifact_abi`. As the curated ML API page, shipping it unchanged would present GitHub-release `catalog.download` + `gensim_utils` as the whole ML surface and omit both our Hub/skops work and the PR’s artifact ABI layer.
- **Notes:** After merge, add automodules for whichever I/O layer we keep: HEAD `lexnlp.ml.model_io` + `lexnlp.ml.catalog.hub`, and/or PR `lexnlp.ml.artifact_io` + `artifact_abi` if those survive. Keep `gensim_utils` only while `lexnlp/ml/gensim_utils.py` remains (it does on HEAD). Do not document pickle-only loading as the supported bundled-model path if we keep `.skops`.

### documentation/docs/source/modules/api/nlp.rst  (batch 4, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** New 23-line autodoc of `lexnlp.nlp.en.stanford`, `lexnlp.nlp.en.segments.heading_heuristics`, and `lexnlp.nlp.train.en.train_section_segmanizer`. All three modules exist on HEAD and PR under those exact names (the `segmanizer` spelling is the historical filename, also used by HEAD `test_section_segmanizer_normalization.py`). The new hierarchy/chunk/payload APIs are correctly left to `guides/lossless_segmentation_api.rst` rather than duplicated here. Documenting Stanford does not reintroduce the Java dependency; it already ships behind `LEXNLP_USE_STANFORD`.
- **Notes:** `heading_heuristics.py` itself has a small two-dot style diff (quote style / line wrap, plus a `# -*- coding: utf-8 -*-` header) that is not this rst file. No need to reject the page.

## Batch 6 (extract/en Sphinx guides: conditions through geoentities, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: two-dot `git diff ec55727 1d4e48f` **matches** the pasted three-dot hunks for all ten files, and `git diff 51c07bc ec55727` is empty on this set. The pasted `-` lines **are** current master. These pages were not modernised on HEAD; the PR is RST underline hygiene except `copyright.rst`, which also retargets autodoc. `get_*` symbols cited below exist on HEAD `lexnlp/extract/en/*.py`. Mode stays `100755` except `copyright.rst` (`100755` → `100644`).

### documentation/docs/source/modules/extract/en/conditions.rst  (batch 6, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Two-dot is underline-only on the same 94-line file: title over/under-line `============` (12) → 70 `=` to match `:mod:`lexnlp.extract.en.conditions`: Extracting conditional statements`; `Extracting conditions` 16 → 21 dashes; `Customizing conditional statement extraction` 16 → 44 dashes. `.. autofunction:: get_conditions` is unchanged and matches HEAD `def get_conditions(`. No API, dependency, or example text is rewritten.
- **Notes:** Keep. Mode remains `100755` (pre-existing on HEAD).

### documentation/docs/source/modules/extract/en/constraints.rst  (batch 6, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Same 113-line page; only RST heading rules. Title over/under-line 12 → 70 `=` for `:mod:`lexnlp.extract.en.constraints`: Extracting constraint statements`; `Extracting constraints` 16 → 22 dashes; `Customizing constraint statement extraction` underline lengthened to the title. `.. autofunction:: get_constraints` still matches HEAD `def get_constraints(`.
- **Notes:** Harmless Sphinx warning fix. Mode stays `100755`.

### documentation/docs/source/modules/extract/en/copyright.rst  (batch 6, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** The only content rewrite in this batch (33 → 25 lines; mode `100755` → `100644`). HEAD autodoc is `get_copyright`, a symbol that does **not** exist on HEAD or PR (`lexnlp/extract/en/copyright.py` exports `get_copyrights` / `get_copyright_list` / `get_copyright_annotations` / `get_copyright_annotation_list`; no `get_copyright` alias). HEAD’s doctest also calls `lexnlp.extract.en.conditions.get_conditions` on the © example. The PR retargets `.. autofunction:: get_copyrights`, notes optional source-text retention (`return_sources`), and gives a `code-block` that actually imports `get_copyrights`. That is a docs bugfix, not a rollback. What to restore: the sibling-style unit-test URL (`test_data/lexnlp/extract/en/tests/test_copyright`) and a printed Hughes/`©` example using `get_copyrights` (HEAD tuple shape ` (sign, date, name) ` is still what HEAD `get_copyrights` yields).
- **Notes:** Do **not** put `get_copyright` back. Do not take the PR’s `copyright.py` typing rollback (`list[tuple[...]]` → `List[Union[Tuple[...]]]`) as part of this rst merge; that is a different file. Keep the `100644` mode.

### documentation/docs/source/modules/extract/en/courts.rst  (batch 6, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** 63-line file, underline-only. Title over/under-line 12 → 60 `=` for `:mod:`lexnlp.extract.en.courts`: Extracting court references`; `Extracting courts` 16 → 17 dashes. `.. autofunction:: get_courts` matches HEAD `def get_courts(text: str, language: str = "en")`.
- **Notes:** Mode stays `100755`.

### documentation/docs/source/modules/extract/en/cusip.rst  (batch 6, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** 58-line file, underline-only. Title over/under-line 12 → 48 `=` for `:mod:`lexnlp.extract.en.cusip`: Extracting CUSIP`; subsection underline 16 → 21 dashes. `.. autofunction:: get_cusip_list` matches HEAD `def get_cusip_list(text: str) -> list[dict[str, Any]]`. The PR does not change the (pre-existing, copy-pasted) subsection title `Extracting conditions`.
- **Notes:** Optional while cherry-picking: rename that heading to `Extracting CUSIPs`. Not a regression. Mode `100755`.

### documentation/docs/source/modules/extract/en/dates.rst  (batch 6, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** 61-line file, underline-only. Title over/under-line 12 → 58 `=` for `:mod:`lexnlp.extract.en.dates`: Extracting date references`; `Advanced usage and customization` 16 → 32 dashes. `Extracting dates` was already 16/16 and is untouched. Neither side mentions pickle vs `.skops`. `get_dates` still exists on HEAD.
- **Notes:** Out of scope for this rst: two-dot `lexnlp/extract/en/dates.py` is a large rewrite (+784/−418). Do not treat this heading fix as approval of that Python change. Mode `100755`.

### documentation/docs/source/modules/extract/en/definitions.rst  (batch 6, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** 42-line file, underline-only. Title over/under-line 12 → 70 `=` for `:mod:`lexnlp.extract.en.definitions`: Extracting definition statements`; subsection underline 16 → 22 dashes. `.. autofunction:: get_definitions` matches HEAD `def get_definitions(`. Pre-existing copy-paste heading `Extracting constraints` is unchanged.
- **Notes:** Optional rename to `Extracting definitions`. Mode `100755`.

### documentation/docs/source/modules/extract/en/distances.rst  (batch 6, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** 74-line file, underline-only. Title over/under-line 12 → 56 `=` for `:mod:`lexnlp.extract.en.distances`: Extracting distances`; `Extracting conditions` 16 → 21 dashes (title text still wrong); `Customizing distance extraction` 16 → 31 dashes. `.. autofunction:: get_distances` matches HEAD `def get_distances(`.
- **Notes:** Optional rename of `Extracting conditions` → `Extracting distances`. Mode `100755`.

### documentation/docs/source/modules/extract/en/durations.rst  (batch 6, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** 73-line file, underline-only. Title over/under-line 12 → 56 `=` for `:mod:`lexnlp.extract.en.durations`: Extracting durations`; `Extracting constraints` 16 → 22 dashes (title text still wrong); `Customizing duration statement extraction` 16 → 41 dashes. `.. autofunction:: get_durations` matches HEAD `def get_durations(`. Example tuple `[('second', 12, 0.0001388888888888889)]` is unchanged.
- **Notes:** Optional rename of `Extracting constraints` → `Extracting durations`. Mode `100755`.

### documentation/docs/source/modules/extract/en/geoentities.rst  (batch 6, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** 32-line file, underline-only. Title over/under-line 12 → 85 `=` for `:mod:`lexnlp.extract.en.geoentities`: Extracting geographic and geopolitical entities`; `Extracting courts` 16 → 17 dashes (title text still the courts copy-paste). `.. autofunction:: get_geoentities` matches HEAD `def get_geoentities(`.
- **Notes:** Optional rename of `Extracting courts` → `Extracting geoentities`. Mode `100755`.

## Batch 8 (nlp Sphinx guides + root index.rst, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: for the nine `documentation/docs/source/modules/nlp/**` files, two-dot `git diff ec55727 1d4e48f` **matches** the pasted three-dot hunks (`git diff 51c07bc ec55727` is empty on that set). Those pasted `-` lines **are** current master. For root `index.rst` they are **not**: merge-base → HEAD is `+36/−2` (Python `>=3.13,<3.15`, extras table, `lexnlp.extract.ner`, `.skops`); two-dot vs PR is `+3/−179`. HEAD `documentation/docs/source/conf.py` already comments out `sphinx_automodapi.automodapi` / `smart_resolver` and `pyproject.toml` does not depend on `sphinx-automodapi`, so HEAD's remaining `.. automodapi::` directives are dead. PR `conf.py` enables only `sphinx.ext.autodoc` + `autosummary`. Judgments compare against HEAD.

### documentation/docs/source/modules/nlp/en/segments_paragraphs.rst  (batch 8, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Two-dot is RST hygiene plus a copy-paste title fix: HEAD over/under-line is 12 `=` wrapping `:mod:`lexnlp.nlp.en.segments.pages`: Segmenting paragraphs in text`; the PR retitles to `:mod:`lexnlp.nlp.en.segments.paragraphs`` and lengthens the rule to 71 `=`. Body text already named `paragraphs`. Autodoc `.. automodapi:: lexnlp.nlp.en.segments.paragraphs` / `:include-all-objects:` becomes `.. automodule::` / `:members:`, which is the directive that HEAD's enabled `sphinx.ext.autodoc` actually implements. Mode `100755` → `100644`. Public HEAD symbols (`get_paragraphs`, `get_paragraph_list`, `get_paragraph_spans`, `get_paragraph_span_list`, `build_paragraph_break_features`) remain `def`s in `paragraphs.py` and will still be documented.
- **Notes:** Do not treat this rst as approval of the two-dot `paragraphs.py` rewrite (PR adds `_normalise_paragraph_breaks` and types `List`/`Tuple` instead of HEAD `list`/`tuple`). Keep the `100644` mode.

### documentation/docs/source/modules/nlp/en/segments_sections.rst  (batch 8, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Same 22→21-line page. Title over/under-line 12 → 67 `=` to match `:mod:`lexnlp.nlp.en.segments.sections`: Segmenting sections in text`. Autodoc swap `automodapi` + `:include-all-objects:` → `automodule` + `:members:`. HEAD `DocumentSection`, `get_sections`, `get_section_spans`, `get_sections_re` stay defined in `sections.py`. Mode remains `100755`.
- **Notes:** Optional while cherry-picking: drop the executable bit like `segments_paragraphs.rst`. Out of scope: two-dot `sections.py` typing (`List[DocumentSection]` vs HEAD builtins).

### documentation/docs/source/modules/nlp/en/segments_sentences.rst  (batch 8, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Copy-paste title fix plus the same autodoc/underline pattern. HEAD title is `:mod:`lexnlp.nlp.en.segments.sections`: Segmenting sentences in text`; PR corrects the module to `sentences` and lengthens the rule to 69 `=`. `.. automodule:: lexnlp.nlp.en.segments.sentences` / `:members:` replaces dead `automodapi`. Mode `100755` → `100644`. HEAD `get_sentence_span`, `get_sentence_span_list`, `get_sentences`, `get_sentence_list`, `build_sentence_model` remain in `sentences.py`.
- **Notes:** Keep the `100644` mode. Do not take the PR `sentences.py` `Union`/`Tuple`/`List` rollback with this rst.

### documentation/docs/source/modules/nlp/en/segments_titles.rst  (batch 8, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Underline-only plus autodoc retarget. Title over/under-line 12 → 79 `=` for `:mod:`lexnlp.nlp.en.segments.titles`: Segmenting and identifying titles in text`. `automodapi` / `:include-all-objects:` → `automodule` / `:members:`. Narrative and GitHub/email links are unchanged. Mode stays `100755`.
- **Notes:** Optional `chmod` to `100644`. The `.. _nlp_en_segments_titles:` label is unchanged, so `nlp.rst` `:ref:` links keep working.

### documentation/docs/source/modules/nlp/en/segments_utils.rst  (batch 8, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** Same 22→21-line page. Title over/under-line 12 → 61 `=` for `:mod:`lexnlp.nlp.en.segments.utils`: Utilities for segmenting`. Autodoc swap to `.. automodule:: lexnlp.nlp.en.segments.utils` / `:members:`. No API text rewritten. Mode `100755`.
- **Notes:** `:members:` will document functions defined in `utils.py`; it will not pull in imported names the way automodapi `:include-all-objects:` would. That is the correct match for HEAD's autodoc-only conf.

### documentation/docs/source/modules/nlp/en/tokens.rst  (batch 8, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** 123→122 lines. Title over/under-line 12 → 48 `=` for `:mod:`lexnlp.nlp.en.tokens`: Working with tokens`; subsection rules for `Tokenizing text`, `Stemming and lemmatizing text`, `Working with parts-of-speech`, and `Collocations` are lengthened to the heading. Prose, NLTK `word_tokenize` / `pos_tag` links, and the `COLLOCATION_SIZE` note are unchanged. Autodoc `automodapi` / `:include-all-objects:` → `automodule` / `:members:`. HEAD `get_tokens`, `get_stems`, `get_lemmas`, `get_nouns`/`get_verbs`/`get_adjectives`/`get_adverbs`, and `COLLOCATION_SIZE = 10000` still exist. Mode `100755`.
- **Notes:** `automodule :members:` omits undocumented module-level data (`STOPWORDS`, `BIGRAM_COLLOCATIONS`, `TRIGRAM_COLLOCATIONS`, `COLLOCATION_SIZE`) unless we add `:undoc-members:`. Optional. Do not take the PR `tokens.py` `List` annotation rollback with this rst. Both sides still `pickle.load` `stopwords.pickle` / collocation pickles — that loader is a different file.

### documentation/docs/source/modules/nlp/en/transforms_character.rst  (batch 8, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** 23→22 lines. Title over/under-line 12 → 94 `=` for `:mod:`lexnlp.nlp.en.transforms.characters`: Transforming text into character-oriented features`. Autodoc swap to `.. automodule:: lexnlp.nlp.en.transforms.characters` / `:members:`. The `.. _nlp_en_transforms_characters:` label (plural `characters`) is unchanged and still matches `nlp.rst`. Mode `100755`.
- **Notes:** Filename stays `transforms_character.rst` (singular) on both sides. Not a regression.

### documentation/docs/source/modules/nlp/en/transforms_tokens.rst  (batch 8, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** 23→22 lines. Title over/under-line 12 → 86 `=` for `:mod:`lexnlp.nlp.en.transforms.tokens`: Transforming text into token-oriented features`. Autodoc swap to `.. automodule:: lexnlp.nlp.en.transforms.tokens` / `:members:`. Narrative unchanged. Mode `100755`.
- **Notes:** Same optional `chmod` as the other leftover `100755` pages.

### documentation/docs/source/modules/nlp/nlp.rst  (batch 8, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 91
- **Summary:** Two-dot matches the paste (`+29/−27`; mode `100755` → `100644`). RST title rule 12 → 46 `=`. The language blurb goes from “Currently, the following languages are stable: English: `lexnlp.nlp.en`” to “English components are available under `:mod:`lexnlp.nlp.en``” — accurate: HEAD `lexnlp/nlp/` is still only `en/` + `train/` (Portuguese lives under `lexnlp.extract.pt`, not here). The WIP `.. attention::` GitHub/email box is dropped. New prose points at `:ref:`segmentation_current_api``, `:ref:`lossless_segmentation_api``, and `:ref:`segmentation_v2_design`` (PR-only pages; batch 4 already said keep them). A hidden `:glob:` toctree `en/*` actually registers the child pages that HEAD only `:ref:`-linked. All six segment/transform `:ref:` targets remain.
- **Notes:** Cherry-pick this file **together with** `guides/segmentation_current_api.rst`, `guides/lossless_segmentation_api.rst`, and `development/segmentation_v2_design.rst`, or drop those three `:ref:` lines so Sphinx does not warn. Do not read the English-only reword as deleting `lexnlp.extract.de`/`es`/`pt`. Keep `100644`.

### index.rst  (batch 8, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 94
- **Summary:** Arthur, the pasted `+3/−145` is merge-base, not HEAD. True two-dot is `+3/−179`: HEAD's 180-line root `index.rst` (Python **3.13** / ``>=3.13,<3.15``, ``uv``, extras ``[arrow]`` ``pyarrow>=17`` / ``[hub]`` ``huggingface_hub>=0.25`` / ``[ner]`` ``spacy>=3.7`` / ``[tika]`` ``tika>=2.6.0`` / ``[stanford]``, `lexnlp.extract.ner` + `prefer_spacy=True`, bundled artifacts ``.pickle`` → ``.skops`` via `lexnlp.ml.model_io.load_bundled_model`) is replaced by a four-line `.. include:: README.rst` wrapper. The DRY include is the right shape — HEAD `README.rst` is already the superset of this file (`git diff ec55727:index.rst ec55727:README.rst` is `+59/−15`, including Quick Setup and the Pipfile ban). It is a regression only if the include target is the **PR** `README.rst`, which documents Python **3.10–3.13**, default `uv python install 3.12`, and “Persisted Python model files use pickle/joblib formats”.
- **Notes:** Cherry-pick the four-line wrapper onto HEAD and **leave HEAD `README.rst` in place**. Do not take PR `README.rst` as the include target. `MANIFEST.in` `include index.rst` and `scripts/create_release_branch.sh` (`cp …/index.rst`) still work on the stub. Sphinx's master doc is `documentation/docs/source/index.rst` (batch 4), not this root file.

## Batch 10 (all_locales dispatch + common annotations/copyrights, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs are three-dot `51c07bc...pr30`. Judgments below use two-dot `git diff ec55727 1d4e48f`. Several pasted “fixes” (`AddressAnnotation.__int__` → `__init__`, citation `if self.reporter` → `if self.court`/`if self.source`, CUSIP checksum tag writing `self.ppn` → `self.checksum`, unused `from typing import List` in `copyright_parser.py`) are **already on HEAD**. Net PR effect on those files is type-hint / constructor rollback, not the bugfix.

### lexnlp/extract/all_locales/money.py  (batch 10, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** Two-dot is `+22/−7`, larger than the paste. The real dispatch change is small and mostly good: replace `ROUTINE_BY_LOCALE.get(Locale(locale).language, ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code])` plus positional `routine(text, float_digits)` with `get_language_routine(...)` and keyword `routine(text=text, float_digits=float_digits)`. EN/DE `get_money_annotations(text, float_digits=4)` both accept those names, so the keyword call is a robustness win. The rest undoes HEAD modernisation: `from collections.abc import Generator` + `Generator[MoneyAnnotation]` become `from typing import Generator` + `Generator[MoneyAnnotation, None, None]`, and a redundant `# -*- coding: utf-8 -*-` cookie is re-added. Neither side wires `lexnlp.extract.pt.money` (HEAD already has that module); this file is not the PT deletion.
- **Notes:** Keep keyword dispatch. Take `get_language_routine` only together with `languages.py` **without** dropping `LANG_PT`. Restore `collections.abc.Generator` / `Generator[MoneyAnnotation]`. Drop the coding cookie.

### lexnlp/extract/all_locales/percents.py  (batch 10, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** Same two-dot shape as money (`+22/−7`). Keyword `routine(text=text, float_digits=float_digits)` matches both EN and DE `get_percent_annotations(text, float_digits=4)`. Helper vs HEAD `.get(...)` is behaviour-equivalent. Typing rollback is the regression: `collections.abc.Generator` / `Generator[PercentAnnotation]` → `typing.Generator` / `Generator[PercentAnnotation, None, None]`. Portuguese `lexnlp.extract.pt.percents` exists on HEAD and is still unwired here on both sides.
- **Notes:** Same cherry-pick rule as money.py. Do not treat the paste’s unused-`Locale` import swap as HEAD losing `Locale`; HEAD still uses `Locale` for the `.get` lookup.

### lexnlp/extract/all_locales/tests/test_dispatch.py  (batch 10, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** New 125-line file; HEAD has no counterpart. Tests lock in named-arg DE amount dispatch (`return_sources` mapped from `extended_sources`), DE date dispatch (`strict`/`locale`/`base_date`/`threshold` with a `Locale` object), English fallback for unknown locales (`fr-FR` → `en`), and court-citation language defaulting plus German fallback. Those are real contracts the PR’s `amounts.py`/`dates.py`/`court_citations.py` implement and HEAD still calls positionally (`routine(text, extended_sources, float_digits)`), so this file **fails on HEAD as-is**. It does not itself delete anything.
- **Notes:** Cherry-pick **with** the matching all-locale dispatchers. Do **not** take the rest of the PR tests dir: two-dot on `lexnlp/extract/all_locales/tests/` is `+157/−1635` and deletes HEAD `test_pt_config_csv.py`, `test_pt_csv.py`, `test_pt_locale_routing.py`, `test_pt_routing.py`, `test_languages_extras.py`, and `test_locale_context_manager.py`. Keep those.

### lexnlp/extract/all_locales/tests/test_locales.py  (batch 10, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Two-dot `+32/−8`. Additive test `test_unsupported_locale_uses_explicit_fallback` is the right pin for `get_language_routine('fr-FR', …, LANG_EN|LANG_DE)` and language-hit-over-fallback (`de-DE` with only `'de'` still returns German). Existing `test_locales_convert` cases are unchanged. Quote-style churn (`"en"` → `'en'`) is noise. The new test imports `get_language_routine`, which HEAD `languages.py` does not define.
- **Notes:** Keep the new test only if we keep the helper (and `LANG_PT` on `languages.py`). Restore HEAD double quotes. Do not let this file ride in with the PT test deletions listed above.

### lexnlp/extract/common/annotations/address_annotation.py  (batch 10, part 5/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 96
- **Summary:** Arthur, the pasted `__int__` → `__init__` is merge-base, not HEAD. HEAD already renamed the constructor (see `lexnlp/extract/common/annotations/tests/test_address_annotation_ctor.py`, which the PR tree **deletes**). Two-dot vs HEAD replaces the documented API `AddressAnnotation(coords: tuple[int, int], locale: str = "en", text: str = "")` — name hardcoded `""` into `TextAnnotation` — with required `(name: str, locale: str, coords: Tuple[int, int], text: str = '')`, drops the docstring, and rolls `tuple[int, int]` / `list[str]` back to `typing.Tuple` / `List`. That breaks HEAD’s ctor tests (`AddressAnnotation(coords=(10, 25), text="221B Baker Street")` plus default locale `"en"`).
- **Notes:** HEAD `lexnlp/extract/en/addresses/addresses.py` currently passes `name=""` into this class, which HEAD’s signature does not accept. That is a HEAD bug; the fix is an optional `name: str = ""` keyword on the **HEAD** constructor, not the PR’s required name-first signature. Do not take this file.

### lexnlp/extract/common/annotations/citation_annotation.py  (batch 10, part 6/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 97
- **Summary:** The paste’s `if self.reporter:` → `if self.court:` / `if self.source:` is already on HEAD (`ec55727` `get_dictionary_values` plus `tests/test_citation_annotation_bugfix.py`). Two-dot vs HEAD does **not** change those guards. It does strip the constructor docstring, replace PEP 604 `int | None = None` / `str | None = None` / `tuple[int, int]` / `list[str]` / `dict[str, Any]` with `int = None` / `str = None` / `Tuple` / `List` / `Dict`, and drop `from typing import Any` in favour of the old typing barrel. That is a type-hint and docs regression with no behaviour win.
- **Notes:** Keep HEAD. The court/source independence tests would still pass against the PR body of `get_dictionary_values`, but taking the PR file would delete those tests (PR has no `annotations/tests/` tree) and weaken the constructor types those tests construct (`reporter=None`, `court="SCOTUS"`).

### lexnlp/extract/common/annotations/cusip_annotation.py  (batch 10, part 7/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 97
- **Summary:** Merge-base really did write `df.tags['Extracted Entity Checksum'] = self.ppn` under `if self.checksum:`. HEAD already fixed that to `if self.checksum is not None: … = self.checksum` (checksum digit `0` must not vanish) and went further: `ppn: str | bool | None`, `checksum: str | int | None`, and `get_cite_value_parts` coerces non-string `ppn` to `""` so a boolean private-placement flag never leaks into the cite path. Two-dot `+46/−75` **reverts** those types to `ppn: str = None` / `checksum: str = None`, restores `self.ppn or ''` in cite parts (bool `True` becomes `"True"`), and drops the docstring. The checksum-tag line on the PR matches HEAD; everything around it is a downgrade.
- **Notes:** HEAD `tests/test_cusip_annotation_optional.py` pins `coords`-first construction and `None` defaults. Do not take this file. The paste is not the net merge.

### lexnlp/extract/common/annotations/text_annotation.py  (batch 10, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** Two-dot `+20/−15`. The one real upgrade vs HEAD is `get_int_value`: HEAD still has a pylint-suppressed bare `except:`; the PR uses `except Exception:` (E722 / PEP 8, still swallows `ValueError`/`TypeError` plus everything else except `BaseException`). The rest is modernisation rollback: `coords: tuple[int, int]` → `Tuple[int, int]`, `list[str]` → `List[str]`, re-import `typing.Tuple, List`, and slice spacing `coords[0] : coords[1]` → `coords[0]: coords[1]`. `safe_cast` stays `except (ValueError, TypeError)` on both sides.
- **Notes:** Cherry-pick only the `except Exception:` (or, better, match `safe_cast` and catch `(ValueError, TypeError)`). Keep HEAD PEP 604 builtins. Do not take the typing imports.

### lexnlp/extract/common/copyrights/copyright_parser.py  (batch 10, part 9/10)
- **Regression:** NO
- **Verdict:** REJECT
- **Confidence:** 91
- **Summary:** Paste deletes unused `from typing import List` — already gone on HEAD (`git show ec55727:…/copyright_parser.py` has no typing import). Two-dot is formatting only: extra blank line after the module header and `phrase.text[ptrn.start : ptrn.end]` → `phrase.text[ptrn.start: ptrn.end]`. Constructor still passes `name=ptrn.name`, `coords`, `text`, `locale`. No behaviour change, no unused-import win vs HEAD.
- **Notes:** Leave HEAD. Taking the PR file fights HEAD Black/ruff slice spacing for zero gain.

### lexnlp/extract/common/copyrights/copyright_parsing_methods.py  (batch 10, part 10/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 94
- **Summary:** Paste looks like dropping unused `Pattern` from `from typing import Pattern, List, Tuple`. HEAD already dropped that import and uses PEP 604 `list[PatternFound]` / `list[tuple[int, int, int]]`. Two-dot `+34/−34` **re-adds** `from typing import List, Tuple` and rewrites every return type back to `List[...]`. Regexes, `pre_process_found_matches`, and company-name slicing are the same. `# type: Pattern` comments remain unresolved on both sides (PR still does not import `Pattern`).
- **Notes:** Keep HEAD. The `# pylint: disable/enable=unused-import` sandwich on HEAD is leftover noise; do not “fix” it by restoring `typing.List`.

## Batch 12 (DE extractors + beautifier + pickle artifacts, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs are three-dot `51c07bc...pr30`. Judgments below use two-dot `git diff ec55727 1d4e48f`. The paste’s “fixes” are smaller than the net merge: several `except:` → `except Exception:` hunks are real vs HEAD, but they ride along with PEP 604 / `collections.abc` rollback and a pickle-for-skops swap. `date_model.pickle` / `model.pickle` are **added** (`A`) and HEAD `date_model.skops` / `model.skops` are **deleted** (`D`).

### lexnlp/extract/common/text_beautifier.py  (batch 12, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 94
- **Summary:** Arthur, the pasted `+2/−2` is merge-base. True two-dot is `+93/−~80` of quote-style churn plus a type-hint rollback, with one real lint fix. HEAD already uses PEP 604 (`str | tuple[str, int, int]`, `list[int]`, `tuple[str, int] | None`, `str | None`) and has **no** `typing` import. The PR re-adds `from typing import List, Tuple, Optional, Union` and rewrites those to `Union[...]`, `List[int]`, `Optional[Tuple[str, int]]`. The only behaviour change vs HEAD is `except:  # pylint:disable=bare-except` → `except Exception:` in `unify_quotes_braces` and `unify_quotes_braces_coords` (E722; still swallows everything short of `BaseException`).
- **Notes:** Cherry-pick only the two `except Exception:` lines (or, better, catch the actual failures from `unify_quotes_braces_unsafe`). Keep HEAD builtins, double quotes, and the expanded `TRANSFORMED_WORDS` / `term_coords` wrapping. Do not take the `typing` barrel.

### lexnlp/extract/de/citations.py  (batch 12, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** Two-dot is `+64/−~60`, not the paste’s one-line except swap. HEAD already uses `from collections.abc import Generator`, `Generator[CitationAnnotation]`, `list[CitationAnnotation]`, `Generator[dict]`. The PR rolls those to `from typing import Dict, Generator, List` and `Generator[..., None, None]` / `List` / `Dict`, and reformats `CitationAnnotation(...)`. The useful delta vs HEAD is `except:` → `except Exception:` around `get_dates(date, "de")` (module still has `# pylint: disable=bare-except`, now a lie). No citation-pattern or dictionary-key change.
- **Notes:** Take the `except Exception:` (or `(IndexError, KeyError, TypeError)` — `list(...)[0]["value"]` is what actually fails). Restore `collections.abc` / builtin generics. Keep HEAD `"de"` double quotes. The paste is not the net merge.

### lexnlp/extract/de/court_citations.py  (batch 12, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** The paste’s unused `from lexnlp.extract.all_locales.languages import Locale` delete is already gone on HEAD. Two-dot `+110/−~100` does two real things: (1) good — public helpers stop sharing module-global `parser = CourtCitationsParser()` (`self.items` / `self.language` are mutated per call) and instead do `CourtCitationsParser().parse(...)`; (2) bad — `collections.abc.Generator` / `tuple[int, int]` / `list[CourtCitationAnnotation]` / `list[tuple[str, int]]` become `typing.List, Tuple, Generator` plus `Generator[..., None, None]`, and the `split_text_by_keywords` docstring is stripped. Registry strings and `get_dates` usage are unchanged.
- **Notes:** Keep per-call `CourtCitationsParser()` (or a thread-local) on top of HEAD types. Restore the docstring and `"de"` quotes. Do not treat the paste as HEAD still importing `Locale`.

### lexnlp/extract/de/date_model.pickle  (batch 12, part 4/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Security regression. Two-dot is `A` 54 794-byte zlib/joblib pickle (`78 5e…`) and `D` HEAD `lexnlp/extract/de/date_model.skops` (4 090 276 bytes). HEAD `dates.py` loads via `lexnlp.ml.model_io.load_bundled_model("./date_model.pickle")`, which prefers the `.skops` sibling with `trusted=True` against `DEFAULT_TRUSTED_ALLOWLIST`. The PR tree has no `.skops` and the paired `dates.py` switch is `load_joblib_model` (arbitrary code on load). This undoes the bundled-sklearn skops migration (`skops>=0.11`).
- **Notes:** Keep `date_model.skops`. Do not re-add the pickle. Training still writes `date_model.pickle` from `dates_de_classifier.train_default_model`; that is a trainer output path, not a reason to ship pickle. Same pattern as EN `date_model.pickle` in other batches.

### lexnlp/extract/de/dates.py  (batch 12, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 94
- **Summary:** Two-dot `+110/−36` mixes a real concurrency/API fix with the pickle-loader revert. HEAD is a 42-line shim: `MODEL_DATE = load_bundled_model(.../date_model.pickle)` then `get_dates = parser.get_dates` (and list/annotation aliases) on one shared `DeDateParser`. That parser mutates `self.text` / `self.locale` / `self.dates`, so concurrent calls race. The PR’s `_build_parser()` / `_coerce_locale()` (copy via `Locale.get_locale()`) plus fresh parser per public call is the right shape, and exposing `base_date` → `RELATIVE_BASE` and `threshold` on `get_date_annotations` matches the all-locales dispatch tests. The regression is `from lexnlp.ml.model_io import load_bundled_model` → `from lexnlp.utils.unpickler import load_joblib_model`, `text: str = None` instead of `str | None`, and `typing.Generator/List/Optional` instead of HEAD’s builtins (HEAD file currently has almost no annotations because it only aliases).
- **Notes:** Reimplement `_build_parser` / `_coerce_locale` / per-call wrappers **on** `load_bundled_model`. Keep a module-level `parser = _build_parser()` because `tests/test_dates.py` still calls `parser.passed_general_check`. Default locale remains `Locale("de-DE")`. Do not take `load_joblib_model`.

### lexnlp/extract/de/de_date_parser.py  (batch 12, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** Two-dot `+151/−~140`. The load-bearing fix: HEAD still does `re.sub(CUSTOM_DATES_SEPARATOR, "\n", self.text)` then `split("\n")`, which **shortens** the string (`" und "` → `"\n"`) so coordinates after a coordinated “und” date are wrong relative to the caller’s document. The PR walks `CUSTOM_DATES_SEPARATOR.finditer` and offsets `text_part_start + match.start()`, then yields `source_text[location_start:location_end]`. That is why `test_point_inside_with_two_dates` can assert `['15. Februar 1972', '29. Dezember 1972']`. Also good vs HEAD: `except:` → `except Exception:` on `w2n.convert`, and `except Exception as e: print(str(e))` → `except (TypeError, ValueError): self.dates = []`. Regressions: `collections.abc.Generator` / `str | None` / `list[DatePart]` / `list[int]` → `typing.List/Optional/Generator[..., None, None]`; `text: str = None`; deletion of the `get_word_parts` / `get_date_annotations` docstrings; `if source_text is None: raise RuntimeError` undoes HEAD’s empty-input guard (`if not self.text: return`).
- **Notes:** Port the span-based split, classifier coords, and narrower excepts onto HEAD types/docstrings. Keep skipping empty fragments. Prefer `str | None = None` over `text: str = None`. Copying `self.locale = Locale(locale.get_locale())` is better than mutating `self.locale.language` in place — keep that.

### lexnlp/extract/de/definitions.py  (batch 12, part 7/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 92
- **Summary:** Paste deletes unused `Optional` — HEAD already has `from collections.abc import Generator` and no `Optional`. Two-dot `+64/−62` is quote/wrap churn plus typing rollback: `list[PatternFound]` / `list[DefinitionAnnotation]` / `Generator[DefinitionAnnotation]` become `List[...]` / `Generator[..., None, None]`, and `yield from dfs` becomes a no-op `for d in dfs: yield d`. Regexes, `make_de_definitions_parser`, and `UniversalDefinitionsParser` wiring are identical.
- **Notes:** Keep HEAD. Do not “fix” an unused import that is already gone.

### lexnlp/extract/de/model.pickle  (batch 12, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 97
- **Summary:** Same security regression as `date_model.pickle`. Two-dot `A` 29 064-byte joblib pickle and `D` HEAD `lexnlp/extract/de/model.skops` (3 917 590 bytes). Runtime `dates.py` loads `date_model.*`, not this file; it still sits in `scripts/reexport_bundled_sklearn_models.py` (`Path("lexnlp/extract/de/model.pickle")`) as a bundled sklearn artifact HEAD already re-exported to skops. Shipping pickle again is the insecure format, not a no-op.
- **Notes:** Keep `model.skops`. Nobody on either tree `load`s `./model.pickle` at import; that is not a reason to restore pickle. Reject together with the `load_joblib_model` caller in `dates.py`.

### lexnlp/extract/de/percents.py  (batch 12, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 95
- **Summary:** Two-dot `+55/−55`. Buried in typing/quote rollback is a real HEAD bug: after `real_amount = PERCENT_UNITS_MAP.get(unit_name, Decimal(0)) * amount` (e.g. `50 * Decimal(0.01)` → `0.5`), HEAD does `real_amount = round(amount, float_digits)` and stores the **unscaled** 50 in `PercentAnnotation.fraction`. The PR correctly does `round(real_amount, float_digits)`. Everything else is a downgrade: `collections.abc.Generator` / `Generator[PercentAnnotation]` / `list[dict]` → `typing.Generator/List` + 3-arg Generator; HEAD `PERCENT_PTN = rf"""...{amounts_parser.NUM_PTN}..."""` → `r"""...{num_ptn}...""".format(num_ptn=amounts_parser.NUM_PTN)`.
- **Notes:** Cherry-pick only `round(real_amount, float_digits)`. Keep the f-string pattern and HEAD types/`"prozent"` quotes. Verify against DE percent tests after the one-line fix — `fraction` values will change from `amount` to `amount * 0.01`.

### lexnlp/extract/de/tests/test_dates.py  (batch 12, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 91
- **Summary:** Two-dot `+137/−~120` plus mode `100644` → `100755` (HEAD test is not executable; PR makes it so). The one assertion to keep is in `test_point_inside_with_two_dates`: `text[slice(*annotation.coords)]` equals `['15. Februar 1972', '29. Dezember 1972']` — that pins the `de_date_parser` coordinate fix. The rest drops HEAD docstrings (`test_dates`, `test_date_reverse_order`, `test_negative_jahr`, `test_point_inside`, `test_file_samples`, `debug_test_train_classifier`), rolls `list[DateAnnotation]` → `List[DateAnnotation]`, and churns quotes. Existing span/value/`source` checks are otherwise the same five-date statute fixture.
- **Notes:** Add the coords-substring assert to HEAD; restore `100644` and the docstrings. Do not take `from typing import List`. Pairs with `de_date_parser.py` — do not merge this test without the span-based split or it will fail on HEAD.

## Batch 14 (EN constraints + contract-type loaders/tests + cusip, 10 file-chunks) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs are three-dot `51c07bc...pr30`. Judgments below use two-dot `git diff ec55727 1d4e48f`. Several pastes are a fraction of the net merge: `constraints.py` is `+57/−83` not `+42/−36`; `runtime_model.py` is `+177/−68` not `+159/−36`; `test_runtime_model.py` is `+229/−224` not `+244/−10`; `cusip.py` is `+28/−29` not `+1`. HEAD already has `lexnlp.ml.model_io` (`dump_model` / `load_model` / `load_bundled_model`, `skops>=0.11`, `pandas>=2.2.0,<3`) and `write_pipeline_to_catalog` → `(Path, bool)` writing `pipeline_contract_type_classifier.skops`.

### lexnlp/extract/en/constraints.py  (batch 14, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** The ReDoS rewrite is real and worth taking. HEAD still uses a `.*?`-led `CONSTRAINT_PATTERN_TEMPLATE` plus `regex` `capturesdict()`, which retries from every character on trigger-free sentences. The PR splits that into delimiter-prefixed `RE_CONSTRAINT` plus a leftover `RE_CONSTRAINT_WITH_POST` search and a cursor so spans/pre-text stay contiguous; PR `test_condition_constraint_security.py` pins the 10k-char bound and the `"greater than" / "within" / "after"` span triple. Net two-dot also rolls HEAD `collections.abc.Generator` / `tuple[str | None, ...]` / `list[...]` / `Generator[ConstraintAnnotation]` back to `typing.Generator, List, Optional, Tuple` with 3-arg `Generator[..., None, None]`, replaces the f-string combine with `"{0} {1}".format`, and **stops honouring `strict`**: HEAD’s `if strict and (num_pre + num_post == 0): continue` is gone and `strict` is now an unused parameter. First-branch matches always set `post=""`.
- **Notes:** Cherry-pick the two compiled patterns, the cursor loop, and the sentence-start post-text branch onto HEAD types/quotes/f-strings. Restore `strict` (skip when both `pre` and `post` are empty). Do not take the `typing` barrel. Pair with `lexnlp/extract/en/tests/test_condition_constraint_security.py` (not in this batch).

### lexnlp/extract/en/contracts/contract_type_detector.py  (batch 14, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** HEAD `detect_contract_type` still does `type_vector[0]` / `type_vector[1]` on a `Series` whose index is contract-type **labels**. Under HEAD `pandas>=2.2.0,<3` that is deprecated positional fallback; under the PR’s `pandas>=2.2.0,<4` it is label lookup and breaks `test_series_positioning.py`. The `.iloc[0]` / `.iloc[1]` swap is the fix to keep. This file never went through skops: HEAD still `joblib.load`s the caller-supplied RF path. `load_joblib_model` is pickle-family ABI shims (`lexnlp.utils.unpickler`), not a security upgrade — the PR unpickler module itself says it “does not make unpickling safe.” Two-dot also re-adds `from typing import List`, `list[str]` → `List[str]`, and quote churn.
- **Notes:** Port only the `iloc` reads (and the docstring `[0]` → `.iloc[0]` wording) onto HEAD. Prefer routing the RF file through `lexnlp.ml.model_io.load_model` / `load_bundled_model` if we ever ship a `.skops` sibling; do not treat `load_joblib_model` as a skops replacement. Keep `list[str]` and double quotes.

### lexnlp/extract/en/contracts/predictors.py  (batch 14, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 94
- **Summary:** Same pandas positional bug in `infer_classification` (`predictions[0]` / `predictions[1]` vs string labels). Keep `.iloc`. Default-pipeline tags are unchanged (`pipeline/is-contract/0.2`, `LEXNLP_IS_CONTRACT_MODEL_TAG`, `pipeline/is-contract/0.1`, `pipeline/contract-type/0.1`, `pipeline/contract-type/0.2-runtime`); HEAD’s base class still loads those via `load_model(..., trusted=True)`. The legacy is-contract fallback `cloudpickle.load` → `load_sklearn_model` is still arbitrary-code pickle, just with sklearn rename/tree shims — prefer `lexnlp.ml.model_io.load_model(legacy_path)` so a `.skops` sibling is honoured. Error-text pointing at `download_github_release('pipeline/is-contract/0.1', prompt_user=False)` and `ensure_runtime_contract_type_model(force=True)` is a real UX improvement. Regressions: `collections.abc.Iterable` / `bool | tuple[bool, float]` / `str | Iterable[str]` → `typing.Iterable, Tuple, Union`.
- **Notes:** Cherry-pick `iloc` plus the fallback `RuntimeError` messages onto HEAD types. Do not drop `from collections.abc import Iterable`. Do not replace `cloudpickle.load` with raw pickle if `load_model` can take the path.

### lexnlp/extract/en/contracts/runtime_model.py  (batch 14, part 4/10)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 96
- **Summary:** Security + API rollback wrapped around genuinely better catalog/trust behaviour. HEAD docstring is “Python 3.13+ compatible”; PR says “Python 3.11-compatible”. HEAD `CONTRACT_TYPE_MODEL_FILENAME = "pipeline_contract_type_classifier.skops"` plus `LEGACY_CONTRACT_TYPE_MODEL_FILENAME = "...cloudpickle"`; PR deletes the legacy name and writes `.cloudpickle` via `pickle.dump`. HEAD `load_pipeline_for_tag` is `load_model(path, trusted=True)`; PR opens the file and `load_sklearn_model` (pickle). `write_pipeline_to_catalog` return type `tuple[Path, bool]` → `Path`, which breaks HEAD `scripts/train_contract_type_model.py` (`destination, wrote = write_pipeline_to_catalog(...)`) and `tests/test_runtime_model_sklearn18.py`. Typing rollback `tuple[list[str], list[str], dict[str, int]]` → `Tuple[List[str], ...]`. Worth porting onto **skops**: `atomic_output_path`, `threadpool_limits(CONTRACT_TYPE_TRAINING_THREADS=1)`, fail-closed `AssetTrustError` / `verify_trusted_asset_file` (do not train after a checksum mismatch), `get_local_candidate_tag` so a trusted release tag is never overwritten by a locally trained candidate, `max_docs_per_label=0` meaning “no cap”, and `max_features` as a parameter (HEAD hardcodes `120000`; PR default `75_000` — keep 120000 unless the quality gate is re-baselined).
- **Notes:** HEAD catalog has `get_path_from_catalog` / `invalidate_catalog_cache` only; `get_exact_path_from_catalog`, `get_local_candidate_tag`, `get_catalog_directory`, `download_github_release_to_path`, and `lexnlp.ml.artifact_io` are PR-only and must be cherry-picked first. Keep member-name sort **or** tar order, not both — HEAD sorts `archive.getmembers()` for determinism; PR iterates the tarball as stored because the corpus release is checksum-pinned. Do not reintroduce `multi_class="multinomial"` (HEAD already dropped it for sklearn 1.8). Do not take `pickle.dump`.

### lexnlp/extract/en/contracts/tests/test_contract_type_quality_gate.py  (batch 14, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** New 137-line file; absent on HEAD. It pins `verify_fixture_sha256` tamper rejection and `evaluate_duplicate_group_holdout` (accept unchanged candidate, reject `accuracy_top1` 0.75→0.74 at `max_regression=0.0`, fail-closed on missing `validation_metrics` / `split_sha256`, reject split/recipe mismatch). Those functions do **not** exist on HEAD `scripts/contract_type_quality_gate.py` (HEAD still has `parse_metrics` / `load_baseline_metrics` / `REQUIRED_METRIC_KEYS` including `accuracy_topn`). The tests are the right shape for the PR quality-gate rewrite; they cannot land alone.
- **Notes:** Take them together with the script (not in this batch). Prefer `scripts/tests/` next to HEAD `scripts/tests/test_train_contract_type_model.py` rather than under `lexnlp/extract/en/contracts/tests`. `accuracy_top3` in the fixture is the PR alias for HEAD `accuracy_topn` — keep HEAD’s canonical key name if the script still aliases.

### lexnlp/extract/en/contracts/tests/test_contract_type_training.py  (batch 14, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** New 91-line file; absent on HEAD. Locks `build_duplicate_group_holdout` as deterministic, leak-free (`train` ∩ `test` empty), excludes the two cross-label `"cross-label"` / `"CROSS-LABEL"` samples (indices 11 and 12), and freezes `split_sha256=520b6df4cb604777c0b60bd46422bb76ef3cadcc2d0039622a7a2acec7ec6c92` plus `split_assignment_sha256` sensitivity to group→label maps. HEAD `scripts/train_contract_type_model.py` has only `parse_args` / `score` / `main` — no holdout helpers — so this file fails on current master until that script is cherry-picked.
- **Notes:** Same placement note as the quality-gate tests (`scripts/tests/`). Keep the exact SHA and the `validation_size=0.4` group-count assert; they are the recipe fingerprint, not a dependency downgrade.

### lexnlp/extract/en/contracts/tests/test_model_tag_overrides.py  (batch 14, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 94
- **Summary:** Two-dot `+57/−39`. The PR **deletes** HEAD `test_probability_predictor_uses_model_io_loader_for_default_pipeline`, which asserts `lexnlp.ml.predictor.load_model(path, trusted=True)` against a `.skops` placeholder and that the catalog tag equals `_DEFAULT_PIPELINE`. That test is the skops load path; dropping it is a regression. Additive and good: `test_is_contract_missing_models_error_has_installed_api_guidance` / `test_contract_type_missing_models_error_has_installed_api_guidance` (messages must mention `download_github_release` + `pipeline/is-contract/0.1` and `ensure_runtime_contract_type_model(force=True)`). Fallback monkeypatch `cloudpickle.load` → `unpickler.load_sklearn_model` matches the predictors.py pickle fallback and should instead patch `lexnlp.ml.model_io.load_model` if we keep HEAD’s loader.
- **Notes:** Restore the `load_model(..., trusted=True)` test verbatim. Add the two guidance tests. Do not switch the fallback mock to pickle-only `load_sklearn_model`.

### lexnlp/extract/en/contracts/tests/test_runtime_model.py  (batch 14, part 8/10)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 95
- **Summary:** Near-total rewrite (`+229/−224`). HEAD tests that **must stay**: `write_pipeline_to_catalog` returns `(Path, True/False)`, does not overwrite when `force=False`, creates parents, skips when the **legacy** `pipeline_contract_type_classifier.cloudpickle` exists, force-overwrites to **`.skops`**, `CONTRACT_TYPE_MODEL_FILENAME.endswith(".skops")`, `LEGACY_CONTRACT_TYPE_MODEL_FILENAME.endswith(".cloudpickle")`, and `force=True` does not consult/download the target tag. The PR deletes all of that and retargets writes at `.cloudpickle` + `load_sklearn_model`. Worth grafting onto the skops tests: per-label cap `0` uses every tar member in stored order, cap `2` keeps two `A` samples, atomic repair of a truncated artifact (no leftover `.*.tmp`), `get_path_from_catalog` aliases a missing release tag to `get_local_candidate_tag` without creating the release directory, `write_pipeline_to_catalog` of non-manifest bytes onto a pinned tag raises `ChecksumError` and leaves no file, and `ensure_runtime_contract_type_model` must **not** train after `ChecksumError`.
- **Notes:** `test_ensure_runtime_contract_type_model_force_trains` on the PR writes to `get_local_candidate_tag(RUNTIME_CONTRACT_TYPE_TAG)` and expects `max_features=75_000`; HEAD writes the release tag and has no `max_features` kwarg. Keep HEAD’s `(destination, True)` mock return. Pair with `test_runtime_model_sklearn18.py`, which the PR does not touch in this batch but will break if `write_pipeline_to_catalog` stops returning a tuple.

### lexnlp/extract/en/contracts/tests/test_series_positioning.py  (batch 14, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 96
- **Summary:** New 27-line file; absent on HEAD. Builds a `Series([0.9, 0.1], index=["EMPLOYMENT AGREEMENT", "SERVICES AGREEMENT"])` and asserts both `ProbabilityPredictorContractType.infer_classification` and `ContractTypeDetector.detect_contract_type` return `"EMPLOYMENT AGREEMENT"`. That is exactly the pandas 2.2+/3 label-vs-position bug on HEAD (`predictions[0]` looks up label `0`, not row 0). No pickle/skops/typing interaction.
- **Notes:** Land together with the `.iloc` edits in `predictors.py` and `contract_type_detector.py` or this file fails on current master. Do not “fix” the test by switching the index to integers.

### lexnlp/extract/en/cusip.py  (batch 14, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 93
- **Summary:** Paste is the one-line `text=code` kwarg. Two-dot is `+28/−29`: that kwarg plus a full typing/quote rollback. HEAD already types `CusipAnnotation.__init__(..., text: str | None = None)` and `to_dictionary_legacy()["text"]` is `self.code`; the extractor just never passed `text`, so `TextAnnotation.text` stayed `""`. Passing `text=code` is the right fill-in. Everything else is a downgrade: `collections.abc.Generator` / `dict[str, Any]` / `list[dict[str, Any]]` / `Generator[CusipAnnotation]` → `typing.Any, Dict, Generator, List` with 3-arg Generator, plus `'[\@\#\*]'` / `'123456789ABC'` quote churn and `10+n` spacing.
- **Notes:** Cherry-pick only `text=code` onto HEAD’s `CusipAnnotation(...)` call. Keep builtin generics and the `ppn: str | bool | None` / `checksum: str | int | None` annotation types (those live in `cusip_annotation.py`, not this file). `checksum=0` handling is already correct on HEAD’s annotation class.
