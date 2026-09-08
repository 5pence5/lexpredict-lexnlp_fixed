## Batch 1 (CI/config/docs, 10 files) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs in this batch are merge-base (51c07bc, PYTHON 3.11 era) vs pr30, not HEAD (ec55727, PYTHON 3.13, uv_build) vs pr30. Verified via `git show HEAD:` / `git show pr30:` and `git merge-base HEAD pr30`. Judgments below compare against HEAD (the GOOD state).

### .github/dependabot.yml (batch 1, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Adds a new 9-line Dependabot config scheduling weekly `github-actions` updates with `dependencies` + `github-actions` labels. HEAD has no dependabot.yml, so this is purely additive with no downgrade or deleted modernisation.
- **Notes:** No version conflicts; complements the SHA-pinned actions used elsewhere in the PR.

### .github/workflows/asset-drift.yml (batch 1, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Good parts: pins `actions/checkout@v4` to `checkout@de0fac2e... # v6.0.2`, `setup-python@v5` to `v6.2.0`, `setup-uv@v6` to `v9.0.0` with `version: "0.12.0"`, adds an ephemeral latest-runtime smoke build, and moves contract candidate `pipeline/is-contract/0.1` to `0.2` (matches HEAD publish env MODEL_TAG 0.2). Bad part: PR env is `PYTHON_VERSION: "3.11"` vs HEAD `"3.13"` (verified via git show), a Python downgrade hidden because the pasted hunk omits the env block.
- **Notes:** Keep SHA pins, `prune-cache: true`, smoke step, and `0.2` candidate; restore `PYTHON_VERSION: "3.13"` before merge.

### .github/workflows/ci.yml (batch 1, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Large expansion (HEAD 335 lines vs PR 946 lines) adding genuinely useful gates: multi-Python matrix, SHA-pinned actions, `segmentation-quality` benchmarks, docs `-W`, cross-platform smoke, `minimum-dependencies` lowest-direct, `supply-chain` pip-audit/SBOM, reproducible double-build with SOURCE_DATE_EPOCH, and aggregate status. But the Python policy regresses: pasted env `3.11` to `3.12` plus matrix `["3.10", "3.11", "3.12", "3.13"]` contradicts HEAD `requires-python >=3.13,<3.15` (uv_build) and HEAD CI `PYTHON_VERSION: "3.13"`.
- **Notes:** Keep new jobs; rebase matrix/default onto 3.13/3.14 and `uv_build`. Base suite widening `pytest lexnlp` to `pytest lexnlp scripts/tests` plus `ruff check --select E9,F63,F7,F82` and `--check-current` are fine.

### .github/workflows/publish-contract-model.yml (batch 1, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Restructures single `contents: write` publish job into `build` (`contents: read`) + `publish` (`contents: write`, `environment: model-release`, concurrency group), adds dual-stack verification (minimum ABI via `constraints/model-artifact-abi.txt` plus latest locked stack), trusted-manifest SHA256/size checks, immutable release (refuses to replace differing bytes instead of `--clobber`), and quality gates on both stacks. This hardening is an improvement. The regression is `PYTHON_VERSION: "3.11"` (merge-base) to `"3.12.13"` vs HEAD `"3.13"`.
- **Notes:** Keep least-privilege split, `persist-credentials: false`, `verify_trusted_asset_file`, no-clobber logic; bump `PYTHON_VERSION` back to `"3.13"`.

### .github/workflows/publish-contract-type-runtime-model.yml (batch 1, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Same hardening pattern as the is-contract publisher (read/write split, `model-release` environment, trusted manifest, immutable upload) plus contract-type specifics: `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, deterministic-rebuild byte comparison, and extended holdout gate flags (`--max-holdout-*-regression 0.0`). Valuable and not a revert. Same Python downgrade: HEAD `"3.13"` vs PR `"3.12.13"`.
- **Notes:** Keep determinism check and dual `minimum.json`/`latest.json` gates; restore `PYTHON_VERSION: "3.13"`.

### .gitignore (batch 1, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** One-character change removing the trailing slash: `libs/stanford_nlp/` to `libs/stanford_nlp`. Both forms ignore the Stanford directory; the slash-less form additionally matches a same-named file. No dependency, API, or modernisation revert.
- **Notes:** Harmless churn; optionally restore the trailing slash for directory-only intent, but not a merge blocker.

### .readthedocs.yaml (batch 1, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** Adds modern Read the Docs v2 config (`.readthedocs.yaml` with dot): `ubuntu-24.04`, Python `3.12`, `sphinx` `fail_on_warning: true`, and `uv sync --extra docs` install. HEAD only has legacy `readthedocs.yml` (Python `3.8`, `python-requirements.txt`, `image: latest`), so this is a modernisation, not a downgrade.
- **Notes:** Confirm the stale `readthedocs.yml` is removed in the same PR (PR tree shows only `.readthedocs.yaml`); keeping both would be confusing.

### .travis.yml (batch 1, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Deletes the 40-line dead Travis config (Python `3.4`, `oracle-java8-set-default`, `pip install -r python-requirements*.txt`, old NLTK downloaders, `coveralls`, embedded Slack token). Travis is long dead and HEAD AGENTS.md already calls it a historical reference; removal is cleanup plus secret hygiene.
- **Notes:** Quotes `language: python`, `python: - '3.4'`, and `secure: ZRKZ...` token being removed.

### AGENTS.md (batch 1, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Rewrite adds useful policy (model ABI pins joblib 1.5.0 / NumPy 1.26.4 / pandas 2.2.0 / scikit-learn 1.7.2 / SciPy 1.13.0, strict `0.0` non-regression gates, Tika 3.3.2 loopback, NLTK SHA catalog) but regresses the Python/packaging baseline vs HEAD: PR states `>=3.10,<3.14`, default `3.12`, and omits `uv_build`, while HEAD is `>=3.13,<3.15`, default `3.13`, `uv_build` backend. PR tree also reintroduces `setup.py`. Pasted `-` lines (`setuptools`, `>=3.10,<3.13`, default `3.11`) are the stale merge-base, not HEAD.
- **Notes:** Keep the validation/quality sections; restore `uv_build`, `>=3.13,<3.15`, default `3.13`, and drop the reintroduced `setup.py` direction.

### DEPENDENCY_MODERNIZATION_PLAN.md (batch 1, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 70
- **Summary:** Deletes the 59-line completed planning doc (zero-skip policy, uv consolidation, staged model-gate rollout). Its enforceable content now lives in AGENTS.md (`skip-audit`) and CI (`ci/skip_audit.py`, quality gates), so deletion is archival cleanup with no code, dependency, or API downgrade.
- **Notes:** Low confidence only because intent to archive vs retain history is a team preference; no technical regression either way.


## Batch 3 (README/docs/ci, 10 files) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs in this batch are merge-base (51c07bc, Python 3.11 era) vs pr30, not HEAD (ec55727, `requires-python >=3.13,<3.15`, `uv_build`, `.skops`, `batch`/`ner` extras) vs pr30. Verified via `git show HEAD:` / `git show pr30:` and `git merge-base HEAD pr30`. Judgments below compare against HEAD (the GOOD state).

### README.md (batch 3, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Rewrites the 197-line HEAD README (Travis badge + `>=3.13,<3.15` + `uv_build` + extras table + `lexnlp.extract.batch`/`ner` + `.pickle→.skops` + releases history) into a 108-line CI-badge doc with a cleaner asset taxonomy (bundled models vs NLTK vs pipeline classifiers vs Stanford/Tika) and `ruff`/`skip_audit`/`--check-current` validation. The structure is an improvement, but vs HEAD it downgrades Python (`3.10, 3.11, 3.12, or 3.13` + `uv python install 3.12` vs HEAD `3.13` minimum / `>=3.13,<3.15`) and deletes all HEAD modernisation docs (extras pins, `uv_build>=0.9,<0.10`, `asyncio.TaskGroup` batch, hybrid NER, `.skops` migration, dual-licensing detail, releases list).
- **Notes:** Keep PR's 4-way resource distinction and `bootstrap_assets.py --contract-model --contract-type-model` docs; restore `>=3.13,<3.15` + default `3.13`, `uv_build` backend section, extras table (`pyarrow>=17`, `huggingface_hub>=0.25`, `spacy>=3.7`, `tika>=2.6.0`), `batch`/`ner`/`.skops` sections, and releases history (or link to `changes.rst`) before merge.

### README.rst (batch 3, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** RST mirror of the README.md rewrite: HEAD 224 lines (`Python 3.13 (minimum; >=3.13,<3.15)`, `uv pip install -e ".[dev,test]"`, extras table, `lexnlp.extract.ner`, Travis/coveralls badges) becomes PR 125 lines (`Python 3.10, 3.11, 3.12, or 3.13`, `uv sync --frozen --python 3.12`, `|CI| |Documentation|` badges, same 4-way asset taxonomy). Same trade as the `.md`: better install/validation narrative, but Python downgrade plus deletion of HEAD's `uv_build`/extras/`batch`/`ner`/`.skops`/releases content.
- **Notes:** Same edits as README.md: restore `>=3.13,<3.15` + `3.13` default, `uv_build`, extras pins, `ner`/`batch`/`.skops` docs, and full release history; keep PR's `bootstrap --nltk` / `--contract-model --contract-type-model` / `--stanford` / `--tika` + `LEXNLP_USE_TIKA=true scripts/run_tika.sh` and `ruff check --select E4,E7,E9,F` + `skip_audit.py` validation blocks.

### ci/check_dist_contents.py (batch 3, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** Expands HEAD's 92-line banned-files checker (which already bans `Pipfile`/`Pipfile.lock`/`python-requirements*.txt` and uses `collections.abc.Iterable` + `list[str]`) into a 374-line wheel/sdist/installed parity gate. It preserves all HEAD bans, adds `documentation/docs/build/` to `BANNED_SUBSTRINGS`, adds `BANNED_WHEEL_PREFIXES` (`ci/`, `documentation/`, `libs/`, `notebooks/`, `scripts/`), duplicate-member/resource detection, `RUNTIME_RESOURCE_SUFFIXES` (`.csv`, `.json`, `.pickle`, `.pickle.gzip`, `.txt`, `.xml`), `MIN_NONEMPTY_LINE_COUNTS`, and `--source-root`/`--installed` modes. No dependency, async, or type-hint revert.
- **Notes:** `chmod 644→755` is harmless. Follow-up (not a blocker): `RUNTIME_RESOURCE_SUFFIXES` omits HEAD's `.skops` siblings from the `.pickle→.skops` migration, so `.skops` files are parity-invisible; add `".skops"` to the suffix tuple in a follow-up. `REQUIRED_SEGMENTATION_SDIST_MEMBERS` (7 `sota_segmentation`/`segmentation_*` paths) is PR-feature-specific and correctly scoped to the sdist check.

### constraints/model-artifact-abi.txt (batch 3, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 75
- **Summary:** Adds the 10-line pinned producer ABI (`joblib==1.5.0`, `numpy==1.26.4`, `pandas==2.2.0`, `scikit-learn==1.7.2`, `scipy==1.13.0`, `threadpoolctl==3.6.0`) with a header stating artifacts are serialized with the oldest supported ABI and must also pass quality gates on the latest locked stack. Absent in both HEAD and merge-base, so purely additive; the older `numpy==1.26.4` vs HEAD runtime `numpy>=2.3,<3` is an intentional oldest-ABI producer pin, not a runtime dependency downgrade.
- **Notes:** Comment cites `Python 3.12.13` enforced by `lexnlp.ml.artifact_abi` and publish workflows vs HEAD `requires-python >=3.13,<3.15`; treat as oldest-supported producer runtime (dual-stack gate covers latest), consistent with the `publish-contract-model` hardening in batch 1. No edit required unless the team standardises the producer on `3.13`.

### documentation/docs/requirements.txt (batch 3, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 70
- **Summary:** Deletes the 5-line legacy docs requirements (`sphinx`, `sphinx_rtd_theme`, `sphinx-markdown-tables`, `recommonmark`, `pyyaml`, no trailing newline). Deletion is part of the documented retirement of split-requirements/legacy RTD: PR replaces it with `.readthedocs.yaml` (v2, `ubuntu-24.04`, Python `3.12`, `fail_on_warning: true`, `uv sync --extra docs`) plus a `docs` extra (`sphinx>=7.4,<10`, `sphinx-rtd-theme>=3,<4`). No code, dependency, or API revert.
- **Notes:** Merge only together with `.readthedocs.yaml` + `docs` extra in the same PR; verify a clean docs build passes without `sphinx-markdown-tables`/`recommonmark`/`pyyaml` (PR `conf.py` drops the markdown-dependent paths, but `fail_on_warning: true` will catch any leftover). Low confidence reflects that docs-build proof, not the deletion direction.

### documentation/docs/source/about.rst (batch 3, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 88
- **Summary:** Rewrites HEAD's 43-line stale `about.rst` (old feature bullets, `ContraxSuite Projects` link farm including deployment automation, `drafting a whitepaper` citation note) into 36 lines describing maintained functionality (sentence/paragraph/page/section/title segmentation; amounts/money/dates/courts/citations extraction; EN/DE/ES locales; models; clustering utilities) plus `Related projects` and a `support@contraxsuite.com` citation contact. Pure prose modernisation with no version, dependency, or API change.
- **Notes:** Dropped deployment-automation link and whitepaper-drafting sentence are intentional cleanup, not regressions. Mode change `100755→100644` is correct hygiene.

### documentation/docs/source/changes.rst (batch 3, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 76
- **Summary:** Keeps the full 2.3.0→0.1.0 history (only fixing RST underline lengths) and adds a 9-bullet `Unreleased` section documenting the PR: `uv` lock/matrix, constrained producer ABI + quality gates, verified bootstrapping + `Apache Tika 3.3.2`, retirement of Pipenv/Travis/legacy RTD, experimental `2.4.0a1` hierarchy, SaT adapter, strict-budget chunking, versioned embedding-payload serialiser, and hermetic gates with an explicit `not held-out SOTA evidence` disclaimer. The changelog itself deletes nothing, but its first bullet enshrines the regressed matrix (`Python 3.10–3.13`) vs HEAD (`>=3.13,<3.15`).
- **Notes:** Keep the `Unreleased` structure and SOTA disclaimer; edit bullet 1 to `Python 3.13–3.14` (HEAD baseline) before merge and confirm the `Tika 3.3.2` + `0.1`→`0.2` model-asset claims match the bootstrap/manifest state at merge time.

### documentation/docs/source/conf.py (batch 3, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 86
- **Summary:** Modernises HEAD's 176-line legacy Sphinx config (hardcoded `version = release = '2.3.0'`, `copyright '2015-2021'`, `os.path` + `../../` sys.path, `language = None`, `master_doc`, `doctest`/`coverage`/`intersphinx` extensions) into 81 lines: dynamic version via `importlib.metadata.version("lexnlp")` with `lexnlp.__version__` fallback, `copyright '2015–2026, ContraxSuite, LLC and LexPredict, LLC'`, `pathlib` `REPOSITORY_ROOT = parents[3]`, `root_doc = "index"`, `language = "en"`, `autodoc`/`autosummary`/`githubpages` extensions, `autodoc_member_order = "bysource"`, and `exclude_patterns = ["api/**", "build/**"]`. Adds `from __future__ import annotations`; no dependency or API downgrade.
- **Notes:** Dropping `doctest`/`coverage`/`intersphinx` is fine (unused here); `autosummary` + `api/**` exclusion (obsolete apidoc snapshot; curated reference under `modules/api/`) is an improvement. `parents[3]` from `source/conf.py` correctly resolves to the repo root where the old `abspath("../../")` resolved to `documentation/`. Mode `100755→100644` is correct.

### documentation/docs/source/development/segmentation_v2_design.rst (batch 3, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** Adds a new 406-line design record for the experimental `2.4.0a1` structure-first segmentation Retriever-chunking stack (exact half-open spans, `tree_sha256`/`manifest_id`/`chunk_id`/`payload_metadata_sha256` identities, `MONOTONIC`/`ARBITRARY` token policies with `TokenSearchLimitExceeded`, `PRESERVE` vs `PACK_SIBLINGS`, hermetic quality/benchmark gates, statute scaling lane, LegalBench-RAG promotion methodology). Absent in HEAD and merge-base, so purely additive with no deleted modernisation. SOTA claims are correctly hedged (`not held-out SOTA evidence`, `intended to support that comparison, not a demonstrated SOTA result`).
- **Notes:** The `NUPunkt` paragraph notes `Python 3.11+` vs PR's `3.10` support matrix — informative, not a code revert; re-check that sentence against the final `requires-python` after the Python-baseline edit. Citations (`Frohmann2024`, `Bommarito2025`, `Shaukat2026`, `Prior2026`, Docling, LegalBench-RAG) are references, not dependency pins.

### documentation/docs/source/guides/lossless_segmentation_api.rst (batch 3, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 86
- **Summary:** Adds a new 356-line user guide for the PR's additive `2.4.0a1` API (`segment_document`, `StructureProfile.CONSERVATIVE`/`STATUTE`, `StructuralMode.REPLACE`/`AUGMENT`, `SaTSentenceSegmenter`, `chunk_document`/`iter_chunks`, `TokenCounterPolicy`, `ContainerPolicy`, `render_embedding_payload`, `PayloadBudgetExceeded`) with explicit non-goals (legacy APIs unchanged; no discontiguous/PDF-geometry/table-cell/OCR/footnote model; SaT/NUPunkt/CharBoundary/Docling are challengers, not defaults). Purely additive docs for new code; no existing API, type-hint, async, or dependency change.
- **Notes:** Guide correctly states `DEFAULT_MAX_CHARS == 4000` is a guardrail not an optimal retrieval size, token mode never infers capability, and `payload.text` must be sent unchanged. No version-pin or modernisation concern.

### documentation/docs/source/modules/api/utilities.rst  (batch 5, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Adds a new 60-line Sphinx reference page documenting existing utility modules (amount_delimiting, decorators, iterating_helpers, line_processor, phrase_finder, map, parse_df, unpickler, unicode_lookup). Purely additive docs; no existing file modified and no dependency, async, or type-hint change.
- **Notes:** Verified on master that all referenced modules/symbols exist (DelimitedBlock, get_delimited_blocks, infer_delimiters in lexnlp/utils/amount_delimiting.py; lines_processing/line_processor.py and phrase_finder.py; utils/unicode/unicode_lookup.py).

### documentation/docs/source/modules/extract/de/amounts.rst  (batch 5, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Fixes RST heading underline lengths (12-char `============` extended to match the 52-char title; `-----------------` to `------------------`). Cosmetic Sphinx correctness fix with no code, dependency, or API effect.
- **Notes:** Old short underlines would emit Sphinx "title underline too short" warnings; the new lengths are correct.

### documentation/docs/source/modules/extract/de/citations.rst  (batch 5, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Same RST underline-length fix as the other doc pages (title underline extended to match `:mod:`lexnlp.extract.de.citations`: Extracting citations`; section underline `-----------------` to `--------------------`). Docs-only, no behavior change.
- **Notes:** None.

### documentation/docs/source/modules/extract/de/dates.rst  (batch 5, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Fixes the title underline length to match `:mod:`lexnlp.extract.de.dates`: Extracting date references`. Docs-only Sphinx warning fix; no code or dependency change.
- **Notes:** 2-line change (+2/-2), nothing else touched.

### documentation/docs/source/modules/extract/de/durations.rst  (batch 5, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** RST underline-length fix for the title and the `Extracting durations` section heading. Identical in nature to the other de/* doc fixes; no functional change.
- **Notes:** None.

### documentation/docs/source/modules/extract/de/percents.rst  (batch 5, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** RST underline-length fix for the title and the `Extracting percents` section heading. Docs-only, no code, API, or dependency effect.
- **Notes:** None.

### documentation/docs/source/modules/extract/en/acts.rst  (batch 5, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** RST underline-length fix for the title and the `Extracting acts` section heading. Docs-only Sphinx correctness fix with no behavior change.
- **Notes:** None.

### documentation/docs/source/modules/extract/en/amounts.rst  (batch 5, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Fixes title underline plus both section underlines (`Extracting amounts`, `Converting text to numbers`) to match heading lengths. Docs-only formatting fix; no code or API change.
- **Notes:** 4-line change (+4/-4), all underline characters.

### documentation/docs/source/modules/extract/en/citations.rst  (batch 5, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** RST underline-length fix for the title and the `Extracting citations` section heading. Docs-only, no functional or dependency change.
- **Notes:** None.

### documentation/docs/source/modules/extract/en/companies.rst  (batch 5, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Retargets the companies doc page from `lexnlp.extract.en.entities.nltk_re` to `lexnlp.extract.en.entities.nltk_maxent` and replaces a broken doctest-style example with a clean `get_companies` snippet. This is a fix, not a downgrade: on master `nltk_re.py` defines only pattern builders (get_company_type_pipe, create_company_pattern) while `get_companies` lives in `nltk_maxent.py`, and the old example called the nonexistent path `get_entities.nltk_re.get_companies`.
- **Notes:** Mode change 100755 to 100644 is correct normalization. The rewrite drops the old example outputs and stale test-link; acceptable since the old example was un-runnable. Both modules coexist on master so no module is removed.

### documentation/docs/source/modules/extract/en/money.rst  (batch 7, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST formatting fix. Extends the over-short `============` title over/underlines to match the `:mod:`lexnlp.extract.en.money`` title length and extends the `Extracting money and currency references` section underline (`-----------------` to 40 dashes), plus a blank line after the ISO 4217 list and trailing-newline cleanup. No code, API, dependency, or behavior change.
- **Notes:** +4/-4, all underline/whitespace characters; fixes Sphinx "title underline too short" warnings.

### documentation/docs/source/modules/extract/en/percents.rst  (batch 7, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST underline-length fix. Replaces the 12-char `============` title adornment with a full-length rule matching `:mod:`lexnlp.extract.en.percents`` and fixes the `Extracting conditions` section underline (`-----------------` to `---------------------`). No functional or dependency change.
- **Notes:** +3/-3, underline characters only.

### documentation/docs/source/modules/extract/en/pii.rst  (batch 7, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST underline-length fix. Extends the `============` title adornment to match the long `:mod:`lexnlp.extract.en.pii`` title and fixes the `Extracting PII` section underline (`-----------------` to `--------------`). No code, API, or dependency change.
- **Notes:** +3/-3, underline characters only.

### documentation/docs/source/modules/extract/en/ratios.rst  (batch 7, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST underline-length fix. Replaces the 12-char `============` title adornment with a full-length rule for `:mod:`lexnlp.extract.en.ratios`` and corrects the `Extracting conditions` section underline. No functional change.
- **Notes:** +3/-3, underline characters only.

### documentation/docs/source/modules/extract/en/regulations.rst  (batch 7, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST formatting fix. Corrects the title over/underline (`============` to full-length rule for `:mod:`lexnlp.extract.en.regulations``) and two section underlines (`Extracting constraints` and `Customizing regulation extraction`). No code, API, or dependency change.
- **Notes:** +4/-4, underline characters only.

### documentation/docs/source/modules/extract/en/trademarks.rst  (batch 7, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST underline-length fix. Extends the `============` title adornment to match `:mod:`lexnlp.extract.en.trademarks`` and fixes the `Extracting conditions` section underline. No functional or dependency change.
- **Notes:** +3/-3, underline characters only.

### documentation/docs/source/modules/extract/en/urls.rst  (batch 7, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST underline-length fix. Replaces the 12-char `============` title adornment with a full-length rule for `:mod:`lexnlp.extract.en.url`` and fixes the `Extracting constraints` section underline. No code or API change.
- **Notes:** +3/-3, underline characters only.

### documentation/docs/source/modules/extract/es/dates.rst  (batch 7, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 97
- **Summary:** Docs-only RST title underline fix. Extends the `============` over/underline to match the `:mod:`lexnlp.extract.es.dates`` title length. No content, code, or dependency change.
- **Notes:** +2/-2, underline characters only.

### documentation/docs/source/modules/extract/extract.rst  (batch 7, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 88
- **Summary:** Docs-only rewrite of the extract package index. Replaces the Locale example lists with grouped EN/DE/ES reference lists, converts the inline `::` example to a `.. code-block:: python` block, adds a hidden `.. toctree:: :glob:` for `de/* en/* es/*`, and rewrites the NLP/Stanford and classifier notes to describe bootstrapped assets and the migration runbook. Also normalizes mode 100755 to 100644. No Python code, dependency, async, or type-hint change.
- **Notes:** Net -84/+74 lines. The trimmed DE list (amounts, citations, dates, durations, percents) and single linked ES page (`extract_es_dates`) match the .rst files actually present in `documentation/docs/source/modules/extract/de/` and `es/`; the removed DE/ES `:ref:` links (e.g. `extract_de_courts`, `extract_es_courts`) pointed at pages that do not exist on master, so dropping them fixes broken refs rather than regressing. Minor loss of inline examples (e.g. "ten pounds", "32 CFR 170") is acceptable.

### documentation/docs/source/modules/nlp/en/segments_pages.rst  (batch 7, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 92
- **Summary:** Docs-only fix. Corrects the over-short `============` title adornment for `:mod:`lexnlp.nlp.en.segments.pages`` and replaces the `.. automodapi:: lexnlp.nlp.en.segments.pages` + `:include-all-objects:` directive with standard `.. automodule::` + `:members:`. No code or dependency change.
- **Notes:** `automodapi` is correct to remove here: `documentation/docs/source/conf.py` has `sphinx_automodapi.automodapi` commented out, so `automodapi` is an unknown directive while `automodule` works. Sibling pages (segments_sentences/titles/paragraphs/sections, tokens, transforms) still use `automodapi` on master; follow-up normalization there is out of scope for this file.

## Batch 9 (all_locales dispatchers + package init, 10 files) — reviewed 2026-09-08 vs HEAD ec55727

Note on base: pasted diffs in this batch are merge-base (51c07bc, Python 3.11 era) vs pr30, not HEAD (ec55727) vs pr30. Verified via `git show HEAD:` / `git show pr30:` and `git merge-base HEAD pr30` (=51c07bc). Judgments below compare against HEAD (the GOOD state).

### lexnlp/__init__.py  (batch 9, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Bumps `__version__` `2.3.0` to `2.4.0a1` (and `__license__` `.../blob/2.3.0/LICENSE` to `.../blob/2.4.0a1/LICENSE`) and adds a legacy mutable-`MODELS_REPO` fallback (`globals().get("MODELS_REPO")` vs `_IMPORTED_MODELS_REPO`, trailing-slash normalisation) plus `_IMPORTED_MODELS_REPO`. The version bump matches the PR's experimental segmentation stack, but the fallback reintroduces mutable module-state config that HEAD deliberately replaced with pure env vars (`LEXNLP_MODELS_REPO`/`LEXNLP_MODELS_REPO_SLUG`), and it reverts HEAD's single-line `DEFAULT_MODELS_REPO` to the old parenthesised form.
- **Notes:** Keep `2.4.0a1` if the release intends it and the `100755→100644` mode fix; drop or gate the `MODELS_REPO`/`_IMPORTED_MODELS_REPO` globals fallback and restore HEAD's single-line `DEFAULT_MODELS_REPO: str = f"https://..."` formatting.

### lexnlp/extract/all_locales/amounts.py  (batch 9, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Replaces the inline `ROUTINE_BY_LOCALE.get(Locale(locale).language, ROUTINE_BY_LOCALE[DEFAULT_LANGUAGE.code])` with the new `get_language_routine(locale, ROUTINE_BY_LOCALE, DEFAULT_LANGUAGE)` helper and correctly splits the divergent backend signatures (DE `parse_annotations(text, float_digits=4, return_sources=True)` vs EN `get_amount_annotations(text, extended_sources=True, float_digits=4)`). That branching fixes a real latent bug in HEAD where positional `routine(text, extended_sources, float_digits)` mis-binds DE's second/third params. The cost is a modernisation revert: `collections.abc Generator` to `typing Generator`, `Generator[AmountAnnotation]` to `Generator[AmountAnnotation, None, None]`, plus a `# -*- coding: utf-8 -*-` header.
- **Notes:** Keep the DE `return_sources=extended_sources` vs EN `extended_sources=extended_sources` keyword split and helper; restore `from collections.abc import Generator` and `-> Generator[AmountAnnotation]` and drop the coding header. No locale loss (EN/DE only in both HEAD and PR).

### lexnlp/extract/all_locales/citations.py  (batch 9, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Same helper refactor as the other dispatchers: inline `.get(...)` fallback becomes `get_language_routine(locale, ROUTINE_BY_LOCALE, DEFAULT_LANGUAGE)` with identical `yield from routine(text)` behaviour. No locale loss (EN/DE only in both revisions) and no signature change, so the only downside is the `collections.abc Generator` to `typing Generator` and `Generator[CitationAnnotation]` to `Generator[CitationAnnotation, None, None]` revert plus coding header.
- **Notes:** Keep helper; restore `from collections.abc import Generator` and `-> Generator[CitationAnnotation]`.

### lexnlp/extract/all_locales/copyrights.py  (batch 9, part 4/10)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 95
- **Summary:** Adds the `get_language_routine` helper but deletes our ES/PT support: HEAD imports `LANG_ES`, `LANG_PT`, `get_copyright_annotations_es`, `get_copyright_annotations_pt` with a 4-entry `ROUTINE_BY_LOCALE`, while PR30 keeps only `LANG_EN`/`LANG_DE` and a 2-entry map. Callers with `es`/`pt` locales silently fall back to English instead of native routines. It also reverts `collections.abc Generator` to `typing Generator` and modern `-> Generator[CopyrightAnnotation]` to `Generator[..., None, None]`.
- **Notes:** Must restore `from lexnlp.extract.es.copyrights import get_copyright_annotations as get_copyright_annotations_es`, the `pt` equivalent, `LANG_ES`/`LANG_PT` imports, and the 4-entry `ROUTINE_BY_LOCALE` before merge; keep helper but with restored map.

### lexnlp/extract/all_locales/court_citations.py  (batch 9, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 82
- **Summary:** Switches to `get_language_routine(locale, ROUTINE_BY_LOCALE, LANG_DE)` (preserving the German-default fallback) and coerces the `language` argument to `language or (locale_language if in ROUTINE_BY_LOCALE else LANG_DE.code)` instead of HEAD's pass-through `yield from routine(text, language)` which forwards `None`. The coercion is arguably a fix for backends expecting a string, but the patch deletes HEAD's docstring, reverts `collections.abc Generator` to `typing Generator`, and downgrades `language: str | None = None` to `language: str = None`.
- **Notes:** Keep the `annotation_language` coercion if the DE backend requires non-None; restore `from collections.abc import Generator`, `-> Generator[CourtCitationAnnotation]`, `str | None` hint, and the docstring. No locale loss (DE-only map in both).

### lexnlp/extract/all_locales/dates.py  (batch 9, part 6/10)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** Adds `get_language_routine` plus DE-specific adaptation (`strict = False if strict is None`, `locale=Locale(locale)` for DE vs raw string otherwise, keyword call) matching PR30's rewritten `de/dates.py` (`_coerce_locale`/`_build_parser` with `Locale` objects). However it deletes ES/PT: HEAD has `LANG_ES`/`LANG_PT`, `get_date_annotations_es/pt`, and a 4-entry map, while PR30 has only EN/DE. It also reverts `collections.abc Generator` to `typing Generator`, `bool | None`/`datetime | None` to `Optional[...]`, and `Generator[DateAnnotation]` to `Generator[..., None, None]`.
- **Notes:** Must restore ES/PT imports and 4-entry `ROUTINE_BY_LOCALE`; keep helper/strict-default only after verifying against HEAD's `en/dates.py (text, strict: bool | None, locale: str | None, base_date, threshold)` vs DE parser-bound `get_date_annotations` signatures, and restore modern hints.

### lexnlp/extract/all_locales/definitions.py  (batch 9, part 7/10)
- **Regression:** YES
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 92
- **Summary:** Same pattern: `get_language_routine(locale, ROUTINE_BY_LOCALE, DEFAULT_LANGUAGE)` with unchanged `yield from routine(text, **kwargs)` is fine, but the patch drops ES/PT (HEAD 4-entry map with `get_definition_annotations_es/pt` and `LANG_ES`/`LANG_PT`; PR 2-entry EN/DE only), so `es`/`pt` callers regress to English. Also reverts `collections.abc Generator` to `typing Generator` and `-> Generator[DefinitionAnnotation]` to `Generator[..., None, None]`.
- **Notes:** Restore `es`/`pt` imports and map entries; keep helper and restore `from collections.abc import Generator`.

### lexnlp/extract/all_locales/durations.py  (batch 9, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Functionally equivalent refactor to `get_language_routine(locale, ROUTINE_BY_LOCALE, DEFAULT_LANGUAGE)` with `yield from routine(text=text, float_digits=float_digits)`; no locale loss (EN/DE only in both). The regression is purely modernisation: `collections.abc Generator` to `typing Generator`, `-> Generator[DurationAnnotation]` to `Generator[..., None, None]`, single-line `ROUTINE_BY_LOCALE` and signature expanded, plus coding header.
- **Notes:** Keep helper/keyword call; restore `from collections.abc import Generator` and `-> Generator[DurationAnnotation]`.

### lexnlp/extract/all_locales/geoentities.py  (batch 9, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Replaces the inline fallback with `get_language_routine` with identical positional delegation; no locale loss (EN/DE map unchanged). The patch reverts HEAD's modernisation: `collections.abc Generator` to `typing Generator, List, Optional, Tuple, Dict`, `list[DictionaryEntry]`/`list[str] | None`/`dict[str, tuple[list[str], list[str]]] | None` to `List[...]`/`Dict[...]`/`Optional[...]`, `text_languages: list[str] | None = None` to `List[str] = None`, double to single quotes, and deletes HEAD's full parameter docstring.
- **Notes:** Keep helper; restore HEAD signature (`list[...]`/`... | None`, `from collections.abc import Generator`, double quotes) and the docstring. `List[str] = None` is also a typing error (should be `Optional`).

### lexnlp/extract/all_locales/languages.py  (batch 9, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Adds genuinely useful `get_language_routine(locale_name, routines, default_language)` (explicit per-callsite default, documents EN-default vs court-citation DE-default) and a thread-safety `_LOCALE_LOCK = RLock()` around `LocaleContextManager` (HEAD has no lock). But it reverts HEAD modernisation: `collections.abc Sequence` to `typing Optional/TypeVar/Mapping`, `str | None` to `Optional[str]`, double to single quotes, detailed docstrings to the old stdlib-style docstring, `locale.getlocale()`-tuple save to `setlocale()`-string save, and deletes `LANG_PT`/`LANGUAGES` entry (HEAD `LANGUAGES = [EN, DE, ES, PT]`, PR `[EN, DE, ES]`).
- **Notes:** Keep `_LOCALE_LOCK` + `get_language_routine`/`Routine = TypeVar('Routine')`; restore `LANG_PT`, 4-entry `LANGUAGES`, `from collections.abc import Sequence`, `str | None` hints, double quotes, and HEAD docstrings. Verify lock acquisition/release against concurrent `LocaleContextManager` use before merge.

### lexnlp/extract/common/date_parsing/datefinder.py  (batch 11, part 1/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 92
- **Summary:** The pasted 1-line hunk (`"""` -> `r"""` in `refine_pattern_start_end`) is misleading; the true HEAD-vs-pr30 diff is ~227 lines and is a wholesale revert of modernisation. It swaps builtin generics/f-strings/double-quotes for legacy `typing.List/Tuple/Dict/Optional`, `.format()` and single quotes, and deletes the HEAD locale hardening (`has_real_locale`, language-only fallback on `as_dt is None`, dateutil-fallback guards). It also reverts `rf"..."` to `fr'...'`, `yield from range_strings` to an explicit loop, and `TIME_PATTERN` from `rf"""...{TIME_PERIOD_PATTERN}...` interpolation to `r"""....format(time_periods=..., timezones=...)`.
- **Notes:** Examples: HEAD `def tokenize_string(self, text: str) -> list[tuple[str, str, dict[str, list[str]]]]` vs PR `-> List[Tuple[str, str, Dict[str, List[str]]]]`; HEAD `f'"{c}": [{self.captures[c]}]'` vs PR `'\"{}\": [{}]'.format(...)`; HEAD strict `if len(digits) == 3 or ((len(months) == 1) and (len(digits) == 2))` vs PR split into two branches. Nothing in this file should merge as-is; if the `r"""` docstring fix is wanted, cherry-pick only that character.

### lexnlp/extract/common/dates.py  (batch 11, part 2/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** The PR contains one real bug fix: `get_dateparser_dates` builds a copied `settings = {**self.dateparser_settings, 'STRICT_PARSING': strict, 'DATE_ORDER': ...}` instead of HEAD's mutate-then-restore of `self.dateparser_settings` (which aliases the class-level `DEFAULT_DATEPARSER_SETTINGS` and even saves `PREFER_DAY_OF_MONTH` but restores `DATE_ORDER`). It also narrows `except Exception: print(str(e))` to `except (OverflowError, TypeError, ValueError): self.dates = []`, removing a `print`. The rest is a modernisation revert: `str | None`/`list`/`dict`/`Generator[DateAnnotation]` become `Optional`/`List`/`Dict`/`Generator[..., None, None]`, double quotes become single, and the detailed `get_dates`/`get_date_annotations`/`get_date_annotation_list` docstrings plus the coordinated-phrase newline comment are deleted.
- **Notes:** Keep only the settings-copy and the narrowed-except (without `print`); keep HEAD signatures (`text: str | None`, `-> list[DateAnnotation]`), docstrings, and `self.locale.language` handling. Flag behavior change: HEAD raises `RuntimeError('Define text and language.')` on empty text while PR silently `return`s, and PR replaces `self.locale.language = ...` mutation with `self.locale = Locale(locale.get_locale())` plus `text: str = None` legacy default.

### lexnlp/extract/common/dates_classifier_model.py  (batch 11, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** The one good hunk is `except:` plus `# pylint: disable=broad-except` becoming `except Exception:`, which is a genuine lint fix. Everything else is a style/typing revert: HEAD's `collections.abc.Callable`, `list[tuple[str, list[datetime.date]]]`, `set[str] | None`, f-strings and black-formatted `sklearn.pipeline.Pipeline` blocks are rewritten to `typing.List/Tuple/Callable/Dict/Union/Set/Optional`, single quotes, `.format()` and old pipeline layout.
- **Notes:** Keep `except Exception:`; reject `def build_date_model(input_examples: List[Tuple[str, List[datetime.date]]], ... alphabet_char_set: Optional[Set[str]] ...)` and `def get_date_features(... characters: List[str] ...)` in favor of HEAD `list[...]`/`set[...] | None` plus `from collections.abc import Callable`. Quote/style-only churn (`f"Accuracy: ..."` vs `"Accuracy: {0}%...".format(...)`) has no functional value.

### lexnlp/extract/common/definitions/common_definition_patterns.py  (batch 11, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** The rename `l`/`l_upper` to `letter`/`letter_upper` is a genuine readability fix (`l` is confusable with `1` and trips linters) with no logic change (`is_correct = name[acr_index] == letter_upper`). The remainder reverts HEAD modernisation: `from collections.abc import Callable` + `from re import Match` and `list[PatternFound]` become `from typing import List, Match, Callable` and `List[PatternFound]`, with double-to-single quote churn and black-to-legacy reformatting of `peek_quoted_part`/`collect_regex_matches*`.
- **Notes:** Cherry-pick only the `letter`/`letter_upper` hunk; keep HEAD imports and `-> list[PatternFound]` annotations. Functional behavior is otherwise identical.

### lexnlp/extract/common/definitions/universal_definition_parser.py  (batch 11, part 5/10)
- **Regression:** NO
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** The pasted hunk (`-from typing import Dict, Generator, Union`) is stale: HEAD already lacks that import (verified via `git show HEAD:`), so there is no import cleanup left to merge. The true HEAD-vs-pr30 diff is whitespace-only reformatting of `make_annotation_from_pattern` (`text=phrase.text[ptrn.start : ptrn.end]` black spacing vs `phrase.text[ptrn.start: ptrn.end]` plus line splits) with identical behavior.
- **Notes:** No functional change either way; reject to preserve HEAD black formatting. Do not re-add any `typing` import here.

### lexnlp/extract/common/durations/durations_parser.py  (batch 11, part 6/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** The pasted `from typing import List, Pattern, Callable` -> `from typing import List` hunk hides the real direction: HEAD has zero `typing` imports and uses PEP 585 `list[DurationAnnotation]`, while PR reintroduces `from typing import List` and rewrites every annotation to `List[DurationAnnotation]`/`List[List[DurationAnnotation]]`/`List`. It also reverts black formatting (`text[a.coords[1] : b.coords[0]]` to `text[a.coords[1]:b.coords[0]]`, `""` to `''`) with no behavior change.
- **Notes:** Keep HEAD: no `typing` import, `-> list[DurationAnnotation]`, `ant_group: list[DurationAnnotation]`, `annotations: list`. Merging would reintroduce the deprecated `typing.List` alias our modernisation removed.

### lexnlp/extract/common/fact_extracting.py  (batch 11, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Two small hunks are good: `target_types = include_types` becomes `set(include_types)`, preventing aliasing of the caller's set (companion to the new regression test). But the file is otherwise a broad revert: `from collections.abc import Callable` plus `set[...]|None`/`dict[...]`/`list[...]` and full `parse_text`/`ensure_parser_arguments_en/de` docstrings become `from typing import Callable, Dict, Set, List, Any, Tuple`, `Set`/`Dict`/`List` annotations, single quotes, and stripped docs. Critically it reintroduces a bug: HEAD `extra_args.get(result_fmt_key)` (which honors the `fmt_dict`->`fmt_class` fallback) is reverted to `extra_args.get(result_fmt)`, silently losing extra-arg wiring for `fmt_dict` calls.
- **Notes:** Keep only `target_types = set(include_types)` (HEAD already has `target_types = set(ALL_ANT_TYPES)`; `difference_update` vs `-=` after the copy is equivalent). Reject the `typing` rewrite, the `extra_args.get(result_fmt)` regression, and docstring deletions.

### lexnlp/extract/common/ocr_rating/ocr_rating_calculator.py  (batch 11, part 8/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** The pasted `Dict` import removal is trivial compared to the true HEAD-vs-pr30 revert: HEAD uses `from collections.abc import Callable`, `list[str]`/`list[str] | None` and reuses `from lexnlp.utils.cosine import cosine_similarity`, while PR reintroduces `from typing import Optional, List, Callable`, rewrites signatures to `Optional[List[str]]`, re-adds the obsolete `# -*- coding: utf-8 -*-` header, and inlines `numpy.dot(...)/numpy.linalg.norm(...)` instead of calling the shared `cosine_similarity` utility. Quote churn (double to single) and `10.0` to `10.` accompany the revert.
- **Notes:** Merging would violate the "prefer existing utilities under `lexnlp/utils/`" guideline and reintroduce deprecated `typing` aliases. Keep HEAD `def init_language_data(self, data_folders: list[str], language_file_paths: list[str] | None = None)` and `return cosine_similarity(...)`.

### lexnlp/extract/common/tests/test_fact_extractor.py  (batch 11, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** The new `test_exclusions_do_not_mutate_global_annotation_types` test (with `from unittest.mock import patch`, `ExtractingFunction`, `patch.dict(FactExtractor.func_by_lang, {'test': ...})` and `parse_text(..., extract_all=True, exclude_types={AnnotationType.money})`) is valuable and directly guards the `fact_extracting.py` aliasing fix. The rest of the diff is style/docstring revert: HEAD double quotes and `make_geoconfig`/`test_one_fact_en`/`test_all_facts_en` docstrings become single quotes with docs deleted and calls reformatted.
- **Notes:** Cherry-pick only the new test plus its `ExtractingFunction`/`patch` imports; keep HEAD quotes, docstrings, and `FactExtractor.parse_text(text, ..., include_types={...})` formatting.

### lexnlp/extract/common/tests/test_universal_courts_parser.py  (batch 11, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** The substantive change is good: `load_en_courts` and `make_en_parser` stop fetching `https://raw.githubusercontent.com/LexPredict/lexpredict-legal-dictionary/1.0.2/en/legal/us_courts.csv` over the network and instead use the bundled `COURTS_DATA_PATH = Path(get_module_path()) / "config" / "en" / "us_courts.csv"` (verified present at `lexnlp/config/en/us_courts.csv`), making tests hermetic/offline. The remainder reverts HEAD: detailed `test_compare_to_legacy_parser`/`load_en_courts`/`make_en_parser` docstrings are deleted, double quotes become single, and black formatting is undone.
- **Notes:** Keep `from pathlib import Path`, `from lexnlp import get_module_path`, `COURTS_DATA_PATH`, `pandas.read_csv(self.COURTS_DATA_PATH)` and `ptrs.dataframe_paths = [str(self.COURTS_DATA_PATH)]`; reject docstring deletions and quote/formatting churn.

### lexnlp/extract/de/tests/test_laws.py  (batch 13, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 92
- **Summary:** Fixes an off-by-one expectation: `(18, 24)` becomes `(19, 23)` for `AAÜG` in `Dies ist durch das AAÜG geschehen.`, and adds the invariant `ret[0].text == text[slice(*ret[0].coords)]`. The new coords are correct (`AAÜG` starts at index 19).
- **Notes:** `- (18, 24)` included surrounding spaces; `+ (19, 23)` matches `text.index('AAÜG')`.

### lexnlp/extract/de/tests/test_money.py  (batch 13, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** Strengthens the existing money test by asserting 5 annotations with amounts `[Decimal('300'), Decimal('10800'), Decimal('2500'), Decimal('230.55'), Decimal('10800')]` and currencies `['EUR'] * 5`. Test-only addition, no production-code downgrade.
- **Notes:** No `-` behavior lines; purely additive assertions on `get_money_annotations(text)`.

### lexnlp/extract/de/tests/test_percents.py  (batch 13, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Corrects percent expectations from raw amounts to fractions: `15.0->Decimal('0.15')`, `18.0->Decimal('0.18')`, `20->Decimal('0.2')`, including `res[0].fraction`. This matches the correct semantics `fraction = amount * Decimal(0.01)`.
- **Notes:** Pairs with the `lexnlp/extract/de/percents.py` fix `round(amount, ...)` -> `round(real_amount, ...)`; master had `real_amount = round(amount, float_digits)` bug.

### lexnlp/extract/en/addresses/address_features.py  (batch 13, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 98
- **Summary:** Pure variable rename in `_load_set_from_lines`: `l` becomes `line` in both set comprehensions. No semantic, API, or dependency change.
- **Notes:** `_norm(l.strip())` -> `_norm(line.strip())`; safe to merge.

### lexnlp/extract/en/addresses/addresses.py  (batch 13, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Fixes `get_address_annotations` to set `text=text[start:end]` instead of the whole document `text=text`, renames unused loop var to `_address`, and reorders kwargs. This aligns `AddressAnnotation.text` with `coords`, as verified by the new test in this batch.
- **Notes:** This hunk only; broader file-level reversions outside this hunk (e.g. skops-loader removal, `typing.List/Tuple` vs modern syntax) belong to other chunks and are not endorsed here.

### lexnlp/extract/en/addresses/addresses_clf.pickle  (batch 13, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 30
- **Summary:** Binary model blob `addresses_clf.pickle` differs with no reviewable diff. It may be an intentional retrained classifier for the SOTA-segmentation work, but provenance, training code, and metrics are absent, so it cannot be accepted blindly.
- **Notes:** Binary files differ; require training provenance plus accuracy/no-regression check and skops-sibling handling before merging. Unsure — low confidence by nature.

### lexnlp/extract/en/addresses/tests/test_addresses.py  (batch 13, part 7/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 90
- **Summary:** Adds `test_address_annotation_constructor_and_source_span` mocking `get_address_spans` and asserting `coords`, `text == text[start:end]`, and serialized `Extracted Entity Text`. Also cleans lambda names (`l`/`t` -> `spans`/`span`). Directly covers the `addresses.py` fix above.
- **Notes:** New imports `patch` and `AddressAnnotation` are test-only.

### lexnlp/extract/en/amounts.py  (batch 13, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 88
- **Summary:** Two lint-level cleanups: drops the unused `as invalid_operation` from `except InvalidOperation`, and narrows bare `except:` to `except Exception:` around `text2num(found_item)`. The latter is strictly better (no longer swallows `KeyboardInterrupt`/`SystemExit`).
- **Notes:** No logic change; `except Exception: continue` preserves existing skip-on-bad-amount behavior.

### lexnlp/extract/en/citations.py  (batch 13, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** Narrows annotation span from `match.span()` to `match.span(1)` (excludes the leading `[\s,:(]` delimiter) and populates `CitationAnnotation(text=source_text)` with pre-stripped `source_text`. This makes `coords` consistent with `text`/`source`.
- **Notes:** Supported by new tests asserting `text[slice(*coords)] == text`; typing-style reversions elsewhere in this file are outside this hunk and not endorsed here.

### lexnlp/extract/en/conditions.py  (batch 13, part 10/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 78
- **Summary:** Rewrites condition extraction for speed but breaks the API: the `pre`/`post` regex groups are deleted, `post` is now always `""`, `strict` is silently ignored, and `coords` changes from `match.span()` to `(cursor, match.end())`. The perf concern about leading `.*?` may be real, but this discards documented behavior.
- **Notes:** `CONDITION_PATTERN_TEMPLATE` loses `(?P<pre>.*?)`/`(?P<post>.*?)`; `capturesdict()` pre/post handling and `if strict and (num_pre == 0 or num_post == 0): continue` are deleted; also reverts modern `collections.abc Generator` / `str | None` / `list[...]` to `typing` forms.

### lexnlp/extract/en/date_model.pickle  (batch 15, part 1/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 95
- **Summary:** Re-adds the legacy insecure `date_model.pickle` blob (0 -> 56085 bytes) that master deleted when migrating to SECURE `.skops`. Master has only `lexnlp/extract/en/date_model.skops`; the PR has only the `.pickle`. Merging reintroduces arbitrary-code-on-load persistence.
- **Notes:** Known master state: skops migration + `skops>=0.11` dependency; PR drops both. See companion `date_model.py` revert below.

### lexnlp/extract/en/date_model.py  (batch 15, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** Reverts the secure loader `from lexnlp.ml.model_io import load_bundled_model` / `MODEL_DATE = load_bundled_model(...date_model.pickle)` (skops-aware with legacy fallback) to legacy `from lexnlp.utils.unpickler import load_joblib_model` / `load_joblib_model(...)`. This undoes the skops security work and points at a `.pickle` path that does not exist on master (asset is `.skops`).
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/extract/en/date_model.py` confirms; pasted three-dot diff understates it by showing old `import joblib` base.

### lexnlp/extract/en/dates.py  (batch 15, part 3/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** The pasted one-line hunk (`except:` -> `except Exception:`) is a stale merge-base artifact: master at ec55727 already has `except Exception:` (dates.py:255) with the pylint disable. The TRUE net effect (`git diff ec55727..pr30`) is a broad modernisation revert: `collections.abc Generator` -> `typing Generator`, inline `rf"""...{DATE_MAX_LENGTH}...` -> old `r"""...format(max_length=...)`, builtin `list` -> `List`, Google-style docstrings -> old `:param:` style, double -> single quotes, plus mode 644->755.
- **Notes:** No functional gain in this file; reject the churn. The `except Exception` line itself is a no-op vs master.

### lexnlp/extract/en/definition_parsing_methods.py  (batch 15, part 4/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 82
- **Summary:** Pasted hunk (dropping unused `Set` from the typing import) looks trivial, but the true diff vs master is a style/typing revert: master uses builtin `tuple[int, int]`, double quotes, and inline `rf"""...{join_collection(...)}...` with `TRIGGER_WORDS_PTN`; the PR reintroduces `from typing import Pattern, List, Tuple`, `Tuple[int, int]`, single quotes, `.format(max_term_chars=..., trigger_list=...)`, backslash continuations, and import reordering. No behavior fix.
- **Notes:** If the unused `Set` import still exists anywhere, remove just that word on master; do not take this file wholesale.

### lexnlp/extract/en/definitions.py  (batch 15, part 5/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 85
- **Summary:** Same pattern: pasted hunk drops an unused `Tuple` import, but the true diff reverts master modernisation (`from collections.abc import Generator`, `list[DefinitionCaught]`, `Generator[DefinitionAnnotation]`, double quotes, black formatting) back to `from typing import Generator, List`, `List[...]` / `Generator[..., None, None]`, single quotes, and re-wrapped signatures. No functional change worth the revert.
- **Notes:** Take at most the one-word unused-import deletion if it still applies on master.

### lexnlp/extract/en/entities/stanford_ner.py  (batch 15, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Contains one genuine bug fix buried in style churn: in `get_persons`/`get_organizations`/`get_locations`, master tags the whole document per sentence (`STANFORD_NER_TAGGER.tag(get_tokens_list(text))` inside `for sentence in get_sentence_list(text)`), while the PR correctly tags `get_tokens_list(sentence)`. The rest of the file reverts modernisation (`collections.abc Generator` -> `typing Generator`, double -> single quotes, dict/call reformatting).
- **Notes:** Cherry-pick ONLY the three `text` -> `sentence` lines; discard the `typing Generator`, quote, and formatting churn. Covered by the new mocked test in part 7/10.

### lexnlp/extract/en/entities/tests/test_stanford_ner.py  (batch 15, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Adds a valuable dependency-free `test_stanford_ner_tags_each_sentence_once` (RecordingTagger + monkeypatched `get_sentence_list`/`get_tokens_list`) proving each sentence is tagged exactly once with per-sentence tokens for persons/organizations/locations. It validates the part 6/10 fix without Stanford jars. The file also needlessly reformats three existing Stanford-gated tests from master's black multi-line style to old single-line style.
- **Notes:** Keep the new `RecordingTagger` test verbatim; discard the reformatting of `test_stanford_name_example_in` / `test_stanford_org_example_in` / `test_stanford_locations`.

### lexnlp/extract/en/ratios.py  (batch 15, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 65
- **Summary:** The functional hunk passes `extended_sources=False` to both `get_amounts()` calls in `get_ratio_annotations`, so ratio halves are parsed as bare amounts without attaching surrounding NP/prev-unit context; given `get_amounts(extended_sources=True)` is the default on both branches, this plausibly tightens the `len(amount)==1` gate. The same file also reverts typing modernisation (`collections.abc Generator`, `tuple[...]|...`, `list[...]` -> `typing Generator`/`Union`/`Tuple`/`List`) and double -> single quotes.
- **Notes:** Cherry-pick ONLY the two `extended_sources=False` arguments pending a targeted ratios test run; discard the typing/quote reformatting (`NUM_PTN` replace chains, `RatioAnnotation(...)` rewrapping, signature rewraps).

### lexnlp/extract/en/tests/test_citations.py  (batch 15, part 9/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** Adds two useful tests: `test_citation_annotation_has_exact_source_span` (coords/text/slice consistency for `get_citation_annotations`) and `test_citation_serializes_source_and_court_without_reporter` (direct `CitationAnnotation` tag serialization). They cover the genuine `citations.py` fix (span `match.span()` -> `match.span(1)` plus `text=source_text`, judged MERGE in batch 13). The file simultaneously reverts master's modernized docstrings and black formatting to old single-quote/collapsed style.
- **Notes:** Keep the two new tests; discard the reformat of `test_get_citations`/`test_get_citations_as_dict`. Pair the span test with the `citations.py` span fix -- it FAILS on unfixed master (master leaves `text=""` and uses whole-match coords).

### lexnlp/extract/en/tests/test_condition_constraint_security.py  (batch 15, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 60
- **Summary:** New standalone test file (absent on master) with a valuable sub-second DoS bound (`test_long_trigger_free_and_near_miss_text_is_bounded`, 10k-char inputs) plus three multi-span `coords`/`pre`/`post` assertions for conditions/constraints. The timing test is safe to take, but the span expectations (notably `post=""` throughout and cursor-based coords like `(0,28)`/`(28,46)`) appear to codify the conditions/constraints rewrite REJECTED in batch 13 for deleting `pre`/`post` groups and ignoring `strict`.
- **Notes:** Merge the perf-bound test; do NOT merge the three span-preservation tests until `conditions.py`/`constraints.py` behavior is settled against master -- verify `get_condition_annotations`/`get_constraint_annotations` imports and expected spans on ec55727 first. Unsure on span correctness without running master; hence low confidence.

### lexnlp/ml/README.md  (batch 17, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** Documents the PR's new download surface (`download_github_release_to_path(tag, *, manifest_path=None, force=False) -> Path`, single manifest-pinned prompt instead of the old 2-prompt size flow, `LEXNLP_MODELS_REPO`/`LEXNLP_MODELS_REPO_SLUG` config, `_local-candidates/` namespace). The prose matches the new code in this batch and does not itself downgrade pins, types, or async behavior.
- **Notes:** Verified against ec55727: master README still describes the 4-step prompt flow; the "prompt once with manifest-pinned size" line is accurate only if the `download.py` manifest work is taken. Pair with the download.py verdict.

### lexnlp/ml/artifact_abi.py  (batch 17, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 92
- **Summary:** New file pins `MODEL_ARTIFACT_RUNTIME = {"python": "3.12.13", "numpy": "1.26.4", "pandas": "2.2.0", "scikit_learn": "1.7.2", "scipy": "1.13.0", "joblib": "1.5.0", "threadpoolctl": "3.6.0"}` and makes `assert_model_artifact_runtime()` fail unless models are built with `constraints/model-artifact-abi.txt` (a file absent on master). Python 3.12.13 is below master's `requires-python = ">=3.13,<3.15"` and numpy 1.26.4 is below master's `numpy>=2.3,<3`, so this enshrines an older ABI as the build gate.
- **Notes:** Quote: `"python": "3.12.13"`, `"numpy": "1.26.4"` vs master `requires-python >=3.13,<3.15`, `numpy>=2.3,<3`, `scikit-learn>=1.5`, `skops>=0.11`. Missing on master (`git show ec55727:lexnlp/ml/artifact_abi.py` = absent).

### lexnlp/ml/artifact_io.py  (batch 17, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Adds `atomic_output_path()` (sibling tmp + fsync + mode-preserving atomic replace, symlink refusal, empty-file refusal) which is genuinely useful and reused by the new downloader, plus `atomic_pickle_dump()` which makes `pickle.dump` the persistence target. The atomic-write helper is an improvement; the pickle-dump entry point directly contradicts master's secure `.skops` direction (`lexnlp/ml/model_io.py` with `CANONICAL_SUFFIX = ".skops"`, `dump_model`/`load_model(trusted=True)`).
- **Notes:** Keep `atomic_output_path` as-is; drop or reimplement `atomic_pickle_dump` on top of `model_io.dump_model` (skops) before merging. Missing on master; `import pickle` + `pickle.dump(value, output_file)` is the regressive symbol.

### lexnlp/ml/catalog/__init__.py  (batch 17, part 4/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** Adds useful hardening: `_catalog_tag()` with `.as_posix()` (Windows-separator fix), `get_local_candidate_tag()`/`get_catalog_directory()` with traversal validation, and `get_exact_path_from_catalog()` plus a `_local-candidates/` fallback in `get_path_from_catalog`. In the same file it removes master's thread-safety modernisation (`_TAG_DICT_LOCK = threading.Lock()`, double-checked locking in `_get_tag_dict_cached`/`invalidate_catalog_cache`) and reverts `dict[str, Path] | None` to `Optional[Dict[str, Path]]`.
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/ml/catalog/__init__.py`: `-_TAG_DICT_LOCK`, `-with _TAG_DICT_LOCK:` are the regressions; `LOCAL_CANDIDATE_TAG_PREFIX = "_local-candidates"`, `_catalog_tag`, `get_local_candidate_tag` are the keepers. Cherry-pick helpers + posix fix, restore the lock.

### lexnlp/ml/catalog/download.py  (batch 17, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 82
- **Summary:** Adds a real security improvement over master: trusted `AssetManifest`/`TrustedAsset` with manifest-bound SHA-256 + size + filename checks, HTTPS hostname pinning, tag-traversal rejection, atomic install via `artifact_io.atomic_output_path`, cached-verified short-circuit with `force=` override, and `verify_trusted_asset_file/payload` helpers, while preserving legacy `download_release`/`download_asset` `-> None` return contracts. In the same file it deletes master's resilience modernisation: `build_retry_session()`/`_session()` (`requests.Session` + `urllib3.util.retry.Retry` on 429/500/502/503/504, connection pooling, context-managed streaming) reverted to bare `requests.get`.
- **Notes:** `git show pr30:.../download.py | grep -c Session/Retry` = 0 vs master = 12. Keep the entire manifest/verification/atomic-install path; restore `_session().get(...)` + `build_retry_session` (and the socket-leak context manager) underneath it. Manifest `models_repo_slug LexPredict/lexpredict-lexnlp` matches master `DEFAULT_MODELS_REPO`.

### lexnlp/ml/catalog/release_asset_manifest.json  (batch 17, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New packaged trust manifest (`schema_version: 1`, `models_repo_slug: LexPredict/lexpredict-lexnlp`, 9 assets with `tag/filename/size/sha256`). It extends master's `test_data/model_quality/release_asset_manifest.json` (7 assets, no `schema_version`) with the two pipeline entries `pipeline/is-contract/0.2` (size 38794068, sha `083bab99...`) and `pipeline/contract-type/0.2-runtime` (size 33769286, sha `3b04a8a9...`) that the new downloader and security tests depend on.
- **Notes:** Absent on master (`git show ec55727:lexnlp/ml/catalog/release_asset_manifest.json` = missing). Verify the two new digests against the actual release bytes before merging; ensure the `test_data/...` copy is updated in the same cherry-pick so the byte-identical test can pass.

### lexnlp/ml/catalog/tests/test_catalog_path.py  (batch 17, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 90
- **Summary:** Adds one good test (`test_catalog_tags_use_posix_separators_on_windows` for `_catalog_tag`) but deletes master's four thread-safety tests (`test_get_tag_dict_cached_builds_cache_once`, `test_get_tag_dict_cached_thread_safety`, `test_invalidate_catalog_cache_is_thread_safe`, `test_invalidate_catalog_cache_clears_cache`) that pin the `_TAG_DICT_LOCK` behavior removed in `catalog/__init__.py`. Deleting coverage for a concurrently-removed lock masks the regression and violates the skip/test-integrity policy.
- **Notes:** True diff `git diff ec55727..pr30 -- .../test_catalog_path.py` is +10/-133. Keep the posix test; restore all four deleted thread-safety tests together with the lock.

### lexnlp/ml/catalog/tests/test_download_security.py  (batch 17, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** New 628-line security/compatibility suite for the manifest downloader: legacy `get_asset` index + trusted-filename paths, deferred `raise_for_status` behavior, legacy `verify_md5` helper, `None`-return contracts, env-vs-`MODELS_REPO` precedence, packaged-manifest pin checks, unlisted-tag network embargo, custom-repo mismatch, unsafe-tag matrix, hostname and symlink-escape rejection, atomic install/permission preservation, truncation/oversize handling, and verified-cache network avoidance. No master counterpart; purely additive coverage.
- **Notes:** Depends on `release_asset_manifest.json` pins (`pipeline/is-contract/0.2` sha `083bab99...`, `contract-type/0.2-runtime` sha `3b04a8a9...`) and on keeping the `download.py` manifest path -- merge together. No downgrade symbols observed.

### lexnlp/ml/predictor.py  (batch 17, part 9/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** Switches default-pipeline loading from master's secure `lexnlp.ml.model_io.load_model(path, trusted=True)` (skops, `DEFAULT_TRUSTED_ALLOWLIST`) to `lexnlp.utils.unpickler.load_sklearn_model` (pickle-based), and deletes master's `_patch_legacy_estimator_attributes()` (sigma_/var_/variance_ alias) plus the `MinMaxScaler.clip` repair from the predictor. The replacement `restore_legacy_model_state`/`CompatibilityReport` path re-centralizes pickle repair instead of using the skops gate, contradicting the confirmed `.skops` modernisation. Incidental `X | None` -> `Optional[X]` and quote churn included.
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/ml/predictor.py`: `-from lexnlp.ml.model_io import load_model` / `-return load_model(path, trusted=True)` vs `+from lexnlp.utils.unpickler import (CompatibilityReport, load_sklearn_model, restore_legacy_model_state)`. Do not merge without re-routing through `model_io.load_model`.

### lexnlp/ml/tests/test_artifact_abi.py  (batch 17, part 10/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** New test pins the regressed ABI: `test_assert_model_artifact_runtime_accepts_committed_versions` requires the old `MODEL_ARTIFACT_RUNTIME` dict (python 3.12.13, numpy 1.26.4) to pass, and `test_artifact_metadata_includes_hash_and_runtime` locks `artifact_metadata()` output shape. Merging it would enshrine the `artifact_abi.py` downgrade (below master's `requires-python >=3.13,<3.15` / `numpy>=2.3`) as expected behavior.
- **Notes:** Tied to `lexnlp/ml/artifact_abi.py` (absent on master); reject together. If an ABI test is wanted, rewrite it against master's actual runtime (`numpy>=2.3`, python 3.13+) and `model_io` skops metadata instead.

### lexnlp/nlp/en/segments/payloads.py  (batch 19, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** New 305-line module adding versioned, budget-checked embedding payloads (`EmbeddingPayload`, `ContextFragment`, `render_embedding_payload`, `PayloadBudgetExceeded`, `EMBEDDING_PAYLOAD_SERIALIZATION_VERSION = "lexnlp.embedding_payload.v1"`). Absent on master (`git cat-file -e ec55727:lexnlp/nlp/en/segments/payloads.py` fails); purely additive SOTA-segmentation feature with strict validation (SHA-256 text/metadata digests, exact tokenizer budget, no silent trimming). Modern style (`from __future__ import annotations`, frozen dataclasses, `TypeAlias`).
- **Notes:** Depends on sibling new modules `lexnlp/nlp/en/segments/chunks.py` (`DocumentChunk`) and `hierarchy.py` (`SegmentKind`) — cherry-pick together. Minor nit: imports `Callable, Iterable, TypeAlias` from `typing`; master prefers `collections.abc` for `Callable/Iterable`, but not a blocker.

### lexnlp/nlp/en/segments/section_segmenter.pickle  (batch 19, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Binary change re-adds an insecure pickle artifact. True net effect (`git diff --stat ec55727..pr30`) is `section_segmenter.skops (35469 bytes) -> deleted` and `section_segmenter.pickle (0 -> 6198 bytes) added`; master has only `.skops` in `lexnlp/nlp/en/segments/` and no `.pickle`. This is the confirmed SECURITY REGRESSION (drops `skops>=0.11`, restores arbitrary-code-on-load pickle).
- **Notes:** Pasted three-dot diff (`96a1c20..5e78a3c`, pickle-vs-pickle) hides the deletion; judge via `git diff ec55727..pr30 -- lexnlp/nlp/en/segments/`. Reject together with the `load_joblib_model` caller change in `sections.py`.

### lexnlp/nlp/en/segments/sections.py  (batch 19, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 88
- **Summary:** Mix of a genuine schema fix and modernisation reversions. Valuable part: fixed global feature schema (edge-row clipping via `min()`, `get_section_feature_names` no longer drops columns by document length), plus `has_compatible_line_window` / `has_compatible_feature_width` fail-closed guards and early `return` on empty/mismatched input in `get_sections`. Regression part: loader switched from master's secure `lexnlp.ml.model_io.load_bundled_model(...section_segmenter.pickle)` (skops-preferring) to `lexnlp.utils.unpickler.load_joblib_model`, and style downgraded (`ClassVar[list[str]]` -> `[]`, `collections.abc.Generator` -> `typing.Generator`, f-strings -> `.format`, double -> single quotes).
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/nlp/en/segments/sections.py`: `-from lexnlp.ml.model_io import load_bundled_model` vs `+from lexnlp.utils.unpickler import load_joblib_model`; `-FEATURE_NAMES: ClassVar[list[str]] = []` vs `+FEATURE_NAMES = []`. Keep the `TRAINED_LINE_WINDOW_PRE/POST` + compat-guard logic, re-route loading through `load_bundled_model` and restore master's typing/f-string style.

### lexnlp/nlp/en/segments/sentences.py  (batch 19, part 4/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** No functional improvement; only a loader security downgrade plus cosmetic churn. True diff (`git diff ec55727..pr30`) replaces `from lexnlp.ml.model_io import load_bundled_model` / `load_bundled_model(...sentence_segmenter.pickle)` with `from lexnlp.utils.unpickler import load_joblib_model` / `load_joblib_model(...)`, bypassing the `.skops` sibling master ships. Remainder is downgrade churn: `collections.abc.Generator` -> `typing.Tuple/List/Generator/Union`, builtin generics (`tuple[int,int]`, `None | tuple`) -> `Tuple`/`Union`, double -> single quotes, `_trim_span`/`post_process_sentence` resplit.
- **Notes:** Symbols: `-SENTENCE_SEGMENTER_MODEL: PunktSentenceTokenizer = load_bundled_model(...)` vs `+SENTENCE_SEGMENTER_MODEL: PunktSentenceTokenizer = load_joblib_model(...)`. Nothing worth salvaging; reject whole file.

### lexnlp/nlp/en/segments/title_locator.pickle  (batch 19, part 5/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 98
- **Summary:** Binary change re-adds an insecure pickle artifact. True net effect is `title_locator.skops (1612376 bytes) -> deleted` and `title_locator.pickle (0 -> 295494 bytes) added`; master has only `.skops` and no `.pickle` in this directory. Same SECURITY REGRESSION as the section segmenter: restores pickle arbitrary-code-on-load and drops the `skops` path.
- **Notes:** Pasted diff (`c6e7012..fe5d4bd`, pickle-vs-pickle) is merge-base only; use `git diff ec55727..pr30 -- lexnlp/nlp/en/segments/`. Reject together with the `load_joblib_model` caller change in `titles.py`.

### lexnlp/nlp/en/segments/titles.py  (batch 19, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 82
- **Summary:** Small training-helper improvement buried in loader/style regressions. Keep-worthy: new `TITLE_TRAINING_REQUEST_TIMEOUT = 60` constant and `_download_training_document()` helper adding `timeout=` + `raise_for_status()` to the training download, plus `all_file_lines.extend((row["File"], line) for line in ...)` loop-variable rename. Regressions: loader `load_bundled_model(...title_locator.pickle)` -> `load_joblib_model(...)` (bypasses `.skops`), `collections.abc.Generator` -> `typing.Generator`, quote churn, and `predict_proba(feature_data.to_numpy(dtype=float))` -> `predict_proba(feature_data.to_numpy())` which reintroduces the pandas-dtype noise master silenced.
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/nlp/en/segments/titles.py`. Note master already had `requests.get(file_url, timeout=60).text`, so the only real gain is the named constant + `raise_for_status()`; port just that onto master's loader/`to_numpy(dtype=float)` lines.

### lexnlp/nlp/en/segments/utils.py  (batch 19, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 86
- **Summary:** Adds genuinely useful schema-compat helpers (`TRAINED_LINE_WINDOW_PRE/POST = 3`, `has_compatible_line_window`, `resolve_model_feature_width`, `has_compatible_feature_width` with fail-closed forest/child width checks) consumed by the `sections.py` fix. Regression is the surrounding downgrade: modern `dict[str, int | float]` hints -> `Dict[str, Union[int, float]]` (and re-added `from typing import Dict, Union`), f-strings (`f"doc_char_{character}"`) -> `"...".format(...)`, plus mode change 755 -> 644.
- **Notes:** True diff `git diff ec55727..pr30 -- lexnlp/nlp/en/segments/utils.py` (+91 net on merge-base, +115/-vs-master with the typing reverts). Keep the four new helpers verbatim; restore master's `dict[str, ...]` annotations, f-strings, and minimal imports.

### lexnlp/nlp/en/tests/segmentation_quality.py  (batch 19, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 282-line hermetic test-helper module (`load_fixture`, `canonical_json_sha256`, span/PRF helpers, `deterministic_sentence_spans`, `retrieval_metrics`, `lexical_rank`) for the structure-first segmentation tests. Absent on master; no master code removed or downgraded. Explicitly documented as small-corpus regression/plumbing evidence, not population SOTA claims (`QUALITY_BLOB_RECONSTRUCTION_V3` marker + schema_version check).
- **Notes:** Fixture root `test_data/lexnlp/nlp/en/sota_segmentation` must be cherry-picked with it; verified `git cat-file -e ec55727:<path>` = MISSING, `pr30:<path>` = EXISTS. Minor nit: `Iterable/Mapping/Sequence` imported from `typing` rather than `collections.abc` (master style), not a blocker.

### lexnlp/nlp/en/tests/test_lossless_segment_core.py  (batch 19, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** First half (through `test_initial_all_caps_document_title...` setup) of a new 2366-line `unittest` suite pinning lossless hierarchy/chunk invariants: exact CRLF/unicode reconstruction, backend slice validation, section/clause/list/table nesting, structural-span AUGMENT/REPLACE digests, chunk overlap/provenance digests, token-counter budgets and scaling guards. Absent on master; purely additive, modern (`from __future__ import annotations`), no dependency pins or deprecated APIs touched.
- **Notes:** This block covers file part 1/2 only; imports `lexnlp.nlp.en.segments.chunks/hierarchy/payloads` so it must land with `chunks.py`, `hierarchy.py`, `backends.py`, `payloads.py` and the sota fixtures. No `-` lines exist (new file).

### lexnlp/nlp/en/tests/test_lossless_segment_core.py  (batch 19, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** Second half of the same new 2366-line suite: document-title scoping, parenthetical clauses, indented/bullet list offsets, boundary opt-out in char/token modes, arbitrary-counter search ordering and envelope failures (`TokenSearchLimitExceeded`), heading-fence/brute-force parity, table-vs-outline preemption, and overlap-structure planner matrix. Same assessment as part 9/10: additive coverage for the new segmentation stack, no master regression.
- **Notes:** File part 2/2; merge with part 9/10 as one file. Includes timing-sensitive scaling assertions (e.g. `< 6.0x` for 3x input) that may need triage on slow CI, but that is flakiness risk, not a regression — do not add skip markers per the skip-audit policy.

### lexnlp/nlp/en/tests/test_tokens.py  (batch 21, part 1/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** The pasted +4 chunk itself is additive: new `test_multilingual_wordnet_data_is_available` asserting `wordnet.synsets("chien", lang="fra")`. But the true `git diff ec55727..pr30` for this file is 64 lines of style revert (collapsed multi-line imports, single-quote reformatting, `treebank_pos_map` reflow) that undoes master's formatting. Cherry-pick only the new test onto master, keeping master's import/format style.
- **Notes:** Verified `git show ec55727:<path>` lacks the new test but has the modern formatted version; `git diff ec55727..pr30` shows the new test plus unrelated reflows. The new test needs OMW `fra` wordnet data — ensure bootstrap provides it or the test will fail on clean CI.

### lexnlp/tests/lexnlp_tests.py  (batch 21, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** The pasted datetime hunk looks like a `datetime.utcnow()` deprecation fix, but master already fixed it better: master has `from datetime import UTC, datetime` with `datetime.now(UTC).isoformat()` (tz-aware), while the PR uses `from datetime import datetime, timezone` with `datetime.now(timezone.utc).replace(tzinfo=None).isoformat()` (strips tz back to naive). The true `git diff ec55727..pr30` (473 lines) is a wholesale revert: `collections.abc.Callable` back to `typing.Callable`, `str | None` back to `str = None`, double quotes to single quotes, and detailed docstrings replaced with old ones.
- **Notes:** `UTC` requires Python >=3.11 and master requires `>=3.13,<3.15`, so master's form is correct; PR's `timezone.utc` + `replace(tzinfo=None)` downgrades both style and timestamp semantics (aware `+00:00` vs naive). Reject wholesale; nothing to cherry-pick.

### lexnlp/tests/tests/test_lexnlp_tests.py  (batch 21, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 65
- **Summary:** The pasted hunks pass `file_name=file_name` explicitly to `iter_test_data_text_and_tuple` and wrap `test_extraction_func_on_test_data` in a deterministic `memory_usage` mock (`patch.object(lexnlp_tests, "memory_usage", deterministic_memory_usage)`). Against master this bypasses master's PEP 709 fix, which deliberately uses an explicit `for` loop over `iter_test_data_text_and_tuple()` with no args so the helper's caller-offset logic is exercised. The mock makes the test hermetic but stops exercising the real `memory_usage` path, and the true diff also reverts whole-file double-quote formatting.
- **Notes:** `git show ec55727:<path>` has the `# Python 3.12 inlined list comprehensions (PEP 709)` loop; PR replaces it with `file_name=file_name`. If the mock is wanted, re-apply only the `patch.object` wrapper on top of master's loop version, keeping master style; do not take the `file_name=` bypass or quote reverts.

### lexnlp/tests/tests/test_models_repo.py  (batch 21, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Pasted diff removes one trailing blank line; true `git diff ec55727..pr30` is a single added blank line after `from __future__ import annotations`. No code, dependency, or API change in either direction. Whitespace-only, safe to take.
- **Notes:** No version pins or symbols involved.

### lexnlp/tests/typed_annotations_tests.py  (batch 21, part 5/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** The pasted hunk is a genuine improvement: `except:  # pylint:disable=bare-except` narrowed to `except Exception` (cast block) and `except ValueError` (two `strptime` blocks), which is stricter than master's still-bare excepts. But the true `git diff ec55727..pr30` (252 lines) wraps that fix in a full style/typing revert: double to single quotes, `list[str] | None` back to `List[str] = None`, `tuple[bool, bool]` back to `Tuple[bool, bool]`, `type` back to `Type`, `collections.abc.Callable` removal, and deleted `__init__` docstring.
- **Notes:** Cherry-pick only the three `except` narrowings onto `ec55727` (`except Exception` for the cast, `except ValueError` for date/datetime parses); discard all quote/typing/docstring reverts. Verified via `git diff ec55727..pr30 -- <path>`.

### lexnlp/tests/values_comparer.py  (batch 21, part 6/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 95
- **Summary:** Single-line change `except:  # pylint:disable=bare-except` to `except Exception` in `values_look_equal`. True `git diff ec55727..pr30` confirms this is the only line changed. It narrows bare-except without altering behavior for ordinary failures and matches the modernisation direction (no `KeyboardInterrupt`/`SystemExit` swallowing).
- **Notes:** No other lines, pins, or APIs touched.

### lexnlp/utils/amount_delimiting.py  (batch 21, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 70
- **Summary:** The PR replaces per-call `locale.localeconv()` via `LocaleContextManager` with a hardcoded `NUMERIC_CONVENTIONS` dict for `de_de`/`en_us` (`decimal_point`/`thousands_sep`/`grouping [3,3,0]`) and only consults the OS locale for other locales. Functionally this overlaps master's existing fix: master (`git show ec55727:<path>`) already enforces canonical `de_de` (`,`,`.`) and `en_us` (`.`,`,`) fallbacks plus a generic `if not grouping: grouping = [3, 3, 0]` guard. The PR adds no new locale coverage and introduces a semantic delta (`[v for v in grouping if v > 0] or [3]` drops the `0` repeat sentinel vs master's `[3, 3, 0]`), wrapped in a whole-file revert (`FrozenSet`/`Dict`/`Optional`/`List`/`Set` back to old typing, double to single quotes).
- **Notes:** The hardcoded-dict-to-avoid-missing-locale-packs idea is sound but must be re-applied on master's typing/style with `[3, 3, 0]` semantics preserved if wanted. Reject the file wholesale; true diff is 146 lines, mostly style.

### lexnlp/utils/decorators.py  (batch 21, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** The `safe_failure` rewrite is a real bug fix over master: master (`git show ec55727:<path>`) still has the old `import types` + `isinstance(res, types.GeneratorType)` version that calls `func` twice and uses bare `except:`, with no `functools.wraps`. The PR correctly branches on `inspect.isgeneratorfunction`, adds `@wraps`, catches only `Exception` (letting `KeyboardInterrupt`/`SystemExit` through), and avoids the double call. The cost is a minor import downgrade (`from collections.abc import Callable` back to `from typing import Any, Callable`) plus docstring shortening.
- **Notes:** Merge the two-wrapper logic but restore `from collections.abc import Callable` (correct on `requires-python >=3.13,<3.15`) and master's fuller docstring/double-quote style. Removes `# pylint: disable=bare-except` header, which is good.

### lexnlp/utils/lines_processing/parsed_text_quality_estimator.py  (batch 21, part 9/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 90
- **Summary:** True `git diff ec55727..pr30` (69 lines) is a pure modernisation revert with no functional gain: master's documented `wrap_line(line_or_phrase: LineOrPhrase)` with docstrings, `ClassVar[set[str]]` annotations, and double quotes are replaced by `wrap_line(l: LineOrPhrase)  # noqa: E741`, single quotes, dropped `__init__`/`__repr__`/`split_text_on_lines` docstrings, and `range(len(self.lines))` inflated to `range(0, len(self.lines))`. The pasted `l` to `typed_line` rename does not compensate for reintroducing the `E741` ambiguous-name violation behind a `noqa`.
- **Notes:** Verified `git show ec55727:<path>` has the typed, documented version. Reject wholesale.

### lexnlp/utils/parse_df.py  (batch 21, part 10/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** The functional core is an improvement over master and worth porting: `SEARCH_PTN` changes from consuming `r'(?:^|\W)({})(?:\W|$)'` to lookaround `r'(?<!\w)({})(?!\w)'` so adjacent entities like `Alpha,Beta` are both found, spans switch from `match.span()` to `match.span(1)` for exact entity offsets, per-match `str.contains(r'(?:^|;){matched_str}(?:$|;)')` (unescaped regex injection + O(n) scan) is replaced by a `_row_positions_by_column` index with `.iloc`, alternatives are longest-first deterministically ordered, and `_split_cell_value`/`_compile_atomic_collection_ptn` handle `None`/empty/`separator=None` cleanly. The wrapper is the problem: true `git diff ec55727..pr30` (240 lines) reverts modern `list[str] | tuple[str, ...]`, `dict | None`, `Generator[dict]` hints to `Union`/`List`/`Tuple`, single-quotes the file, and strips the detailed `get_entities`/`get_entity_list` docstrings.
- **Notes:** Port only the `SEARCH_PTN`, `_split_cell_value`, `_compile_atomic_collection_ptn` longest-first ordering, `_row_positions_by_column` lookup, and `match.group(1)`/`span(1)` changes onto master's signatures, `Generator[dict]` hints, and full docstrings. Behavior change to verify with tests: longest-first ordering and lookaround boundaries alter overlapping-entity resolution vs master.

## Batch 23

### python-requirements-full.txt  (batch 23, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 99
- **Summary:** The pasted diff deletes this 44-line legacy snapshot (beautifulsoup4==4.11.1, numpy==1.22.3, scikit-learn==0.23.1, spacy-models 2.3.1 URL, etc.), but `git show ec55727:python-requirements-full.txt` confirms it is already gone on master, and `git diff ec55727..pr30` shows no net change. The deletion is a no-op, not a regression.
- **Notes:** Merge-base copy carried our own "DEPRECATED: use pyproject.toml + uv.lock" header; nothing to preserve.

### python-requirements-notes.txt  (batch 23, part 2/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 99
- **Summary:** Same situation as the other legacy requirement files: the 4-line historical note (dateparser/pandas/docutils pin justifications) is already absent on master (`git show ec55727:...` fails), so the PR's deletion has zero net effect. No modernisation work is undone.
- **Notes:** None; no-op.

### python-requirements.txt  (batch 23, part 3/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 99
- **Summary:** The 86-line snapshot (Sphinx==5.3.0, pandas==1.5.1, scikit-learn==0.24.0, pipenv==2022.11.11, etc., with our DEPRECATED header) is already deleted on master, and the true diff `ec55727..pr30` for this path is empty. Accepting the deletion changes nothing.
- **Notes:** None; no-op. Do not resurrect these pins anywhere.

### readthedocs.yml  (batch 23, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** The true diff deletes the stale v2 config (Python 3.8, `requirements: python-requirements.txt` pointing at a file master already deleted). This is half of a rename modernisation: PR head adds `.readthedocs.yaml` (verified via `git ls-tree -r pr30`) with `os: ubuntu-24.04`, `python: "3.12"`, `method: uv / command: sync / extras: [docs]`, and `fail_on_warning: true`. Deleting the broken old file is correct.
- **Notes:** Cherry-pick together with `.readthedocs.yaml`; consider bumping its `python: "3.12"` to `"3.13"` to match requires-python `>=3.13,<3.15` on master.

### scripts/_artifact_transaction.py  (batch 23, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** New 142-line helper providing rollback-capable staged publication (`publish_staged_files`): hard-link/copy backups, duplicate-destination guard, symlink refusal, per-file publish via `atomic_output_path`, and rollback with retained-backup reporting. It is well-engineered new code, not a revert of anything on master (file is absent at ec55727).
- **Notes:** Imports `atomic_output_path` from `lexnlp.ml.artifact_io`, which exists only on pr30 and is MISSING on master — a standalone cherry-pick ImportErrors. Co-merge with `lexnlp/ml/artifact_io.py` from the same PR (or repoint at master's equivalent if one exists).

### scripts/asset_drift_check.py  (batch 23, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** The functional change migrates downloads to manifest-pinned APIs (`get_exact_path_from_catalog`, `download_github_release_to_path(tag, manifest_path=...)`), which matches the PR's secure-catalog design, plus a case-insensitive SHA compare (`actual_sha.lower() != expected_sha`). Against master this is mixed: the new APIs do not exist in master's catalog (only `get_path_from_catalog`), and the patch reverts modern typing (`dict[str, Any]`/`collections.abc` back to `Dict`/`typing.Iterable`) and drops the exception class name from failure messages (`({exc.__class__.__name__}: {exc})` to `({exc})`).
- **Notes:** Co-merge with the PR catalog modules providing `get_exact_path_from_catalog`/`download_github_release_to_path`; restore `dict[str, Any]`, `collections.abc` imports, and the `{exc.__class__.__name__}: {exc}` failure detail.

### scripts/bootstrap_assets.py  (batch 23, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 75
- **Summary:** Substantively this is a large security hardening master lacks (verified: master is 514 lines with no sha/verify logic, Tika 1.16, `nlp.stanford.edu` URLs). The PR (1244 lines) adds pinned NLTK revision `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` with SHA-256/sizes, Stanford SHA-512 pins with `downloads.cs.stanford.edu` allowlist, safe single-root ZIP extraction with rollback, Tika 3.3.2 (`tika-app-3.3.2.jar`/`tika-server-standard-3.3.2.jar`) with SHA-512, HTTPS host allowlisting, and byte caps. The regressive part is stylistic: `typing.Tuple/List/Iterable` reverts master's builtin generics and `collections.abc` imports.
- **Notes:** Restore modern type syntax (`tuple[...]`, `list[...]`, `collections.abc`). Independently verify the hardcoded NLTK/Stanford/Tika digests and sizes before trusting; confirm the Tika 3.x `tika-server` to `tika-server-standard` rename and Java version requirement fit CI. The `pickle.dump` in `reexport_contract_model_from_legacy` matches master's existing pickle usage in this script and is not the `.skops` regression site.

### scripts/contract_type_quality_gate.py  (batch 23, part 8/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 80
- **Summary:** The PR adds genuinely new quality enforcement absent on master (verified zero `holdout`/`fixture_sha256` hits at ec55727): `duplicate_group_holdout` evidence comparison with `HOLDOUT_SPLIT_IDENTITY_KEYS`, `verify_fixture_sha256`, `schema_version: 2` baselines, and holdout regression thresholds. But it regresses the model loader: master uses skops-aware `lexnlp.ml.model_io.load_model` (docstring notes legacy pickle/cloudpickle plus `.skops`), while the PR reverts to pickle-only `lexnlp.utils.unpickler.load_sklearn_model`, which cannot load master's `.skops` artifacts. It also reverts typing to `Dict/List/Tuple` and drops `zip(..., strict=True)`.
- **Notes:** Keep the holdout/fixture-sha features; restore `load_model`, builtin generics, `collections.abc.Sequence`, and `strict=True` before merging.

### scripts/create_release_branch.sh  (batch 23, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 88
- **Summary:** Complete rewrite of an obsolete two-repo script (references to `-core` checkout, `setup.py`, `Pipfile*`, GNU `readlink -f`, `py3clean`, rsync, interactive select with commented-out git) into a safe single-repo tool: PEP 440 version validation, clean-tree and branch-existence guards, version bump in `pyproject.toml` plus `lexnlp/__init__.py` (`__version__` and versioned `__license__` URL), then `uv lock` plus `uv lock --locked` with no fetch/commit/push. Pure modernisation, no master work lost.
- **Notes:** Requires `python3` and `uv`; fails unless `pyproject.toml`, `__version__`, and `__license__` versions already agree.

### scripts/download_tika.sh  (batch 23, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** Replaces the insecure legacy downloader (Tika 1.16 over `http://www-us.apache.org` via wget, GNU `mkdir --parents`, unquoted `$LEXNLP_USE_TIKA`) with a `set -euo pipefail` wrapper delegating to `bootstrap_assets.py --tika` (hash-pinned Tika 3.3.2 in this PR). Direction matches the AGENTS.md Tika migration notes.
- **Notes:** Depends on the hardened `scripts/bootstrap_assets.py` from this PR; requires `python3` and network access at run time.


### scripts/tests/test_publish_workflows.py  (batch 25, part 1/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 78
- **Summary:** New 62-line test locks in the PR's split build/publish privilege boundary (single `contents: write` confined to `publish`, `persist-credentials: false` in build, no checkout in publish, trusted-manifest outputs, no `--clobber`). This is a security improvement over master HEAD (ec55727), which has a single `publish` job with top-level `contents: write`. The test itself introduces no downgrade.
- **Notes:** Co-dependent: fails against master's single-job workflows until the workflow files are cherry-picked too. Test does not pin `PYTHON_VERSION`; the companion workflow change regresses it `3.13` (master) to `3.12.13` (PR) — fix the version on merge, not the test.

### scripts/tests/test_reexport_bundled_sklearn_models.py  (batch 25, part 2/10)
- **Regression:** YES
- **Verdict:** REJECT
- **Confidence:** 95
- **Summary:** True diff (`ec55727..pr30`) wholly replaces master's skops-era test with a pickle-only test for the PR's rewritten `scripts/reexport_bundled_sklearn_models.py`. Master script defaults to `--format skops` via `skops.io` (`skops>=0.11` in pyproject); PR drops `skops` entirely and the test exercises only `joblib.dump`/pickle plus PR-only `scripts/_artifact_transaction.publish_staged_files`. Merging deletes coverage for `load_model` exception narrowing and layered `.skops.zip` handling.
- **Notes:** `git show ec55727:scripts/tests/test_reexport_bundled_sklearn_models.py` covers `(pickle.UnpicklingError, ValueError, KeyError)` fallback and `try/finally` tmp cleanup — all deleted by this file. PR pyproject has no `skops` pin (`cloudpickle>=2.2.0,<4` only) vs master `skops>=0.11` + `cloudpickle>=3.0`.

### scripts/tests/test_reexport_contract_model.py  (batch 25, part 3/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** New 283-line test (missing on master) adds genuinely good safety coverage: warning/validation/quality-gate failures must preserve existing model+metadata bytes and modes, success gates a staged candidate with isolated `NLTK_DATA`, and metadata output may not alias a model path. Direction is good, but it is written against PR-only APIs (`lexnlp.ml.artifact_abi`, `scripts/_artifact_transaction`, `unpickler.load_sklearn_model`), neither of which exists on master (master uses `lexnlp/ml/model_io.py` + `cloudpickle.load`).
- **Notes:** Cannot merge standalone — imports fail on master. Port the assertions (staged gating, mode preservation, alias guard) onto master's `model_io`/`skops` stack instead of the PR's pickle-transaction stack before merging.

### scripts/tests/test_segmentation_benchmark.py  (batch 25, part 4/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 78
- **Summary:** New 181-line hermetic benchmark test for PR-only `scripts/segmentation_benchmark.py` (`run_benchmark`, `run_statute_scaling_benchmark`, scaling classifier, CLI aliases). It uses no model, corpus, or network and explicitly freezes determinism/losslessness and the linear-vs-quadratic growth classifier. Nothing on master is reverted — `scripts/segmentation_benchmark.py` is missing on master.
- **Notes:** Cherry-pick only with its implementation (`scripts/segmentation_benchmark.py`, `lexnlp/nlp/en/segments/chunks.py`, `hierarchy.py`, `lexnlp/nlp/en/tests/segmentation_quality.py`); the test alone is inert. No pickle/skops involvement.

### scripts/tests/test_segmentation_quality_gate.py  (batch 25, part 5/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 212-line deterministic regression gate for PR-only `scripts/segmentation_quality_gate.py`: frozen MRR/recall/precision numbers, exact per-kind complete-boundary P/R/F1, external-ranking digest binding (`retrieval_manifest_sha256`, `candidate_configuration_sha256`, `candidate_set_sha256`), and CLI prepare-external behavior. Hermetic and additive; master has no equivalent file to regress.
- **Notes:** Asserts `configuration max_chars==180`, `overlap_chars==24`, `anchors 34/34`, `mrr≈0.8333`, 7 `*_sha256` digests. Must land with the gate script plus `test_data/lexnlp/nlp/en/sota_segmentation/*` fixtures in this same batch.

### scripts/train_contract_model.py  (batch 25, part 6/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 85
- **Summary:** True `ec55727..pr30` diff mixes a good centralization (`load_sklearn_model`, `restore_legacy_model_state`, `atomic_pickle_dump`, `artifact_metadata`/`assert_model_artifact_runtime`) with clear modernisation reverts and observability loss. Typing regresses `tuple[str, ...]`/`collections.abc` (master) to `Tuple`/`Dict`/`List` from `typing`; `subprocess.run(cmd, check=True, capture_output=True, text=True)` becomes bare `check=True`; the `except CalledProcessError as exc` handler loses `returncode`/`stdout`/`stderr` capture and the `_decode` helper.
- **Notes:** Keep the atomic-dump/ABI-metadata helpers; restore master's `collections.abc Sequence/Mapping/Iterable`, builtin generics, and the captured-output quality-gate report fields. Pasted three-dot diff hid that master already uses public `pipeline.steps[-1][1]` — PR does not reintroduce `_final_estimator` here.

### scripts/train_contract_type_model.py  (batch 25, part 7/10)
- **Regression:** PARTIAL
- **Verdict:** MERGE WITH EDITS
- **Confidence:** 70
- **Summary:** Adds a real methodological improvement — deterministic duplicate-group holdout (`DUPLICATE_GROUP_HOLDOUT_SEED="lexnlp-global-group-holdout-v1"`, `normalized_text_group`, `split_assignment_sha256`, leakage report) plus top-3 scoring (`accuracy_top1/accuracy_top3`, `zero_division=0`), `TRAINING_RECIPE`, `--max-features`, and `artifact_metadata` — matching the publish-gate holdout arguments. But it silently changes defaults (`--max-docs-per-label` 120→0 full corpus; TF-IDF `max_features` 120000 hardcoded in master's `runtime_model.py` → 75000), reverts typing to `typing.Dict/List/Tuple`, and orphans master's `scripts/tests/test_train_contract_type_model.py` stratification-fallback coverage (deleted in the PR's test list).
- **Notes:** Keep holdout/top-n/recipe; justify or revert the 120→0 and 120000→75000 default changes, restore `collections.abc` typing, add `numpy` import cost note, and retain master's stratify-fallback test.

### test_data/lexnlp/nlp/en/sota_segmentation/boundary_gold.json  (batch 25, part 8/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 82
- **Summary:** New 126-line hermetic fixture (`schema_version 1`, `QUALITY_BLOB_RECONSTRUCTION_V3`) with three exact-span cases (legal abbreviations, section/clause/list, unicode/CRLF/unterminated line) using half-open char spans with zero tolerance. Small, auditable, dependency-free; missing on master so nothing is reverted.
- **Notes:** Must land with `scripts/segmentation_quality_gate.py` and its test; description explicitly states it is regression gold, "not population evidence" — do not cite as SOTA proof.

### test_data/lexnlp/nlp/en/sota_segmentation/legacy_parity.json  (batch 25, part 9/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 85
- **Summary:** New 32-line frozen parity fixture pinning legacy public segmenter shapes (e.g. `"The U.S."` / `"Borrower is Acme."` split, blank-line paragraph shape `"First operative paragraph.\n\n"`). Additive and inert data; no master file is touched.
- **Notes:** Cherry-pick with the segmentation implementation; verifies the additive-API parity claim only.

### test_data/lexnlp/nlp/en/sota_segmentation/legal_edge_cases.json  (batch 25, part 10/10)
- **Regression:** NO
- **Verdict:** MERGE
- **Confidence:** 80
- **Summary:** New 524-line hermetic legal regression corpus (7 cases: mixed newlines/unicode, nested clauses/lists, schedule/part, heading scopes with `expected_ancestry`, abbreviations/citations, numbered-obligations-not-sections, statute profile) with `gold_segments`/`gold_structure` anchors. Manually auditable and explicitly "not population-level SOTA evidence"; absent on master, so purely additive.
- **Notes:** Pair with the quality-gate script/test; anchor-based (not offset-based) so reflow-tolerant. No dependency, pickle, or locale-code impact.

### uv.lock  (batch 27, part 2/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This chunk bumps three pins forward against master HEAD (verified via `git show ec55727:uv.lock`): charset-normalizer 3.4.4→3.4.9, click 8.3.1→8.4.2, coverage 7.13.4→7.15.2, all genuine upstream releases with June–July 2026 upload-times, not downgrades. But the lock was regenerated under the PR's downgraded `requires-python >=3.10,<3.14` instead of master's `>=3.13,<3.15`, so the resolved wheel set drops all cp314 artifacts and re-adds cp310/cp311. Merging this hunk would break Python 3.14 support on master; any wanted version bumps must come from a fresh `uv lock` on master, not by cherry-picking this hunk.
- **Notes:** Master has 16x `charset_normalizer-3.4.4-cp314` and 30x `coverage-7.13.4-cp314` entries; PR has 0x `cp314` for 3.4.9/7.15.2 and adds 13x `charset_normalizer-3.4.9-cp310`. Lock header `requires-python = ">=3.10, <3.14"` (PR) vs `">=3.13, <3.15"` (master). True net diff `ec55727..pr30 -- uv.lock` is 1575 insertions/1381 deletions — do not cherry-pick lockfile hunks.

### uv.lock  (batch 29, part 5/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 87
- **Summary:** This chunk covers the lock header/extras plus the `lexpredict-lexnlp` `requires-dist` specifier block and the lxml 5.4.0→6.1.1 version swap. Several specifier moves are forward vs master HEAD (verified via `git show ec55727:uv.lock` / `pr30:uv.lock`): `lxml>=6.1.1,<7`, `nltk>=3.10.0,<4`, `regex>=2026.7.19,<2027`, `elasticsearch>=8.17.0,<10`, `dateparser>=1.2.2,<2`. But it carries hard regressions: lock header `requires-python` goes `>=3.13,<3.15` (master) to `>=3.10,<3.14` (PR), `scikit-learn==1.7.2` pins below master's resolved 1.8.0, `numpy>=1.26.4,<3` re-allows numpy 1.x against master's `>=2.3,<3` model ABI floor, `pandas>=2.2.0,<4` allows pandas 3.x that master caps at `<3`, and `skops` drops to 0 refs (master has 3) with no `skops>=0.11` specifier. The pasted lxml `5.4.0→6.1.1` is merge-base artefact; true net vs master is `6.1.0→6.1.1`.
- **Notes:** Additive extras (`ruff`, `docs`, `audit`, `sphinx-rtd-theme`, `memory-profiler` in test, `license-expression 30.4.4` via pip-audit) are fine in isolation. Do not cherry-pick lockfile hunks: true net diff `ec55727..pr30 -- uv.lock` is 1575 insertions/1381 deletions; re-apply wanted bumps with a fresh `uv lock` on master (`cp314` refs: master 392, PR 0; `cp310` refs: master 0, PR 144).

### uv.lock  (batch 29, part 6/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 86
- **Summary:** This chunk is the lxml 6.1.1 wheel list plus `markdown-it-py 4.0.0→4.2.0`, `more-itertools 10.8.0→11.1.0`, `nh3 0.3.2→0.3.6`, `nltk 3.8.1→3.10.0` (true net vs master: `3.9.4→3.10.0` plus new `defusedxml` dep), new `msgpack 1.2.1`, and extra `markupsafe 3.0.3` cp313/cp313t wheels. Every version move is forward upstream (lxml, markdown-it-py, more-itertools, nh3 upload-times May–June 2026), not a downgrade. The regression is structural, not versional: the whole resolution was produced under `requires-python >=3.10,<3.14`, so the wheel set adds cp310/cp311/cp313 artefacts and drops all cp314 artefacts (master 392x `cp314`, PR 0x; master 0x `cp310`, PR 144x), breaking master's Python 3.14 support.
- **Notes:** Wanted bumps (markdown-it-py 4.2.0, more-itertools 11.1.0, nh3 0.3.6, nltk 3.10.0, lxml 6.1.1) should be re-resolved via `uv lock` on master, not merged as hunks. `msgpack 1.2.1` is new in PR lock while `skops` (its usual consumer here) is absent — confirm the requiring path before re-locking.
### uv.lock  (batch 31, part 9/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 87
- **Summary:** This chunk bumps six pins forward vs master HEAD (verified via `git show ec55727:uv.lock` / `pr30:uv.lock`): pylint 4.0.4→4.0.6, pytest 9.0.3→9.1.1 (pasted 9.0.2 is merge-base), pytest-cov 7.0.0→7.1.0, pytz 2025.2→2026.3.post1, readme-renderer 44.0→45.0, regex 2025.11.3→2026.7.19 (pasted 2022.3.2 is merge-base artefact), plus new pyparsing 3.3.2 (0 refs on master, 4 on PR via `pip-requirements-parser 32.0.1`). Every version move is a genuine newer upstream release, not a downgrade. The regression is structural: the whole lock was regenerated under `requires-python >=3.10,<3.14` instead of master's `>=3.13,<3.15`, so the wheel set drops all cp314 artefacts and re-adds cp310/cp311; do not cherry-pick this hunk, re-apply wanted bumps with a fresh `uv lock` on master.
- **Notes:** Lock header `requires-python = ">=3.10, <3.14"` (PR) vs `">=3.13, <3.15"` (master); true net diff `ec55727..pr30 -- uv.lock` is 1575 insertions/1381 deletions (master 392x `cp314`, PR 0x). Upload-times confirm forward moves: pylint 4.0.6 (2026-06-14), pytest 9.1.1 (2026-06-19), regex 2026.7.19 (2026-07-19).

### uv.lock  (batch 31, part 10/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 88
- **Summary:** This chunk mixes forward bumps with hard downgrades vs master HEAD (verified via `git show ec55727:uv.lock` / `pr30:uv.lock`): forward are reporters-db 3.2.63→3.2.66, requests 2.33.1→2.34.2 (pasted 2.32.5 is merge-base), rich 14.3.2→15.0.0, ruff 0.15.11→0.16.0 (pasted as new, but master already has ruff). Regressions are scikit-learn 1.8.0→1.7.2 (pasted 1.2.2→1.7.2 is merge-base artefact; true net is a downgrade below master's model ABI) and the scipy split reintroducing 1.15.3 for `python_full_version < '3.11'` alongside 1.17.1/1.18.0 (master is single 1.17.0). As with all lock hunks, the resolution base regresses `requires-python` to `>=3.10,<3.14` with 0x `cp314` (master 392x) and drops `skops` to 0 refs (master 5 refs); regenerate via `uv lock` on master instead of merging.
- **Notes:** `skops` refs: master 5, PR 0 (no `skops>=0.11` specifier); `scipy 1.15.3` wheels are cp310–cp312-only legacy artefacts re-added for the downgraded floor; wanted bumps (reporters-db 3.2.66, requests 2.34.2, rich 15.0.0, ruff 0.16.0, scipy 1.17.1/1.18.0) should be re-resolved on master's `>=3.13,<3.15` base.

### uv.lock  (batch 33, part 13/13)
- **Regression:** PARTIAL
- **Verdict:** REJECT
- **Confidence:** 87
- **Summary:** This tail chunk bumps pins forward vs master HEAD (verified via `git show ec55727:uv.lock` / `git show pr30:uv.lock`): twine 6.2.0→7.0.0, typing-extensions 4.15.0→4.16.0, tzdata 2025.3→2026.3, tzlocal 5.3.1→5.4.4, tqdm 4.67.3→4.70.0, us 2.0.2→3.2.0, wrapt 2.1.1→2.3.0, plus new zipp 4.1.0 via importlib-metadata 9.0.0. The pasted urllib3 2.6.3→2.7.0 is a merge-base artefact; true net vs master is 2.7.0→2.7.0 (no-op). Every version move is a genuine newer upstream release, not a downgrade, but the lock was regenerated under the PR's downgraded `requires-python >=3.10,<3.14` instead of master's `>=3.13,<3.15`, so the wheel set drops all cp314 artefacts and re-adds cp310/cp311; do not cherry-pick this hunk, re-apply wanted bumps with a fresh `uv lock` on master.
- **Notes:** Lock header `requires-python = ">=3.10, <3.14"` (PR) vs `">=3.13, <3.15"` (master); true net diff `ec55727..pr30 -- uv.lock` is 1575 insertions/1381 deletions (master 392x `cp314`, PR 0x). Upload-times confirm forward moves: twine 7.0.0 (2026-07-27), typing-extensions 4.16.0 (2026-07-02), tzdata 2026.3 (2026-07-10), tzlocal 5.4.4 (2026-06-29), wrapt 2.3.0 (2026-07-28). `skops` refs: master 3, PR 0. New `zipp 4.1.0` + `importlib-metadata 9.0.0` exist only to backfill `importlib.metadata` for the regressed py310 floor; master's `>=3.13` needs neither. `us` major bump 2.0.2→3.2.0 matches PR pyproject `us>=3.2.0,<4` vs master `us>=2.0.2`.
