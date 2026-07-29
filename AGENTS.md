# AGENTS.md

This is the quick-start and validation policy for coding agents working on
LexNLP.

## Project baseline

- Package: `lexnlp/`
- Packaging and dependency source of truth: `pyproject.toml` + `uv.lock`
- Supported Python: 3.10–3.13 (`>=3.10,<3.14`)
- Default development/docs interpreter: Python 3.12
- CI: `.github/workflows/`
- Sphinx source: `documentation/docs/source/`

Do not reintroduce Pipenv, split `python-requirements*.txt` snapshots, Travis
CI, or the deprecated `readthedocs.yml` filename.

## Repository map

```text
lexnlp/                     Runtime package
  config/                   Locale configuration
  extract/                  Extraction modules (common, en, de, es, ml)
  ml/                       Model utilities and verified release catalog
  nlp/                      NLP components and training helpers
  tests/                    Shared test infrastructure
  utils/                    Runtime utilities
test_data/                  Fixtures and fixed quality-gate baselines
scripts/                    Asset, model, release, and validation tooling
ci/                         Distribution and test-policy checks
constraints/                Reproducible model-producer ABI
documentation/docs/source/  Sphinx documentation
```

## Reproducible setup

Run from a clone of this repository; commands must not depend on a personal
filesystem path.

```bash
uv python install 3.12
uv sync --frozen --python 3.12 --extra dev --extra test
```

`uv sync` creates `.venv` and installs the project editable. Do not hand-edit
`uv.lock`; update it with `uv lock` after an intentional `pyproject.toml`
change.

## Assets and external services

These resource classes are intentionally separate:

- Bundled sklearn/joblib models are distribution files and require no
  bootstrap.
- NLTK corpora are external data:

  ```bash
  .venv/bin/python scripts/bootstrap_assets.py --nltk
  ```

- Pipeline classifiers are derived from trusted sources. Is-contract uses the
  pinned `0.1` release as its source, downloads `0.2` when published, and
  otherwise creates a local `0.2` re-export. Contract-type builds or reuses
  its runtime artifact from a pinned corpus:

  ```bash
  .venv/bin/python scripts/bootstrap_assets.py \
    --contract-model \
    --contract-type-model
  ```

- Stanford NLP is optional, Java-dependent, and test-gated:

  ```bash
  .venv/bin/python scripts/bootstrap_assets.py --stanford
  ```

- Apache Tika 3.3.2 is an optional out-of-process Java service. The secure
  bootstrap installs the pinned app and server-standard jars; the launcher
  binds to loopback:

  ```bash
  .venv/bin/python scripts/bootstrap_assets.py --tika
  LEXNLP_USE_TIKA=true scripts/run_tika.sh
  ```

Never replace the verified Stanford, Tika, model, or corpus bootstrap with an
unpinned download or commit downloaded third-party asset trees. NLTK resources
come from the official `nltk_data` repository, use a separate pinned SHA-256
catalog (including `omw-1.4` and `omw-2.0`), and are not covered by the
model/corpus release-asset manifest.

## Model compatibility and quality policy

Persisted models are executable pickle/joblib data. Load only artifacts from a
trusted release or producer.

Models must be serialized with the exact oldest supported producer stack in
`constraints/model-artifact-abi.txt`:

- joblib 1.5.0
- NumPy 1.26.4
- pandas 2.2.0
- scikit-learn 1.7.2
- SciPy 1.13.0

Every produced artifact must then direct-load and pass the same fixed-fixture
quality gate with the latest locked dependency stack. Never build a release
artifact only under the latest NumPy ABI.

The quality policy is strict non-regression: every maximum regression argument
remains `0.0`. Do not refresh baseline metrics merely to make a weaker model
pass.

```bash
.venv/bin/python scripts/reexport_bundled_sklearn_models.py --check-current

.venv/bin/python scripts/model_quality_gate.py \
  --baseline-tag pipeline/is-contract/0.1 \
  --candidate-tag pipeline/is-contract/0.2 \
  --baseline-metrics-json test_data/model_quality/is_contract_baseline_metrics.json \
  --max-accuracy-regression 0.0 \
  --max-f1-regression 0.0

.venv/bin/python scripts/contract_type_quality_gate.py \
  --baseline-tag pipeline/contract-type/0.2-runtime \
  --candidate-tag pipeline/contract-type/0.2-runtime \
  --baseline-metrics-json test_data/model_quality/contract_type_baseline_metrics.json \
  --max-accuracy-top1-regression 0.0 \
  --max-accuracy-topn-regression 0.0 \
  --max-f1-macro-regression 0.0 \
  --max-f1-weighted-regression 0.0
```

## Test-integrity policy

- Do not add, remove, or alter `skip`, `skipif`, or `xfail` to conceal a
  failure.
- Required suites must pass completely.
- A genuinely necessary marker requires an inline
  `skip-audit: issue=<link-or-id> expires=YYYY-MM-DD` annotation.
- `ci/skip_audit_allowlist.txt` is reserved for cases that cannot be annotated.

## Validation

Run targeted tests while iterating, then the required checks:

```bash
.venv/bin/ruff check lexnlp scripts ci --select E4,E7,E9,F
.venv/bin/python ci/skip_audit.py
.venv/bin/python scripts/reexport_bundled_sklearn_models.py --check-current
.venv/bin/pytest lexnlp
```

Run the optional Stanford suite after installing its verified assets and Java:

```bash
LEXNLP_USE_STANFORD=true .venv/bin/pytest \
  lexnlp/nlp/en/tests/test_stanford.py \
  lexnlp/extract/en/entities/tests/test_stanford_ner.py
```

Validate distributions in clean environments:

```bash
uv build
.venv/bin/python ci/check_dist_contents.py
```

Build documentation with warnings fatal:

```bash
.venv/bin/sphinx-build -W --keep-going \
  -b html documentation/docs/source documentation/docs/build/html
```

## Implementation and PR checklist

- Preserve public signatures, return shapes, extraction quality, and
  performance unless the change is a measured improvement.
- Keep locale-specific work in its locale/module and add fixed fixtures for
  behavior changes.
- Prefer existing utilities over parallel implementations.
- Verify the supported Python matrix when changing dependencies or
  serialization.
- Verify wheel and sdist contents and install both outside the source tree.
- Record every required asset and quality/performance result in the PR.
- Never include credentials, unverified downloads, or generated third-party
  asset directories.

See `MIGRATION_RUNBOOK.md` for operational detail.
