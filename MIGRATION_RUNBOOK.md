# Dependency and Model Migration Runbook

This is the operational guide for reproducible LexNLP development, dependency
updates, persisted-model production, and release validation.

## 1. Supported toolchain

- Python: 3.10–3.13 (`>=3.10,<3.14`)
- Default validation/docs interpreter: Python 3.12
- Dependency metadata: `pyproject.toml`
- Reproducible lock: `uv.lock`
- Installer and runner: `uv`
- CI: GitHub Actions

Pipenv manifests, split `python-requirements*.txt` snapshots, and Travis CI
were removed because they described incompatible dependency graphs and
unsupported Python versions. Do not recreate them.

The supported range is deliberately `3.10` through `3.13`:

- Python 3.10 is the oldest supported interpreter and scikit-learn 1.7.2 is
  the newest scikit-learn line that still supports it. Keeping that boundary
  avoids dropping a still-maintained interpreter solely for a dependency
  upgrade.
- Python 3.11, 3.12, and 3.13 are tested directly. Linux runs the full suite on
  all four interpreters; macOS and Windows smoke-test the 3.10 and 3.13
  endpoints, including compiled imports and a package build.
- Python 3.9 and older are intentionally retired. They are outside upstream
  security support and prevent adoption of the current numerical and packaging
  stack. This supersedes LexNLP's historical Python 3.6 baseline.
- Python 3.14 is intentionally excluded for now. Gensim 4.4.0, which backs the
  public Doc2Vec vectorizer and legacy contract-type detector APIs, has no
  CPython 3.14 wheels and its extension source fails against removed
  CPython/NumPy internals. Removing or optionalizing Gensim would reduce public
  functionality. Revisit the upper bound when Gensim publishes compatible
  wheels, then validate the full matrix before widening it.

Gensim's Linux wheels target `manylinux_2_28`; Linux systems older than that
glibc baseline may fall back to an unsupported source build. Use a current
manylinux-compatible distribution or container on those hosts.

Published model bytes use the exact numerical stack in
`constraints/model-artifact-abi.txt` under Python 3.12.13. Contract-type
training is single-threaded, and the publish workflow fixes the common BLAS
thread controls to `1`, because parallel reduction order can otherwise change
serialized floating-point coefficients by a few ULPs. The workflow performs a
second clean build and requires the two artifacts to be byte-identical before
publishing. Those bytes must then pass the same quality gates under both the
minimum artifact ABI and the latest locked runtime.

## 2. Fresh setup

From the repository root:

```bash
uv python install 3.12
uv sync --frozen --python 3.12 --extra dev --extra test
```

`uv sync` creates `.venv` and installs the project editable. Use
`--no-editable` for a closer approximation to an installed distribution.

After an intentional dependency change:

```bash
uv lock
uv lock --check
uv sync --frozen --python 3.12 --extra dev --extra test
.venv/bin/python -m pip check
```

Never hand-edit `uv.lock`.

## 3. Bootstrap resources

### Required for the complete base test suite

```bash
.venv/bin/python scripts/bootstrap_assets.py \
  --nltk \
  --contract-model \
  --contract-type-model
```

- NLTK language data is installed outside the Python distribution.
- The is-contract bootstrap uses the pinned `0.1` release as its trusted
  source. It downloads `0.2` when published; before publication it generates a
  local `0.2` re-export from `0.1`.
- The contract-type bootstrap builds or reuses its runtime artifact from a
  pinned corpus.
- Bundled segmentation/extraction models already ship in the distribution and
  must not be downloaded into the source tree.

Changing the repository alone intentionally fails closed: the configured
repository must match the identity in the trusted asset manifest. For a fork
or trusted mirror, set the repository and provide a locally reviewed manifest
whose repository identity, tags, filenames, sizes, and SHA-256 digests describe
that mirror:

```bash
export LEXNLP_MODELS_REPO_SLUG="<owner>/<repository>"
export LEXNLP_ASSET_MANIFEST="/path/to/reviewed-release_asset_manifest.json"
```

Or use a full GitHub API tags endpoint with the same reviewed manifest:

```bash
export LEXNLP_MODELS_REPO="https://api.github.com/repos/<owner>/<repository>/releases/tags/"
export LEXNLP_ASSET_MANIFEST="/path/to/reviewed-release_asset_manifest.json"
```

`LEXNLP_ASSET_MANIFEST` is a local trust root; do not download it implicitly
from the mirror being authorized. Publish workflows may deliberately use the
upstream repository override for trusted source inputs while publishing their
new output to `github.repository`. That source selection does not authorize a
different runtime mirror.

### Optional Stanford NLP

Stanford-dependent extractors require Java and the pinned legacy Stanford
assets:

```bash
.venv/bin/python scripts/bootstrap_assets.py --stanford
```

### Optional Apache Tika

Tika is an out-of-process document parsing service, not a dependency of core
LexNLP extraction. The bootstrap downloads and SHA-512 verifies Apache Tika
3.3.2's app and server-standard jars:

```bash
.venv/bin/python scripts/bootstrap_assets.py --tika
LEXNLP_USE_TIKA=true scripts/run_tika.sh
```

The launcher:

- finds `bin/` relative to the repository rather than the caller's directory;
- binds to `127.0.0.1`;
- verifies the exact Tika version through its HTTP endpoint; and
- refuses to replace an unrelated process already using the port.

Override the default port with `APACHE_TIKA_PORT`; override the binaries
directory with `APACHE_TIKA_BINARIES`.

Stanford, Tika, model, and corpus release assets are pinned, verified, and
installed atomically. NLTK resources come from the official `nltk_data`
repository and are verified with a separate pinned SHA-256 catalog, including
`omw-1.4` and `omw-2.0`; they are not covered by the model/corpus release-asset
manifest. Do not substitute a floating URL, unverified archive extraction, or
a committed third-party asset tree for protected assets.

## 4. Required validation

### Static and policy checks

```bash
.venv/bin/ruff check lexnlp scripts ci --select E4,E7,E9,F
.venv/bin/python ci/skip_audit.py
.venv/bin/python scripts/reexport_bundled_sklearn_models.py --check-current
```

The skip audit forbids unapproved `skip`, `skipif`, and `xfail` markers. A
genuine external limitation requires an inline annotation:

```text
skip-audit: issue=<link-or-id> expires=YYYY-MM-DD
```

### Tests

```bash
.venv/bin/pytest lexnlp
```

After provisioning Stanford assets and Java:

```bash
LEXNLP_USE_STANFORD=true .venv/bin/pytest \
  lexnlp/nlp/en/tests/test_stanford.py \
  lexnlp/extract/en/entities/tests/test_stanford_ner.py
```

Required suites have a 100% pass target. Do not alter test markers, fixtures,
expected extraction output, or metric baselines to hide a regression.

### Documentation

```bash
uv sync --frozen --python 3.12 --extra docs
.venv/bin/sphinx-build -W --keep-going \
  -b html documentation/docs/source documentation/docs/build/html
```

Read the Docs uses the same locked `docs` extra through `.readthedocs.yaml`.

## 5. Distribution validation

Build both artifacts and inspect their contents:

```bash
uv build
.venv/bin/python ci/check_dist_contents.py
```

Install the wheel and sdist independently in clean environments outside the
source tree. Both installs must expose the same runtime package data and model
assets:

```bash
uv venv /tmp/lexnlp-wheel --python 3.12
uv pip install --python /tmp/lexnlp-wheel/bin/python dist/*.whl

uv venv /tmp/lexnlp-sdist --python 3.12
uv pip install --python /tmp/lexnlp-sdist/bin/python dist/*.tar.gz
```

Run `ci/check_dist_contents.py --installed` as used by CI and smoke-import
`lexnlp` from a directory outside the repository.

## 6. Persisted-model ABI policy

Pickle/joblib files encode Python imports and numerical-library internals.
They are executable data: only load models from a trusted LexNLP release or
another trusted producer.

Every bundled or published model must be serialized with the exact producer
stack in `constraints/model-artifact-abi.txt`:

| Component | Producer version |
| --- | --- |
| joblib | 1.5.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.0 |
| scikit-learn | 1.7.2 |
| SciPy | 1.13.0 |

This is the oldest supported numerical ABI, not the latest development
environment. A model serialized under NumPy 2.x may import `numpy._core` and
fail under the supported NumPy 1.26 lower bound.

The release rule is therefore two-sided:

1. produce and quality-test the artifact under the exact constrained stack;
2. direct-load, predict, and repeat the quality gate under the latest locked
   stack.

The model scripts enforce and record producer versions, artifact SHA-256, and
size. Do not bypass that check.

### Bundled sklearn artifacts

Check that all committed artifacts already match the supported runtime:

```bash
.venv/bin/python scripts/reexport_bundled_sklearn_models.py --check-current
```

Only re-export them from the constrained producer environment:

```bash
uv venv /tmp/lexnlp-model-producer --python 3.12
uv pip install \
  --python /tmp/lexnlp-model-producer/bin/python \
  --constraint constraints/model-artifact-abi.txt \
  -e .
/tmp/lexnlp-model-producer/bin/python \
  scripts/reexport_bundled_sklearn_models.py
```

Review the provenance file and prove direct-load/prediction parity under both
the producer and latest locked environments before committing binary changes.

## 7. Strict pipeline-model quality gates

Quality gates compare fixed fixtures and committed metrics. Every allowed
regression remains exactly `0.0`.

### Is-contract model

```bash
.venv/bin/python scripts/model_quality_gate.py \
  --baseline-tag pipeline/is-contract/0.1 \
  --candidate-tag pipeline/is-contract/0.2 \
  --baseline-metrics-json test_data/model_quality/is_contract_baseline_metrics.json \
  --output-json artifacts/model_quality_gate.json \
  --max-accuracy-regression 0.0 \
  --max-f1-regression 0.0
```

To produce a candidate under the constrained stack:

```bash
/tmp/lexnlp-model-producer/bin/python scripts/reexport_contract_model.py \
  --source-tag pipeline/is-contract/0.1 \
  --target-tag pipeline/is-contract/0.2 \
  --baseline-metrics-json test_data/model_quality/is_contract_baseline_metrics.json \
  --force
```

Set `LEXNLP_IS_CONTRACT_MODEL_TAG` to select a validated candidate without an
API change.

### Contract-type model

```bash
.venv/bin/python scripts/contract_type_quality_gate.py \
  --baseline-tag pipeline/contract-type/0.2-runtime \
  --candidate-tag pipeline/contract-type/0.2-runtime \
  --baseline-metrics-json test_data/model_quality/contract_type_baseline_metrics.json \
  --output-json artifacts/contract_type_quality_gate.json \
  --max-accuracy-top1-regression 0.0 \
  --max-accuracy-topn-regression 0.0 \
  --max-f1-macro-regression 0.0 \
  --max-f1-weighted-regression 0.0
```

Train a release candidate only under the constrained producer stack:

```bash
/tmp/lexnlp-model-producer/bin/python scripts/train_contract_type_model.py \
  --target-tag pipeline/contract-type/0.2-runtime \
  --force \
  --output-json artifacts/model_training/contract_type_model_training_report.json
```

Set `LEXNLP_CONTRACT_TYPE_MODEL_TAG` to select a validated candidate.

Do not rewrite baseline JSON merely because a candidate regressed. A baseline
change requires an intentional, reviewed behavior change with evidence that
quality improved or remained equal.

### First publication of trusted model tags

The packaged and drift-check manifests already pin the exact reproducible
bytes for `pipeline/is-contract/0.2` and
`pipeline/contract-type/0.2-runtime`. After the code change first lands in the
upstream repository, dispatch both model publish workflows promptly. Until a
release exists, bootstrap treats an HTTP 404 for those exact default tags as a
staged-publication condition and deterministically rebuilds the same tag from
its pinned source; custom tags and all other trust failures remain fatal.

The scheduled asset-drift workflow is intentionally strict and will fail while
a manifest-pinned release is absent or has different bytes. Do not weaken that
check: publish the byte-identical artifact, verify its size and SHA-256 against
the manifest, then run the drift workflow manually once.

## 8. Minimum and latest dependency checks

Dependency bounds are promises. CI installs the declared direct lower bounds
and runs:

- bundled model compatibility;
- direct-load and prediction for both external models;
- both strict quality gates; and
- the complete base suite.

The normal locked environment covers the latest resolved stack. For every
dependency update, test both ends. Python compatibility changes also require
the full 3.10, 3.11, 3.12, and 3.13 matrix.

## 9. Failure triage

- **NLTK `LookupError`:** rerun `scripts/bootstrap_assets.py --nltk` in the
  same environment/user data location.
- **Missing contract tag:** rerun the matching `--contract-model` or
  `--contract-type-model` bootstrap.
- **Model imports `numpy._core`:** it was produced with an unsupported newer
  NumPy ABI; restore the trusted artifact and rebuild with
  `constraints/model-artifact-abi.txt`.
- **Model metric regression:** reject or improve the candidate; do not loosen
  a threshold.
- **Stanford missing jar/model:** rerun the verified `--stanford` bootstrap and
  confirm Java is on `PATH`.
- **Tika unavailable:** rerun `--tika`, then
  `LEXNLP_USE_TIKA=true scripts/run_tika.sh`; inspect
  `/tmp/tika-server-3.3.2.log`.
- **Skip audit failure:** remove the marker or provide a traceable,
  time-limited annotation.
- **Distribution mismatch:** rebuild from a clean tree and compare wheel/sdist
  package data before changing the checker.
- **Read the Docs failure:** reproduce with the fatal-warning Sphinx command
  above and keep `.readthedocs.yaml`, `pyproject.toml`, and `uv.lock` aligned.
