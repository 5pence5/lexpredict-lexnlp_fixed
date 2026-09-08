[![CI](https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml/badge.svg)](https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml) [![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue)](https://github.com/cicero-im/lexpredict-lexnlp) [![Docs](https://readthedocs.org/projects/lexpredict-lexnlp/badge/?version=latest)](https://lexpredict-lexnlp.readthedocs.io/en/latest/)

# LexNLP by LexPredict
## Information retrieval and extraction for real, unstructured legal text
LexNLP is a library for working with real, unstructured legal text, including contracts, plans, policies, procedures,
and other material.
## LexNLP provides functionality such as:
* Segmentation and tokenization, such as
    * A sentence parser that is aware of common legal abbreviations like LLC. or F.3d.
    * Pre-trained segmentation models for legal concepts such as pages or sections.
* Pre-trained word embedding and topic models, broadly and for specific practice areas
* Pre-trained classifiers for document type and clause type
* Broad range of fact extraction, such as:
    * Monetary amounts, non-monetary amounts, percentages, ratios
    * Conditional statements and constraints, like "less than" or "later than"
    * Dates, recurring dates, and durations
    * Courts, regulations, and citations
* Lossless document segmentation: a hierarchy whose leaves concatenate back to
  the source byte for byte, with deterministic chunking and embedding payloads
* Tools for building new clustering and classification methods
* Thousands of unit tests from real legal documents, at 100% statement coverage

![Logo](https://s3.amazonaws.com/lexpredict.com-marketing/graphics/lexpredict_lexnlp_logo_horizontal_1.png)

# Information
* ContraxSuite: https://contraxsuite.com/
* LexPredict: https://lexpredict.com/
* Official Website: https://lexnlp.com/
* Documentation: http://lexpredict-lexnlp.readthedocs.io/en/latest/ (in progress)
* Contact: support@contraxsuite.com

## Structure
* ContraxSuite web application: https://github.com/LexPredict/lexpredict-contraxsuite
* LexNLP library for extraction: https://github.com/LexPredict/lexpredict-lexnlp
* ContraxSuite pre-trained models and "knowledge sets": https://github.com/LexPredict/lexpredict-legal-dictionary
* ContraxSuite agreement samples: https://github.com/LexPredict/lexpredict-contraxsuite-samples
* ContraxSuite deployment automation: https://github.com/LexPredict/lexpredict-contraxsuite-deploy
Please note that ContraxSuite installations generally require trained models or knowledge sets for usage.

## Licensing
LexNLP is available under a dual-licensing model.  By default, this library can be used under AGPLv3 terms as detailed
in the repository LICENSE file; however, organizations can request a release from the AGPL terms or a non-GPL
evaluation license
by contacting ContraxSuite Licensing at <<license@contraxsuite.com>>.

## Requirements
* Python 3.13 (minimum; supported range `>=3.13,<3.15` is declared in `pyproject.toml`)
* `uv`

## Quick Setup (uv + pyproject)
```bash
cd /path/to/LexNLP
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[dev,test]"
./.venv/bin/python scripts/bootstrap_assets.py --nltk --contract-model
```

## Optional dependency extras

| Extra | Pin | Powers |
| --- | --- | --- |
| `[arrow]` | `pyarrow>=17` | `read_csv_arrow`, PyArrow-backed extraction DataFrames |
| `[audit]` | `pip-audit>=2.7` | The dependency-vulnerability audit run in CI |
| `[docs]` | `sphinx`, `sphinx-rtd-theme` | Building the documentation (the CI build treats warnings as fatal) |
| `[hub]` | `huggingface_hub>=0.25` | `lexnlp.ml.catalog.hub` HF Hub mirror downloads |
| `[ner]` | `spacy>=3.7` | Optional spaCy backend for `lexnlp.extract.ner` (default backend is NLTK; see below) |
| `[tika]` | `tika>=2.6.0` | Apache Tika document-parsing helpers |
| `[stanford]` | _(empty)_ | Hooks for callers that ship their own Stanford CoreNLP jars |

Install one or more via e.g. `uv pip install -e ".[ner,arrow]"`. None
of them are required for the rule-based extractors — install only what
your project actually uses.

## Build system
The project now uses Astral's native [`uv_build`](https://docs.astral.sh/uv/concepts/build-backends/#uv-build) backend — the `[build-system]` in `pyproject.toml` declares `requires = ["uv_build>=0.9,<0.10"]` and `build-backend = "uv_build"`. This drops setuptools/wheel from the build toolchain and keeps the build, resolve and lint toolchain in a single vendor. Build with:

```bash
uv build           # sdist + wheel
uv build --wheel   # wheel only
```

## New in this branch: `lexnlp.nlp.en.segments` (lossless segmentation)

A document hierarchy that accounts for every character of the source. The leaves
concatenate back to the input byte for byte, and the tree carries exact spans, so
a chunk can always be traced to the text it came from.

```python
from lexnlp.nlp.en.segments import (
    ContainerPolicy,
    StructureProfile,
    chunk_document,
    segment_document,
)

# Parse a document into a hierarchy with exact source spans:
hierarchy = segment_document(text, profile=StructureProfile.CONSERVATIVE)
assert hierarchy.reconstruct() == text          # lossless, byte for byte

for segment in hierarchy.segments():
    print(segment.level, segment.kind.value, segment.label, segment.start, segment.end)

# Chunk it under a strict character budget, preserving structure:
chunks = list(
    chunk_document(
        text,
        max_chars=1000,
        overlap_chars=100,
        container_policy=ContainerPolicy.STRUCTURE_PRESERVING,
    )
)
for chunk in chunks:
    print(chunk.identity, chunk.provenance.content_start, chunk.provenance.content_end)
```

Chunking also accepts an explicit token counter, so the budget can be enforced
against the caller's own tokeniser rather than a character proxy:

```python
from lexnlp.nlp.en.segments import TokenCounterPolicy, chunk_document

chunks = list(
    chunk_document(
        text,
        max_tokens=512,
        token_counter=my_tokeniser.count,
        token_counter_policy=TokenCounterPolicy.EXACT,
    )
)
```

Sentence boundaries are pluggable. The Segment Any Text adapter is opt-in,
caller-owned, and imports nothing on its own — you construct and pin the model:

```python
from lexnlp.nlp.en.segments import SaTSentenceSegmenter

segmenter = SaTSentenceSegmenter(model=my_sat_model, backend_id="sat-3l-sm@1.0")
hierarchy = segment_document(text, sentence_segmenter=segmenter)
```

`hierarchy.py`, `chunks.py`, `payloads.py` and `backends.py` import nothing
outside the standard library, so the quality gate and benchmark run on a bare
interpreter:

```bash
python scripts/segmentation_quality_gate.py --json-output artifacts/quality.json
python scripts/segmentation_benchmark.py --mode characters --characters 200000 \
    --min-throughput 10000 --max-peak-mib 64
```

Both are wired into CI. See `documentation/docs/source/guides/lossless_segmentation_api.rst`
for the full API and `development/segmentation_v2_design.rst` for the design notes.

## New in this branch: `lexnlp.extract.batch`
Concurrent and Arrow-native extraction helpers that exercise the Python 3.13 feature set declared in `pyproject.toml`:

```python
from lexnlp.extract.batch import extract_batch, annotations_to_dataframe, find_fuzzy_dates
from lexnlp.extract.en.amounts import get_amount_annotations

# Concurrent batch extraction via ``asyncio.TaskGroup``:
results = extract_batch(get_amount_annotations, docs, max_workers=8)

# Convert any iterable of annotations to a PyArrow-backed pandas DataFrame:
df = annotations_to_dataframe(ann for r in results for ann in r.annotations)

# Fuzzy ISO-date matcher built on the ``regex`` 2024+ engine:
matches = list(find_fuzzy_dates("Shipped 2O24-01-15", max_edits=1))
```

See `MODERNIZATION_ROADMAP.md` §4.0 for the full design.

## New in this branch: `lexnlp.extract.ner` (hybrid NER fallback)

A small statistical NER pass that recovers entities the rule stack
misses (parties, agreement types, OCR-mangled proper nouns):

```python
from lexnlp.extract.ner import (
    HybridNERMatch, augment_rule_matches, extract_entities,
)

# Default backend is NLTK (already a hard dep) — a deliberate
# substitution for spaCy's gated ``en_core_web_sm``. spaCy is opt-in:
matches = extract_entities("Acme Corp. and John Smith signed an NDA.")
print(matches[0])  # HybridNERMatch(start=..., end=..., text='Acme Corp', label='ORG', backend='nltk', score=None)

# Opt into spaCy when you have ``[ner]`` + ``en_core_web_sm`` installed:
matches = extract_entities(text, prefer_spacy=True)

# Merge with the rule stack, dropping hybrid matches that overlap >=50%:
merged = augment_rule_matches(rule_spans, matches)
```

The default NLTK backend needs four corpora downloaded once via
`nltk.download(...)`: `punkt_tab`, `averaged_perceptron_tagger_eng`,
`maxent_ne_chunker_tab`, `words`. See `MODERNIZATION_ROADMAP.md` §2.0.2
for why NLTK is the default and how the spaCy substitution shipped.

## Migrated bundled artifacts: `.pickle` → `.skops`

The 10 bundled sklearn artifacts that previously shipped as `.pickle`
files (`lexnlp/extract/{de,en}/...`, `lexnlp/extract/en/addresses/`,
`lexnlp/extract/ml/en/data/`, `lexnlp/nlp/en/segments/`) have been
re-exported as `.skops` siblings via
`scripts/reexport_bundled_sklearn_models.py --format skops`. The legacy
pickles were deleted; loaders use the new
`lexnlp.ml.model_io.load_bundled_model(legacy_path)` helper that prefers
the `.skops` sibling and falls back to the legacy pickle when present.
Tests that previously ERRORed at collection under sklearn 1.8 + numpy
2.4 (DE court-citation, ML token-sequence) now collect cleanly.

To reproduce or extend the migration on a downstream fork:

```bash
.venv/bin/python scripts/reexport_bundled_sklearn_models.py --format skops
.venv/bin/python scripts/reexport_bundled_sklearn_models.py \
    --format skops --remove-legacy           # delete .pickle siblings
```

## Deprecated Setup Variants
`python-requirements.txt` and `python-requirements-dev.txt` are deprecated and kept only for legacy reproduction. The `Pipfile` / `Pipfile.lock` pair has been removed — `ci/check_dist_contents.py` continues to ban both from built artifacts. Use `uv` with `pyproject.toml` for all local setup and CI workflows.

## Migration Runbook
See `MIGRATION_RUNBOOK.md` for complete migration/triage/quality-gate procedures.

## Test Integrity and Full Validation
- Do not add/remove/modify `skip`, `skipif`, or `xfail` markers to bypass failing tests.
- Target is **100% pass**.
- If Stanford assets are enabled, 100% pass includes both base and Stanford-only suites.

```bash
# Base suite
./.venv/bin/pytest lexnlp

# Stanford-only suite (run when Stanford assets are installed)
PATH=/opt/homebrew/opt/openjdk/bin:$PATH \
LEXNLP_USE_STANFORD=true \
./.venv/bin/pytest \
  lexnlp/nlp/en/tests/test_stanford.py \
  lexnlp/extract/en/entities/tests/test_stanford_ner.py
```

## Releases
* 2.3.0: November 30, 2022 - Twenty sixth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/2.3.0)
* 2.2.1.0: August 10, 2022 - Twenty fifth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/2.2.1.0)
* 2.2.0: July 7, 2022 - Twenty fourth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/2.2.0)
* 2.1.0: September 16, 2021 - Twenty third scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/2.1.0)
* 2.0.0: May 10, 2021 - Twenty second scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/2.0.0)
* 1.8.0: December 2, 2020 - Twenty first scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/1.8.0)
* 1.7.0: August 27, 2020 - Twentieth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/1.7.0)
* 1.6.0: May 27, 2020 - Nineteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/1.6.0)
* 1.4.0: December 20, 2019 - Eighteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/1.4.0)
* 1.3.0: November 1, 2019 - Seventeenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/1.3.0)
* 0.2.7: August 1, 2019 - Sixteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.7)
* 0.2.6: June 12, 2019 - Fifteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.6)
* 0.2.5: March 1, 2019 - Fourteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.5)
* 0.2.4: February 1, 2019 - Thirteenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.4)
* 0.2.3: Junuary 10, 2019 - Twelfth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.3)
* 0.2.2: September 30, 2018 - Eleventh scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.2)
* 0.2.1: August 24, 2018 - Tenth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.1)
* 0.2.0: August 1, 2018 - Ninth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.0)
* 0.1.9: July 1, 2018 - Ninth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.9)
* 0.1.8: May 1, 2018 - Eighth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.8)
* 0.1.7: April 1, 2018 - Seventh scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.7)
* 0.1.6: March 1, 2018 - Sixth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.6)
* 0.1.5: February 1, 2018 - Fifth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.5)
* 0.1.4: January 1, 2018 - Fourth scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.4)
* 0.1.3: December 1, 2017 - Third scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.3)
* 0.1.2: November 1, 2017 - Second scheduled public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.2)
* 0.1.1: October 2, 2017 - Bug fix release for 0.1.0; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.1)
* 0.1.0: September 30, 2017 - First public release; [code](https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.0)
