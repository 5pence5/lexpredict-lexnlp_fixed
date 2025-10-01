# Modernization recommendations for LexNLP

This repository still reflects historical tooling choices and requires a broad modernization effort before it can be maintained comfortably on contemporary Python versions. The notes below group together the most impactful clean-up tasks that emerged while reviewing the code base.

## 1. Refresh build and release automation
- **Replace Travis CI** – the project relies on a `.travis.yml` that still targets Python 3.4, downloads Java 8, and depends on long-deprecated `travis_wait` workflows. Migrating to GitHub Actions (or another actively maintained CI provider) would simplify dependency management and speed up feedback loops. A minimal matrix covering the supported Python versions plus cache configuration for the heavy NLP models would make the automated tests usable again.【F:.travis.yml†L1-L32】
- **Surface release automation** – the README stops at release 2.3.0 (2022). Reinstating an automated release pipeline that builds wheels, publishes to PyPI, and tags GitHub releases would make it easier for downstream projects to consume updates.【F:README.md†L31-L61】

## 2. Reassess Python version support
- **Align declared Python versions** – `setup.py` still advertises compatibility with Python 3.6 even though the README claims Python 3.8 is required. Explicitly documenting and testing against modern versions (3.9 – 3.12) will prevent confusing installation failures.【F:setup.py†L42-L53】【F:README.md†L23-L27】
- **Audit language features** – once the target versions are defined, sweep the code base for obsolete compatibility shims and simplify logic accordingly (e.g., removal of Python 3.6-specific workarounds or `six` usage).

## 3. Update dependency management
- **Review pinned runtime dependencies** – `setup.py` pins dozens of heavy packages (NumPy, pandas, scikit-learn, etc.) to specific patch versions. These constraints make upgrades difficult and can conflict with downstream environments. Converting to compatible version ranges (`>=, <`) based on actual API needs will ease installation friction.【F:setup.py†L58-L85】
- **Clean requirements files** – `python-requirements.txt` still references very old packaging tools (e.g., `pipenv==2022.11.11`) alongside future-dated pins such as `pytz==2025.2`. Curating these files, splitting out optional extras (documentation, dev, notebooks), and leaning on `pyproject.toml` for dependency metadata would modernize packaging.【F:python-requirements.txt†L1-L76】

## 4. Improve developer experience
- **Consolidate documentation sources** – the project mixes `README.md`, `README.rst`, and `index.rst` while ReadTheDocs points to stale versions. Converging on one canonical README and refreshing the documentation pipeline (potentially migrating to MkDocs or up-to-date Sphinx) would reduce duplication.【F:README.md†L1-L61】【F:index.rst†L1-L120】
- **Automate model downloads for local development** – contributors currently have to run shell scripts to fetch Stanford NLP and Apache Tika binaries via Travis-only hooks. Providing a cross-platform CLI or Makefile target would make onboarding less brittle.【F:.travis.yml†L19-L31】【F:libs/download_stanford_nlp.sh†L1-L80】【F:scripts/download_tika.sh†L1-L61】

## 5. Re-engage the test suite
- **Modernize test tooling** – the Travis script still runs `py.test` with pylint integration. Porting the tests to `pytest` 7+, enabling coverage reports locally, and wiring linting through pre-commit will make quality checks reproducible off CI.【F:.travis.yml†L32-L35】
- **Add smoke tests for packaged models** – given the reliance on external NLP models, lightweight integration tests that validate loading and inference across supported platforms would catch regressions earlier.

## 6. Clarify licensing and community expectations
- **Document the dual-license policy** – the README references AGPL/commercial licensing but lacks implementation details (e.g., what parts are proprietary). Providing a dedicated `LICENSE_POLICY.md` or FAQ would reduce confusion for adopters.【F:README.md†L41-L49】
- **Publish contribution guidelines** – adding a CONTRIBUTING guide with coding standards, review expectations, and triage policies would help revive community participation.

## 7. Plan for data and model stewardship
- **Audit bundled data** – large assets in `libs/` and `test_data/` may have gone stale. Verifying download URLs, checksums, and licensing status prevents accidental breakage for fresh clones.【F:libs/download_stanford_nlp.sh†L1-L80】【F:test_data/long_parsed_text.txt†L1-L32】
- **Evaluate model hosting** – long-term, consider hosting core models on a maintained CDN (or publishing to Hugging Face) to avoid reliance on external mirrors that could disappear.

Implementing the roadmap above will make LexNLP far easier to install, test, and extend in modern Python environments.
