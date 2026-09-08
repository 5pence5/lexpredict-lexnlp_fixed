|CI| |Coverage| |Python| |Docs|

LexNLP by LexPredict
====================

Information retrieval and extraction for real, unstructured legal text
----------------------------------------------------------------------

LexNLP is a library for working with real, unstructured legal text,
including contracts, plans, policies, procedures, and other material.

LexNLP provides functionality such as:
--------------------------------------

-  Segmentation and tokenization, such as

   -  A sentence parser that is aware of common legal abbreviations like
      LLC. or F.3d.
   -  Pre-trained segmentation models for legal concepts such as pages
      or sections.

-  Pre-trained word embedding and topic models, broadly and for specific
   practice areas
-  Pre-trained classifiers for document type and clause type
-  Broad range of fact extraction, such as:

   -  Monetary amounts, non-monetary amounts, percentages, ratios
   -  Conditional statements and constraints, like "less than" or "later
      than"
   -  Dates, recurring dates, and durations
   -  Courts, regulations, and citations

-  Lossless document segmentation: a hierarchy whose leaves concatenate
   back to the source byte for byte, with deterministic chunking and
   embedding payloads
-  Tools for building new clustering and classification methods
-  Thousands of unit tests from real legal documents, at 100% statement
   coverage

.. figure:: https://s3.amazonaws.com/lexpredict.com-marketing/graphics/lexpredict_lexnlp_logo_horizontal_1.png
   :alt: Logo

Information
===========

-  ContraxSuite: https://contraxsuite.com/
-  LexPredict: https://lexpredict.com/
-  Official Website: https://contraxsuite.com/lexnlp/
-  Documentation: http://lexpredict-lexnlp.readthedocs.io/en/latest/
   (in progress)
-  Contact: support@contraxsuite.com

Structure
---------

-  ContraxSuite web application:
   https://github.com/LexPredict/lexpredict-contraxsuite
-  LexNLP library for extraction:
   https://github.com/LexPredict/lexpredict-lexnlp
-  ContraxSuite pre-trained models and "knowledge sets":
   https://github.com/LexPredict/lexpredict-legal-dictionary
-  ContraxSuite agreement samples:
   https://github.com/LexPredict/lexpredict-contraxsuite-samples
-  ContraxSuite deployment automation:
   https://github.com/LexPredict/lexpredict-contraxsuite-deploy

Please note that ContraxSuite installations generally require trained models
or knowledge sets for usage.

Licensing
---------

LexNLP is available under a dual-licensing model. By default, this
library can be used under AGPLv3 terms as detailed in the repository
LICENSE file; however, organizations can request a release from the AGPL
terms or a non-GPL evaluation license by contacting ContraxSuite Licensing at
<license@contraxsuite.com>.

Requirements
------------

-  Python 3.13 (minimum; supported range ``>=3.13,<3.15`` is declared in ``pyproject.toml``)
-  ``uv``

Quick Setup (uv + pyproject)
----------------------------

.. code:: bash

   cd /path/to/LexNLP
   uv python install 3.13
   uv venv --python 3.13 .venv
   uv pip install --python .venv/bin/python -e ".[dev,test]"
   ./.venv/bin/python scripts/bootstrap_assets.py --nltk --contract-model

Optional dependency extras
--------------------------

==============  ================================  =============================================================================================================
Extra           Pin                               Powers
==============  ================================  =============================================================================================================
``[arrow]``     ``pyarrow>=17``                   ``read_csv_arrow`` and PyArrow-backed extraction DataFrames
``[audit]``     ``pip-audit>=2.7``                The dependency-vulnerability audit run in CI
``[docs]``      ``sphinx``, ``sphinx-rtd-theme``  Building the documentation (the CI build treats warnings as fatal)
``[hub]``       ``huggingface_hub>=0.25``         ``lexnlp.ml.catalog.hub`` HF Hub mirror downloads
``[ner]``       ``spacy>=3.7``                    Optional spaCy backend for ``lexnlp.extract.ner`` (default backend is NLTK; see ``MODERNIZATION_ROADMAP.md``)
``[tika]``      ``tika>=2.6.0``                   Apache Tika document-parsing helpers
``[stanford]``  *(empty)*                         Hooks for callers that ship their own Stanford CoreNLP jars
==============  ================================  =============================================================================================================

Install with e.g. ``uv pip install -e ".[ner,arrow]"``. None of these
extras are required for the rule-based extractors.

New module: ``lexnlp.extract.ner``
----------------------------------

A small statistical NER pass for entities the rule stack misses
(parties, agreement types, OCR-mangled proper nouns):

.. code:: python

   from lexnlp.extract.ner import (
       HybridNERMatch, augment_rule_matches, extract_entities,
   )

   # Default backend is NLTK (already a hard dep) — a deliberate
   # substitution for spaCy's gated ``en_core_web_sm``. spaCy is opt-in:
   matches = extract_entities("Acme Corp. and John Smith signed an NDA.")

   # Opt into spaCy when ``[ner]`` + ``en_core_web_sm`` are installed:
   matches = extract_entities(text, prefer_spacy=True)

   # Merge with the rule stack, dropping hybrid matches that overlap >=50%:
   merged = augment_rule_matches(rule_spans, matches)

The default NLTK backend needs four corpora downloaded once:
``punkt_tab``, ``averaged_perceptron_tagger_eng``,
``maxent_ne_chunker_tab`` and ``words``. See
``MODERNIZATION_ROADMAP.md`` §2.0.2 for why NLTK is the default.

Bundled artifacts: ``.pickle`` → ``.skops``
-------------------------------------------

The 10 bundled sklearn artifacts that previously shipped as ``.pickle``
files were re-exported as ``.skops`` siblings via
``scripts/reexport_bundled_sklearn_models.py --format skops``. The
legacy pickles were deleted; loaders use
``lexnlp.ml.model_io.load_bundled_model(legacy_path)`` which prefers
the ``.skops`` sibling. Tests that previously ERRORed at collection
under sklearn 1.8 + numpy 2.4 (DE court-citation, ML token-sequence)
now collect cleanly. See ``MODERNIZATION_ROADMAP.md`` §2.3 / Tier B.12.

Deprecated Setup Variants
-------------------------

``python-requirements.txt``, ``python-requirements-dev.txt``, and
``python-requirements-full.txt`` are deprecated and kept only for legacy
reproduction. The ``Pipfile`` / ``Pipfile.lock`` pair has been removed;
``ci/check_dist_contents.py`` still bans both from built artifacts so
they cannot re-enter the sdist/wheel. Use ``uv`` with ``pyproject.toml``
for all local setup and CI workflows.

Releases
--------

-  2.3.0: November 30, 2022 - Twenty sixth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/2.3.0>`__
-  2.2.1.0: August 10, 2022 - Twenty fifth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/2.2.1.0>`__
-  2.2.0: July 7, 2022 - Twenty fourth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/2.2.0>`__
-  2.1.0: September 16, 2021 - Twenty third scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/2.1.0>`__
-  2.0.0: May 10, 2021 - Twenty second scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/2.0.0>`__
-  1.8.0: December 2, 2020 - Twenty first scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/1.8.0>`__
-  1.7.0: August 27, 2020 - Twentieth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/1.7.0>`__
-  1.6.0: May 27, 2020 - Nineteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/1.6.0>`__
-  1.4.0: December 20, 2019 - Eighteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/1.4.0>`__
-  1.3.0: November 1, 2019 - Seventeenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/1.3.0>`__
-  0.2.7: August 1, 2019 - Sixteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.7>`__
-  0.2.6: June 12, 2019 - Fifteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.6>`__
-  0.2.5: March 1, 2019 - Fourteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.5>`__
-  0.2.4: February 1, 2019 - Thirteenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.4>`__
-  0.2.3: Junuary 10, 2019 - Twelfth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.3>`__
-  0.2.2: September 30, 2018 - Eleventh scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.2>`__
-  0.2.1: August 24, 2018 - Tenth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.1>`__
-  0.2.0: August 1, 2018 - Ninth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.2.0>`__
-  0.1.9: July 1, 2018 - Ninth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.9>`__
-  0.1.8: May 1, 2018 - Eighth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.8>`__
-  0.1.7: April 1, 2018 - Seventh scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.7>`__
-  0.1.6: March 1, 2018 - Sixth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.6>`__
-  0.1.5: February 1, 2018 - Fifth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.5>`__
-  0.1.4: January 1, 2018 - Fourth scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.4>`__
-  0.1.3: December 1, 2017 - Third scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.3>`__
-  0.1.2: November 1, 2017 - Second scheduled public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.2>`__
-  0.1.1: October 2, 2017 - Bug fix release for 0.1.0;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.1>`__
-  0.1.0: September 30, 2017 - First public release;
   `code <https://github.com/LexPredict/lexpredict-lexnlp/tree/0.1.0>`__

.. |CI| image:: https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml
.. |Coverage| image:: https://img.shields.io/badge/coverage-100%25-brightgreen
   :target: https://github.com/cicero-im/lexpredict-lexnlp/actions/workflows/ci.yml
.. |Python| image:: https://img.shields.io/badge/python-3.13%20%7C%203.14-blue
   :target: https://github.com/cicero-im/lexpredict-lexnlp
.. |Docs| image:: https://readthedocs.org/projects/lexpredict-lexnlp/badge/?version=latest
   :target: https://lexpredict-lexnlp.readthedocs.io/en/latest/
