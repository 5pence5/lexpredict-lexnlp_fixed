.. _changes:

=========
Changelog
=========

Unreleased
----------

Added
~~~~~

* An experimental lossless English legal-document hierarchy with exact source
  spans, conservative and statute profiles, composable structural, paragraph
  and sentence backends, and realised-tree manifests.  Every character of the
  source is accounted for, and the leaves concatenate back to the input byte
  for byte.
* Deterministic chunking over that hierarchy, in character or explicit
  token-counter mode, with strict budgets, structure-preserving and dense
  container policies, explicit overlap, content-versus-overlap provenance,
  canonical manifests and digest-qualified chunk identities.
* A versioned embedding-payload serialiser for traced headings and table
  headers, with exact caller-tokeniser validation, canonical payload metadata
  identities and typed overflow.
* An opt-in caller-owned Segment Any Text adapter.  It requires a pinned
  backend identity and performs no model import or download of its own.
* ``scripts/segmentation_quality_gate.py`` and
  ``scripts/segmentation_benchmark.py``, which run on a bare interpreter with
  no third-party packages installed.
* Hermetic exact-span, structural, retrieval-plumbing and character/token
  operational regression gates with canonical evidence digests and a dedicated
  statute-profile scaling lane.  These small synthetic fixtures are not
  held-out SOTA evidence; representative legal, layout and multilingual
  evaluation remains a promotion gate.

Security
~~~~~~~~

* Model downloads are verified against a reviewed manifest of SHA256 digests
  shipped in the package.  Previously the only integrity check was an optional
  ``Content-MD5`` response header, supplied by the same server as the bytes it
  attested to.  Repository and asset URLs must be HTTPS, the asset host must
  match the manifest repository, release tags are validated as safe relative
  catalog paths so a tag cannot escape the catalog directory or traverse a
  symlink, and the install is atomic.  A tag absent from the manifest is
  refused before any network request.
* ``nltk`` moves to ``>=3.10.3``, clearing 35 known vulnerabilities in the
  previously pinned line.  ``click``, ``pygments`` and ``soupsieve`` move with
  it, clearing three more.
* Catastrophic backtracking is gone from the condition and constraint
  extractors.  Both wrapped their trigger alternation in wildcard groups, so
  trigger-free text backtracked quadratically: 10,000 characters took 2.1
  seconds in each.  It is now 0.0006 seconds and linear.  This was a
  denial-of-service exposure on attacker-supplied document text.

Fixed
~~~~~

* The multi-locale date dispatcher was unusable outside English.  It passed
  five positional arguments to parsers that accept three, so German, Spanish
  and Portuguese raised ``TypeError`` on every call.
* German amounts silently lost precision.  The English and German routines take
  their parameters in different orders, and positional dispatch handed German
  ``extended_sources=True`` as ``float_digits``, rounding to one decimal place
  instead of four.
* Court citations recorded no language when the caller did not name one.
* The layered definition detector could not load its own artifact: the asset
  pipeline ships a skops archive while the loader and its tests still pointed
  at the retired pickle.
* A clock-time guard in Portuguese ratios could be evaded by backtracking, so
  "10:30 a.m." was extracted as the ratio 10/3.
* ``SectionSegmentizerTrainManager.train_logistic_regression`` asked the
  ``lbfgs`` solver for an ``l1`` penalty, which it does not support, so the
  method raised on every call and could never return a model.
* The test suite no longer writes debug output into the tracked ``test_data``
  tree, which left the working copy dirty after every run.
* Portuguese was unreachable through the locale dispatchers.  ``lexnlp.extract.pt``
  ships native money, percent, amount, duration and citation extractors, but the
  matching ``all_locales`` dispatchers registered only ``en`` and ``de``, so
  ``"pt"`` and ``"pt-BR"`` fell through to the English routine.  Brazilian reais
  were reported as US dollars -- ``R$ 1.500.000,00`` came back with currency
  ``USD`` -- and Portuguese durations were not extracted at all.  The
  dispatchers now resolve the effective language before branching on calling
  convention, so an unregistered locale can no longer take a branch that does
  not match the routine it selected.
* Courts whose name began with the dictionary keyword were silently discarded.
  ``UniversalCourtsParser`` guarded on the keyword with
  ``pattern.search(text, re.IGNORECASE)``, but the second positional parameter
  of ``re.Pattern.search`` is ``pos``, not ``flags``.  ``re.IGNORECASE`` is
  ``2``, so every scan started at offset 2.  Spanish court extraction was
  effectively dead -- all 17 entries in ``es_courts.csv`` begin with
  ``Tribunal`` -- and Portuguese lost ``Tribunal Superior do Trabalho`` while
  ``Supremo Tribunal Federal`` kept working.  Affected ``de``, ``en``, ``es``
  and ``pt`` alike.

Infrastructure
~~~~~~~~~~~~~~

* Dependency upper bounds, so a breaking major release downstream cannot
  silently break installs.  Python support remains ``>=3.13,<3.15``.
* Three CI jobs: a documentation build with warnings fatal, a ``pip-audit``
  supply-chain audit over the exported runtime lock, and the segmentation
  quality and performance gate.
* Statement coverage is 100% across ``lexnlp``, ``scripts`` and ``ci``, with
  ``fail_under = 100`` configured so it stays there.  The suite went from 1,798
  passing with 6 failures to 3,871 passing with none.
* ``ci/check_dist_contents.py`` gains sdist and wheel parity checking,
  duplicate-member and forbidden-member detection.
* The documentation build is warning-clean under ``-W``; it previously emitted
  181 warnings.  ``sphinx.ext.napoleon`` and ``sphinx.ext.todo`` are enabled,
  and ``docs`` and ``audit`` extras declare the dependencies the build and the
  audit actually need.
* Retired ``.travis.yml`` and replaced the broken ``readthedocs.yml``, which
  pinned Python 3.8 and installed from a requirements file that no longer
  exists, with ``.readthedocs.yaml``.
* A ``dependabot.yml`` covering both the ``uv`` and ``github-actions``
  ecosystems.
* ``documentation/PR31_AUDIT.md`` records a measured regression audit of this
  branch against ``master``: a differential probe over 307 public extraction
  callables and a 32-text multilingual corpus, plus a full public-symbol diff
  across all 247 modules.  No feature was lost and no behaviour regressed; the
  only difference from ``master`` is the Portuguese ratio repair.  It also
  corrects an earlier artifact claim of "30 pickle, 0 skops": the real figure
  is 10 skops -- every bundled scikit-learn model -- and 20 pickles that hold
  pure lookup data with no class or reduction opcodes in them.
* ``documentation/LANGUAGE_PARITY_EN_ES_PT.md`` compares English, Spanish and
  Brazilian Portuguese capability by capability, with detection rates measured
  against ``faker``'s locale identifier shapes, and sets out a tiered route to
  parity.  Portuguese now reaches 15 working capabilities against English's 16;
  Spanish reaches 6.

Known limitations
~~~~~~~~~~~~~~~~~

* The paragraph feature window clamps its forward edge by subtracting the
  window size rather than the line position.  The arithmetic is wrong, but the
  bundled paragraph segmenter was trained on vectors produced by it, so
  correcting it without retraining makes the model stop splitting on blank
  lines.  The behaviour is pinned by characterisation tests until the model can
  be retrained and re-exported in the same change.
* The clause and list patterns in the new hierarchy recognise ``1.``, ``1.2``,
  ``A.1``, ``(a)``, ``(i)``, ``1)`` and bullets, but not bare ``a.`` / ``i.``
  or ``Article I``, so outlines using those markers are not detected.
* Spanish remains substantially behind English and Portuguese.  Nine
  extractors have no Spanish implementation at all (money, percents, amounts,
  durations, distances, ratios, citations, trademarks, urls), the Spanish court
  dictionary holds only the 17 Tribunales Superiores de Justicia and omits the
  Tribunal Supremo, the Tribunal Constitucional and the Audiencia Nacional, and
  ``EsLanguageTokens.conjunctions`` still contains the German ``und`` and
  ``oder``.  That last one is deliberately left alone: ``es/courts.py`` unions
  the conjunction list into a character-wise ``line_breaks`` set, where the
  German words are inert only because they are multi-character, so substituting
  the real single-letter Spanish conjunctions without also removing that union
  would shatter every court name containing an ``e``.
* ``pt/pii.py`` covers only email addresses and telephone numbers.  CPF and
  CNPJ are recognised, at 60/60 against generated samples, but only through
  ``pt/identifiers.py``, so a caller asking a Portuguese document for its PII
  receives no national identifiers.  English does not behave this way.
* Portuguese ratio extraction still matches inside Brazilian identifiers, so
  ``CNPJ 12.345.678/0001-95`` yields a spurious ratio.  Court lookup in both
  Spanish and Portuguese is accent-sensitive, so OCR output that has lost a
  cedilla does not match.

2.3.0 - November 30, 2022
-------------------------
 * Updated Python version and upgraded all dependencies.
 * Started using pipenv

2.2.1.0 - August 10, 2022
-------------------------
 * Improved LexNLP handling for companies for the "EN" locale.
 * Unified API pattern for sentence and paragraph segmentation.

2.2.0 - July 7, 2022
--------------------
 * Improved LexNLP handling for dates, durations and persons for the all locales.
 * Added parameterizable contract classifiers.
 * Improved LexNLP handling for ML models.
 * Updated python requirements and tests, retrained ML models to use gensim-4.

2.1.0 - September 16, 2021
--------------------------
 * Improved LexNLP handling for companies for the "EN" locale.
 * Improved LexNLP handling for dates for all locales and dates parser accuracy for the "DE" locale.

2.0.0 - May 10, 2021
--------------------
 * Tune extracting facts from text for different locales.
 * Updated regex patterns and tests for "DE" amount parser.
 * Added support for delimiter inference in "EN" and "DE" amount parsers.

1.8.0 - December 2, 2020
------------------------
 * Improved LexNLP handling for definitions for the "EN" locale.
 * Implemented rating OCR quality in texts.
 * Migrated numeric data in parsers results to decimal format to avoid losing fraction digits.

1.7.0 - August 27, 2020
-----------------------
 * Improved LexNLP handling for dates for the "EN" locale.
 * Implemented lists of exceptions for entity extractors.
 * Implemented strongly typed response for entity extractors.
 * Updated third-party python requirements.

1.6.0 - May 27, 2020
--------------------
 * Update psutil package version from 5.4.0 to 5.6.6.

1.4.0 - December 20, 2019
-------------------------
 * Improved accuracy of locating and converting date phrases into typed format.
 * Introduced new text vectorizing and classifying models.
 * Implemented ML-based definitions locator.

1.3.0 - November 1, 2019
------------------------
 * Made massive improvements to EN definitions and companies parsers.
 * Updated EN dates parser to catch more date formats.
 * Made company parsing strongly typed

0.2.7 - August 1, 2019
----------------------
 * Standardized LexNLP methods response to return a generator of Annotation objects or a generator of dictionaries (tuples)
 * Improved LexNLP handling for definitions for the "EN" locale.
 * Improved LexNLP handling for companies for the "EN" locale.
 * Improved sentence splitting logic.
 * Improved LexNLP unit test coverage.
 * Updated python requirements in python-requirements*.txt.
 * Dropped support for python 3.4 and 3.5.

0.2.6 - Jun 12, 2019
--------------------
 * Improved LexNLP handling for dates for all locales.
 * Improved LexNLP handling for currencies for "EN" locale.
 * Updated documentation for ReadTheDocs.
 * Improved LexNLP unit test coverage.

0.2.5 - Mar 1, 2019
-------------------
 * Improved LexNLP handling for courts for "DE" and "ES" locales.
 * Improved LexNLP handling for dates for "ES" locale.
 * Improved LexNLP handling for amounts, acts, regulations and definitions for "EN" locale.
 * Added CUSIP parser for "EN" locale.
 * Improved LexNLP unit test coverage.

0.2.4 - Feb 1, 2019
-------------------
 * Added universal courts parser, configured LexNLP handling for courts for "DE" locale.
 * Added universal dates parser, configured LexNLP handling for dates for "DE" and "ES" locales.
 * Added definitions, citations and dates parsers for "DE" locale.
 * Added amounts, percents and durations parsers for "DE" locale.
 * Added geo entities parser for "DE" locale.
 * Added courts and definitions parsers for "ES" locale.
 * Added acts parser for "EN" locale.
 * Improved LexNLP unit test coverage.

0.2.3 - Jan 10, 2019
--------------------
 * Updated python requirements.
 * Improved LexNLP handling for definitions and paragraphs.
 * Improved LexNLP unit test coverage.

0.2.2 - Sep 30, 2018
--------------------
 * Improved LexNLP handling for different date formats.
 * Improved LexNLP handling for titles.
 * Improved LexNLP unit test coverage.

0.2.1 - Aug 24, 2018
--------------------
 * Updated python requirements.
 * Improved LexNLP handling for amounts.
 * Optimized processing of sentences and titles.
 * Improved LexNLP unit test coverage.

0.2.0 - Aug 1, 2018
-------------------
 * Improved LexNLP handling for addresses and sentences.
 * Improved LexNLP unit test coverage.

0.1.9 - Jul 1, 2018
-------------------
 * Improved handling of TOC during sentence processing.
 * Added contracts locator to LexNLP.
 * Improved LexNLP handling for citations, titles and definitions.
 * Improved LexNLP unit test coverage.

0.1.8 - May 1, 2018
-------------------
 * Improved LexNLP handling for addresses and currencies.
 * Improved LexNLP unit test coverage.

0.1.7 - Apr 1, 2018
-------------------
 * Improved LexNLP handling for companies, organizations and dates.
 * Implemented generating train/test dataset for addresses.
 * Exclude common false positives for persons parser.

0.1.6 - Mar 1, 2018
-------------------
 * Improved LexNLP unit test coverage.

0.1.5 - Feb 1, 2018
-------------------
 * Improved LexNLP unit test coverage.

0.1.4 - Jan 1, 2018
-------------------
 * Improved LexNLP unit test coverage.
 * Implemented method to get sentence ranges in addition to sentence texts.

0.1.3 - Dec 1, 2017
-------------------
 * Improved LexNLP unit test coverage.

0.1.2 - Nov 1, 2017
-------------------
 * Implemented LexNLP title locator.
 * Implemented additional LexNLP transforms for skipgrams and n-grams.
 * Improved LexNLP handling for parties with abbreviations and other cases.
 * Improved LexNLP handling for amounts with mixed alpha and numeric characters.
 * Improved LexNLP unit test coverage.

0.1.1 - Oct 1, 2017
-------------------
 * Improve unit test framework handling for language and locales.
 * Implemented method and input-level CPU and memory benchmarking for unit tests.
 * Migrated all unit tests to 60 separate CSV files.
 * Added over 1,000 new unit tests for most LexNLP methods.
 * Reduced memory usage for paragraph and section segmenters.
 * Improved handling of brackets and parentheses within noun phrases.
 * Added URL locator to LexNLP.
 * Added trademark locator to LexNLP.
 * Added copyright locator to LexNLP.
 * Improved default Punkt sentence boundary detection.
 * Added custom sentence boundary training methods.
 * Improved handling of multilingual text, especially around geopolitical entities.
 * Improved default handling of party names with non-standard characters.
 * Enhanced metadata related to party type in LexNLP.
 * Improved continuous integration for public repositories.

0.1.0 - Sep 1, 2017
-------------------
 * Refactored and integrate core extraction into separate LexNLP package.
 * Released nearly 200 unit tests with over 500 real-world test cases in LexNLP.
 * Improved definition, date, and financial amount locators for corner cases.
 * Integrated PII locator for phone numbers, SSNs, and names from LexNLP.
 * Integrated ratio locator from LexNLP.
 * Integrated percent locator from LexNLP.
 * Integrated regulatory locator from LexNLP.
 * Integrated distance locator from LexNLP.
 * Integrated case citation locator from LexNLP.
 * Improved geopolitical locator to allow non-master-data entity location.
 * Improved party locator to allow configuration and better handle corner cases

