.. _segmentation_v2_design:

Legal segmentation and retrieval chunking design
================================================

Status and claim boundary
-------------------------

This design record was last reviewed on 4 August 2026.  LexNLP 2.4.0a1 contains
an additive **experimental v1**:

* an exact-span English hierarchy with conservative/statute structure profiles,
  pluggable structural/paragraph/sentence evidence and realised-tree manifests;
* strict character or explicit-token-counter chunking with structure-preserving
  and dense sibling policies;
* authenticated chunk text, provenance and canonical metadata identities;
* a versioned final embedding-payload serialiser; and
* hermetic boundary/retrieval plumbing gates, operational benchmark lanes and
  an opt-in caller-owned Segment Any Text adapter.

The alpha marker permits API/schema refinement before a stable release.
Persisted meaning must change only with explicit schema/serialiser migration.

Not implemented are a locale-neutral/discontiguous source model, PDF/Docling
adapter, typed page geometry/table cells, footnote/cross-reference graphs,
NUPunkt or CharBoundary adapters, automatic payload-aware repacking and a
streaming parser.

Most importantly, no representative held-out legal, layout or multilingual
evaluation has yet established that the heuristics or packing policy are state
of the art.  The branch supplies a versioned architecture and benchmark seam
intended to support that comparison, not a demonstrated SOTA result.

Decision
--------

The design is **structure first, strict budget second**:

1. retain one canonical source string and exact half-open spans;
2. build legal hierarchy before retrieval packing;
3. preserve complete sections, clauses and tables by default;
4. make dense packing, model backends and layout parsing explicit options; and
5. promote a backend/default only when boundary, retrieval and operational
   evidence pass together.

This is consistent with, but not proved by, recent work.  A 36-method
cross-domain preprint reports paragraph grouping as its strongest legal-domain
strategy, while a German-code preprint reports section/subsection-aligned
chunks outperforming the more complex methods it tested [Shaukat2026]_
[Prior2026]_.  Those are dataset-specific hypotheses, not LexNLP results.

Implemented invariants
----------------------

Source fidelity
~~~~~~~~~~~~~~~

* `start` is inclusive and `end` exclusive, measured against
  `DocumentHierarchy.source`.
* `segment.text(source) == source[start:end]` for every node.
* Every non-leaf is an ordered, contiguous, complete child partition.
* Detector normalisation is separate from source text; offsets are never
  recovered by ambiguous substring search.
* Every source character is retained.  v1 has no exclusion/coverage-ledger
  type.

Identity and determinism
~~~~~~~~~~~~~~~~~~~~~~~~

* `HierarchyManifest.tree_sha256` hashes the complete realised tree, including
  labels, levels and attributes.
* `ChunkingManifest.manifest_id` hashes canonical source, hierarchy, budget,
  overlap, boundary/container policy, token-counter and schema evidence.
* Each chunk independently authenticates full text (including overlap), all
  provenance partitions and canonical per-chunk metadata.
* `chunk_id` is source-, manifest- and chunk-metadata-qualified.
* `EmbeddingPayload.payload_metadata_sha256` binds the chunk, exact source and
  final payload text, ordered context, tokeniser identity, maximum/observed
  counts and serialisation version.
* Identical source, pinned backends and configuration produce identical trees,
  chunks and payload identities.

Budget correctness
~~~~~~~~~~~~~~~~~~

* Token mode requires the actual counter and `token_counter_id`.
* Every source slice is independently verified against its strict limit.
* `render_embedding_payload` counts the exact final context-plus-source string.
* Overflow is typed and never silently truncated.
* Text added after rendering is outside the validated payload.

Implemented pipeline
--------------------

`boundary evidence`
    Built-in or supplied structural spans, supplied/default paragraph spans and
    supplied/default sentence spans.  Empty custom predictions remain
    unclassified text.

`hierarchy builder`
    Produces a laminar tree of sections, clauses, list items, tables,
    paragraphs, sentences, text and separators.

`strict-budget packer`
    `ContainerPolicy.PRESERVE` keeps complete legal containers separate;
    `PACK_SIBLINGS` is explicit dense packing.  Oversized containers descend
    through finer boundaries before a hard strict split.  Overlap is inside the
    budget and can be reduced to avoid damaging a fitting container.

`final payload serialiser`
    Prepends true carried ancestry or supplied table headers under one exact
    v1 format, counts it and returns evidence or `PayloadBudgetExceeded`.

The core is intentionally plain-string and contiguous.  A later layout adapter
would need canonical extracted text plus pages, blocks, reading order, geometry
and OCR confidence.  Pages are generally orthogonal physical provenance:
clauses and paragraphs can cross them.

Bounded v1 versus later work
----------------------------

Embedding-payload v1 removes silent context-budget overclaim without requiring
a multi-span layout engine.  It keeps the exact chunk and validates the final
rendered string.  When context causes overflow, the caller reserves headroom
or rechunks.

Automatic context reservation/repacking, target-size optimisation, split-reason
diagnostics, multi-span logical units and table-cell-aware packing require a
new versioned contract.  They must not be inferred from `max_tokens` or hidden
inside a stable serializer.

Optional challengers
--------------------

`Segment Any Text (SaT)`
    The EMNLP paper evaluates punctuation-robust multilingual segmentation,
    including legal-domain experiments [Frohmann2024]_.  LexNLP's adapter
    accepts a caller-created model, requires a pinned backend/config ID, does
    not download weights and rejects non-lossless output.  It currently runs
    per paragraph rather than through upstream multi-document batching.  It is
    an optional challenger, not a promoted default.

`NUPunkt`
    Its authors report high-precision CPU-oriented legal sentence detection
    [Bommarito2025]_.  The current package requires Python 3.11+, whereas this
    LexNLP release supports Python 3.10; it cannot be an unconditional
    dependency without resolving that matrix [NUPunkt]_.

`CharBoundary`
    The same work provides character-level sentence/paragraph models and an
    optional ONNX path [Bommarito2025]_ [CharBoundary]_.  Model revision,
    install size, supported Python/runtime and held-out LexNLP results must be
    measured before adoption.

`Docling`
    Docling represents layout-aware documents and its official hybrid chunker
    starts with structure before token-aware split/merge passes
    [DoclingReport]_ [DoclingChunking]_.  A LexNLP adapter should remain
    optional and declare that offsets address canonical extracted text, not PDF
    bytes or glyphs.

A multilingual sentence backend does not make LexNLP's English structure
profiles multilingual.

Hermetic repository gates
-------------------------

Run the two repository scripts::

    python scripts/segmentation_quality_gate.py \
        --json-output build/segmentation-quality.json

    python scripts/segmentation_benchmark.py \
        --json-output build/segmentation-benchmark.json

CI also runs the lexical-token operational lane.  The quality candidate is
exact source text with `max_chars=180`, `overlap_chars=24`,
`respect_boundaries=True` and `container_policy=preserve`.

Default quality thresholds are:

* exact reconstruction and partial structural-anchor recall 1.0;
* exact zero-tolerance precision/recall/F1 1.0 independently for section,
  clause, list item, paragraph and sentence spans;
* complete structural precision/recall/F1 1.0;
* retrieval MRR at least 0.75, character recall@3 1.0 and context precision@1
  at least 0.25; and
* mean chunk length no greater than 180 characters and indexed-character
  amplification no greater than 1.5.

The final frozen fixture run matched 34/34 partial anchors across conservative
and statute profiles.  Exact true positives were 1 section, 1 clause, 2 list
items, 8 paragraphs and 10 sentences, with no false positives or negatives.
Lexical retrieval MRR was 0.8333, character recall@3 1.0 and context
precision@1 0.5531.  Mean chunk length was 90.55 and amplification 1.0143.

These values come from hermetic, synthetic fixtures: only eight edge cases
(1,351 characters), three complete-boundary cases (221 characters), and two
retrieval documents/six queries (982 characters).  They test invariants and
metric wiring, not population quality.  The report labels this evidence
accordingly and records that token-mode and contextual embedding-payload
retrieval have not been evaluated.

Evidence binding
~~~~~~~~~~~~~~~~

The JSON report records canonical SHA-256 for edge gold, boundary gold,
retrieval manifest, exact candidate configuration, generated candidate set and
rankings, together with its `canonical_digest_format`.  External rankings
must:

* identify retriever, index revision, embedding and tokeniser;
* bind retrieval/configuration/candidate digests; and
* rank generated chunk IDs, indices or exact generated spans, never arbitrary
  snippets.

The external seam currently evaluates character-budgeted exact source chunks.
It does not yet evaluate token-mode or contextual embedding payloads.

Operational evidence
~~~~~~~~~~~~~~~~~~~~

The character gate runs a deterministic 200,000-character legal-like document
three times with 1,000-character chunks and 100-character overlap.  Defaults
require at least 5,000 characters/second and at most 256 MiB of Python memory
traced by `tracemalloc`.  CI separately exercises a 100,000-character lexical
token lane.

A final integration run observed about 31,364 characters/second and 8.71 MiB
traced peak for character mode, and 30,454 characters/second and 4.45 MiB for
token mode.  These are environment-specific regression observations, not a
service objective, RSS measurement or external-backend comparison.

Statute scaling is a separate required lane.  It must generate an increasing
sequence of numbered statute headings (including thousands of candidates), run
`StructureProfile.STATUTE`, assert deterministic/lossless output and gate both
absolute throughput and growth ratios.  The conservative 200k benchmark cannot
stand in for this adversarial profile: an all-pairs heading algorithm can look
fast on conservative text while becoming quadratic on a much smaller statute.
Retain this lane in CI whenever statute detection changes.

External evaluation methodology
-------------------------------

Evaluate every candidate at five layers:

.. list-table::
   :header-rows: 1
   :widths: 18 50 28

   * - Layer
     - Measures
     - Resampling unit
   * - Invariants
     - Reconstruction, span/tree validity, authenticated identities, coverage,
       strict source and final-payload budgets
     - Document
   * - Boundaries
     - Exact P/R/F1 by kind, plus separately reported tolerance windows
     - Document
   * - Packing
     - Structural splits by kind, chunk/token distributions, overlap and index
       amplification/context duplication
     - Document
   * - Retrieval
     - Character precision/recall, recall@k, MRR and nDCG under one fixed index
     - Query, clustered by document
   * - Operations
     - Cold/warm throughput, p50/p95 latency, RSS/native/Python memory, model
       size and index cost
     - Repeated corpus run

LegalBench-RAG provides legal queries with gold character ranges and evaluation
code [LegalBenchRAGPaper]_ [LegalBenchRAGCode]_.  Use its exact released
manifest and component terms; do not silently regenerate its LLM-assisted data.

Corpora
~~~~~~~

Report separately:

* frozen compatibility fixtures and representative long existing documents;
* an adjudicated boundary corpus stratified by contract, statute/regulation,
  judgment/filing, definition, citation, nested list, table and poor OCR;
* a born-digital/OCR layout corpus with reading order, page, table, footnote and
  caption truth; and
* LegalBench-RAG mini/full for retrieval.

Multilingual claims require language-specific sets.  Split by source document
and source family before tuning so near-duplicate clauses cannot cross
train/development/test.  Freeze manifests/hashes and open the test set only
after configuration is locked.

Controlled comparison
~~~~~~~~~~~~~~~~~~~~~

At minimum compare:

* legacy output and a return-model-only wrapper;
* fixed-character and fixed-token windows;
* paragraph grouping and structure-first preserve/dense policies;
* each boundary backend with the same hierarchy/packer;
* exact source chunks versus final contextual payloads; and
* layout adapter versus its plain-text export on identical documents.

Hold retriever, embedding model/revision, similarity, index, query set, `k`,
tokeniser and contextualisation fixed.  Repeat leaders with a materially
different embedding family because chunking/embedding interactions are
plausible [Shaukat2026]_.

Report paired bootstrap 95% intervals at document/query level, distributions
rather than only means, and practical costs.  Predeclare non-inferiority
margins and budgets.

Promotion stages
----------------

`experimental alpha`
    Source/tree/chunk/payload invariants pass and compatibility APIs are
    unchanged.  This is 2.4.0a1's status.

`optional backend`
    Supported Python/OS gates pass, provenance is pinned and it lies on a
    quality/operations Pareto frontier for a declared use case.

`recommended policy/backend`
    Predeclared non-inferiority passes every protected slice, at least one
    primary quality measure improves with uncertainty, and costs remain within
    budget.

`stable default`
    Migration and persisted-format policy are documented, fallbacks are
    explicit, and evidence covers intended legal genres/languages/layouts.

An average win cannot compensate for a protected-subset regression, lost source
traceability or budget violation.

Remaining work
--------------

1. freeze the alpha schemas after downstream feedback and version every
   persisted-format change;
2. build/adjudicate the held-out boundary/layout corpora and publish manifests;
3. run LegalBench-RAG mini/full with exact final payloads and multiple embedding
   families;
4. compare legacy/windows/paragraph/preserve/dense plus SaT, NUPunkt and
   CharBoundary challengers;
5. build the optional layout adapter only with typed reading order, tables,
   footnotes, geometry and OCR confidence;
6. add automatic payload-aware repacking only under a new contract; and
7. recommend/stabilise only after predeclared promotion gates pass.

Primary sources
---------------

.. [Frohmann2024] Frohmann et al., `Segment Any Text: A Universal Approach for
   Robust, Efficient and Adaptable Sentence Segmentation
   <https://aclanthology.org/2024.emnlp-main.665/>`_, EMNLP 2024.
.. [SaT] Segment Any Text authors, `wtpsplit reference implementation
   <https://github.com/segment-any-text/wtpsplit>`_.
.. [Bommarito2025] Bommarito, Katz and Bommarito, `Precise Legal Sentence
   Boundary Detection for Retrieval at Scale: NUPunkt and CharBoundary
   <https://arxiv.org/abs/2504.04131>`_, 2025.
.. [NUPunkt] ALEA Institute, `NUPunkt reference implementation
   <https://github.com/alea-institute/nupunkt>`_.
.. [CharBoundary] ALEA Institute, `CharBoundary reference implementation
   <https://github.com/alea-institute/charboundary>`_.
.. [DoclingReport] Docling team, `Docling Technical Report
   <https://arxiv.org/abs/2408.09869>`_.
.. [DoclingChunking] Docling project, `official chunking documentation
   <https://docling-project.github.io/docling/concepts/chunking/>`_.
.. [LegalBenchRAGPaper] Pipitone and Houir Alami, `LegalBench-RAG
   <https://arxiv.org/abs/2408.10343>`_, 2024.
.. [LegalBenchRAGCode] LegalBench-RAG authors, `benchmark code and data
   <https://github.com/zeroentropy-ai/legalbenchrag>`_.
.. [Shaukat2026] Shaukat, Adnan and Kuhn, `A Systematic Investigation of
   Document Chunking Strategies and Embedding Sensitivity
   <https://arxiv.org/abs/2603.06976>`_, 2026 preprint.
.. [Prior2026] Prior, Milanova and Schultz, `Chunking German Legal Code
   <https://arxiv.org/abs/2605.19806>`_, 2026 preprint.
