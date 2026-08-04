.. _lossless_segmentation_api:

Lossless, structure-first segmentation
======================================

LexNLP 2.4.0a1 adds an experimental English legal-document hierarchy,
strict-budget retrieval chunks and a final embedding-payload contract without
changing the legacy segmentation functions.  The implemented v1 guarantees:

* absolute half-open spans into one unchanged Python string;
* exact reconstruction at every hierarchy level;
* strict character or explicitly counted token budgets;
* versioned hierarchy/chunk manifests and digest-qualified identities; and
* exact final token validation after carried heading or table context is added.

These are source-fidelity and reproducibility guarantees.  They do not prove
that every legal boundary is correct or that this configuration is state of
the art.  The bundled fixtures are small regression evidence; promotion
requires the held-out method in :ref:`segmentation_v2_design`.

Hierarchy API
-------------

::

    from lexnlp.nlp.en.segments import SegmentKind, segment_document

    text = """MASTER AGREEMENT\r\n\r\n1. Services\r\nThe Supplier shall perform.\r\n"""
    hierarchy = segment_document(text)

    assert hierarchy.reconstruct() == text
    for sentence in hierarchy.segments(SegmentKind.SENTENCE):
        assert sentence.text(text) == text[sentence.start:sentence.end]

`DocumentHierarchy.source` is the exact input.  Offsets are Python string
indices, not UTF-8 byte offsets, grapheme indices, PDF coordinates or OCR
boxes.  Every parent's children are ordered, contiguous and cover the parent
exactly.  Non-document nodes are non-empty and realised `segment_id` values
(`kind:start:end`) are unique within a validated hierarchy.

The immutable core types are:

:class:`~lexnlp.nlp.en.segments.hierarchy.Segment`
    A `kind`, `start`, `end`, optional string `label`, optional integer `level`,
    string-pair `attributes` and complete child partition.

:class:`~lexnlp.nlp.en.segments.hierarchy.StructuralSpan`
    A caller-supplied section, clause, list-item or table range.  Supplied
    ranges must be disjoint or properly nested.

:class:`~lexnlp.nlp.en.segments.hierarchy.HierarchyManifest`
    The structural, paragraph and sentence backend identities, structure
    profile, composition mode, schema version and `tree_sha256` of the complete
    realised tree, including labels, levels and attributes.

:class:`~lexnlp.nlp.en.segments.hierarchy.DocumentHierarchy`
    The validated root, exact source and manifest.  `segments()` walks in
    deterministic pre-order and `leaves()` returns the exact leaf partition.

Structure profiles
~~~~~~~~~~~~~~~~~~

`StructureProfile.CONSERVATIVE` is the default.  It recognises high-confidence
legal headings/outlines but avoids promoting an ordinary mixed-case numbered
sequence merely because the numbers increase.

`StructureProfile.STATUTE` opts into more assertive numbered statutory-heading
and contextual nesting rules::

    from lexnlp.nlp.en.segments import StructureProfile

    hierarchy = segment_document(
        text,
        structure_profile=StructureProfile.STATUTE,
    )

Select the statute profile only when the genre supports that assumption.
Neither profile covers every drafting style, and `level` is detected outline
context rather than legal interpretation.

Composing boundary evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With `structural_spans`, `StructuralMode.REPLACE` is the default.
`StructuralMode.AUGMENT` combines caller spans with the selected built-in
profile and records a composite detector identity::

    from lexnlp.nlp.en.segments import (
        SegmentKind,
        StructuralMode,
        StructuralSpan,
        segment_document,
    )

    table_start = text.index("Item | Price")
    hierarchy = segment_document(
        text,
        structural_spans=(
            StructuralSpan(
                SegmentKind.TABLE,
                table_start,
                len(text),
                label="Pricing table",
                attributes=(("page", "4"), ("adapter", "example-v1")),
            ),
        ),
        structural_mode=StructuralMode.AUGMENT,
        structural_backend_id="example-layout-adapter@1",
    )

`paragraph_segmenter` receives one structural region at a time; the selected
`sentence_segmenter` receives each paragraph.  Both yield relative
`(start, end)` or `(start, end, text)` tuples.  Returned spans must be ordered,
non-overlapping, in range and, for three-part results, equal the exact source
slice.  Uncovered characters remain `TEXT` or `SEPARATOR`.  A custom backend
returning no spans supplies no semantic evidence, so LexNLP never fabricates a
paragraph or sentence.

If no sentence callback is supplied, the existing legal-tuned Punkt path is
loaded lazily.  Missing declared model data raises normally rather than
silently selecting another algorithm.  Model-dependent callbacks should
declare stable `sentence_backend_id`, `paragraph_backend_id` and
`structural_backend_id` values.  Callable-derived IDs are convenient but are
not adequate model provenance for a reproducible benchmark.

Optional Segment Any Text backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`SaTSentenceSegmenter` adapts an already-instantiated Segment Any Text model::

    from lexnlp.nlp.en.segments import SaTSentenceSegmenter

    sat_segmenter = SaTSentenceSegmenter(
        make_pinned_sat_model(),
        backend_id="segment-any-text/sat-3l-sm@pinned-revision/split-v1",
        split_kwargs={"threshold": 0.4},
    )
    hierarchy = segment_document(text, sentence_segmenter=sat_segmenter)

LexNLP does not import `wtpsplit`, resolve model names, download weights or add
a mandatory dependency.  The adapter defaults `split_on_input_newlines` to
`False` and rejects output which is not an exact consecutive partition.  It is
invoked per paragraph, so it does not expose the upstream multi-document
batching path.  SaT is an opt-in challenger, not a recommended default; it
needs a pinned held-out comparison on the intended legal corpus.

Chunking API
------------

`chunk_document` accepts source text or an existing hierarchy.
`iter_chunks` emits the same immutable chunks lazily, although v1 still
materialises the source hierarchy.

Character mode
~~~~~~~~~~~~~~

::

    from lexnlp.nlp.en.segments import chunk_document, reconstruct_chunks

    chunks = chunk_document(
        hierarchy,
        max_chars=1_000,
        overlap_chars=100,
        document_id="contract-42/revision-7",
    )
    assert all(chunk.char_count <= 1_000 for chunk in chunks)
    assert reconstruct_chunks(chunks) == text

If neither budget is set, `DEFAULT_MAX_CHARS == 4000` is used as an upper
guardrail, not an optimal legal-retrieval size.  Empty source returns no chunks;
an empty iterable therefore carries no manifest evidence and
`reconstruct_chunks([]) == ""`.

Token mode
~~~~~~~~~~

Token mode requires the downstream counter and a stable counter identity::

    def embedding_token_count(value: str) -> int:
        return len(pinned_tokeniser.encode(value, add_special_tokens=False))

    chunks = chunk_document(
        hierarchy,
        max_tokens=480,
        token_counter=embedding_token_count,
        token_counter_id="embedding-tokeniser@revision/no-special-tokens",
        overlap_tokens=32,
    )

The dependency-free `count_tokens` helper is available only when a caller
deliberately selects its lexical word/punctuation units.  It is never an
implicit model-token fallback.  Character mode rejects a token counter or ID.

Every accepted slice is independently budget-verified.  A non-monotonic
black-box counter can make packing conservative but cannot make a returned
chunk exceed its reported budget.  If every non-empty prefix is over budget,
packing raises.

Container and overlap policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With `respect_boundaries=True`, `ContainerPolicy.PRESERVE` keeps adjacent
complete sections, clauses and tables separate; related list items may pack
within a common list/clause.  `PACK_SIBLINGS` opts into dense sibling packing.
An oversized container descends through finer boundaries and finally hard
splits to preserve the strict budget.

Overlap is inside the budget and is a maximum.  It may be reduced or dropped
when retaining it would split a container which otherwise fits.  Container
policy is intentionally irrelevant when `respect_boundaries=False` disables
structural alignment.

Manifests, provenance and identity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ChunkingManifest` records the source digest/document namespace, hierarchy
manifest, unit/budget/overlap, boundary/container policy, schema/serialiser
versions and token-counter ID.  Canonical JSON produces `manifest_id`.

Each `DocumentChunk` records:

* exact `start`, `end`, `text` and `text_sha256`;
* `new_content_start` separating repeated overlap from fresh source;
* full `provenance`, fresh `content_provenance` and
  `overlap_provenance`;
* `provenance_sha256` over all ordered references and partitions;
* `unit_count` and the shared manifest; and
* `chunk_metadata_sha256` over index, offsets, count, text/provenance digests
  and manifest identity.

`chunk_id` is qualified by source identity, manifest identity and the canonical
chunk-metadata digest.  A changed source, tree, packing configuration, slice or
provenance changes the ID.  `reconstruct_chunks` checks zero-based consecutive
indices, one manifest, continuous fresh-content offsets and the final source
digest.  Constructor validation ensures corrupted overlap or provenance cannot
retain the original chunk identity.  Serialisers can use the public
`compute_provenance_sha256` and `compute_chunk_metadata_sha256` helpers.

Final embedding payload
-----------------------

A chunk budget covers `chunk.text`.  Use `render_embedding_payload` when the
actual embedding string also carries headings or a repeated table header::

    from lexnlp.nlp.en.segments import (
        ContextFragment,
        PayloadBudgetExceeded,
        SegmentKind,
        render_embedding_payload,
    )

    table = next(
        ref for ref in chunks[0].content_provenance.segments
        if ref.kind is SegmentKind.TABLE
    )
    try:
        payload = render_embedding_payload(
            chunks[0],
            token_counter=embedding_token_count,
            tokenizer_id="embedding-tokeniser@revision/special-token-policy",
            max_tokens=512,
            context_fragments=(
                ContextFragment("table_header", "Item | Price", table.segment_id),
            ),
        )
    except PayloadBudgetExceeded:
        # Reserve headroom or rechunk; v1 never truncates.
        raise

    send_to_embedding_model(payload.text)

`lexnlp.embedding_payload.v1` serialises ordered context lines, one blank line,
then the exact chunk; without context it is exactly `chunk.text`.  Automatic
context includes only labelled section/clause/table ancestors which begin
before and enclose the fresh slice.  Explicit `heading` and `table_header`
fragments must be owned by an appropriate `content_provenance` reference.

`EmbeddingPayload` records the exact source/final text, final digest, ordered
context, tokeniser ID, maximum and observed counts.  Canonical
`payload_metadata_sha256` binds the chunk ID, serialisation version, source and
final-text digests, every ordered context role/text/owner ID, tokeniser ID and
maximum/observed counts.  `payload_id` is the chunk-qualified metadata digest.

The renderer never trims or repacks.  Overflow raises `PayloadBudgetExceeded`
with actual and maximum counts.  Any instruction/prefix added after rendering
is outside the validated payload; the guarantee applies only when
`payload.text` is sent unchanged.

Implemented boundary
--------------------

* Legacy page/paragraph/section/sentence/title APIs remain unchanged.
* v1 nodes and chunks are contiguous Python-string ranges.  They cannot express
  discontiguous layout regions, PDF geometry, table cells, OCR confidence,
  footnotes or cross-reference graphs.
* Caller-supplied laminar `TABLE` spans are supported, but no Docling/PDF
  adapter is implemented.
* NUPunkt, CharBoundary and Docling are benchmark candidates, not installed
  backends or recommendations.
* A multilingual sentence backend does not make the English structure profiles
  multilingual.
* The iterator is not an end-to-end streaming parser.
* Exact spans and identities do not establish structural/retrieval quality.

API reference
-------------

.. automodule:: lexnlp.nlp.en.segments.hierarchy
   :members: SegmentKind, StructureProfile, StructuralMode, Segment, StructuralSpan, HierarchyManifest, DocumentHierarchy, segment_document, iter_document_segments

.. automodule:: lexnlp.nlp.en.segments.backends
   :members: SaTSentenceSegmenter, legacy_sentence_segmenter, parts_to_spans

.. automodule:: lexnlp.nlp.en.segments.chunks
   :members: ContainerPolicy, ChunkingManifest, SegmentReference, ChunkProvenance, DocumentChunk, count_tokens, compute_provenance_sha256, compute_chunk_metadata_sha256, iter_chunks, chunk_document, reconstruct_chunks

.. automodule:: lexnlp.nlp.en.segments.payloads
   :members: ContextFragment, EmbeddingPayload, PayloadBudgetExceeded, render_embedding_payload
