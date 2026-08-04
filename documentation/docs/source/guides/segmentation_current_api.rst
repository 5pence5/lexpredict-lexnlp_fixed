.. _segmentation_current_api:

Working with the compatibility segmentation APIs
================================================

The legacy LexNLP 2.3-compatible surface provides English page, paragraph,
section, sentence and title segmentation under :mod:`lexnlp.nlp.en.segments`.
These functions pre-date a general retrieval-chunking abstraction: they answer
different questions, return different shapes and do not all expose source
coordinates.

This page documents those compatibility contracts.  The additive experimental
2.4.0a1 hierarchy, strict-budget chunker and embedding-payload API are separate;
see :ref:`lossless_segmentation_api`.  Their evidence and promotion method are
in :ref:`segmentation_v2_design`.

Capability matrix
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 20 18 20 30

   * - Unit
     - Primary API
     - Coordinates
     - Boundary method
     - Important qualification
   * - Sentence
     - `get_sentence_span`
     - `(start, end, text)`
     - Legal-tuned Punkt plus LexNLP post-processing
     - English only; outer whitespace and OCR-only fragments can be omitted.
   * - Paragraph
     - `get_paragraph_spans`
     - `(start, end, text)`
     - Bundled line-window classifier
     - A recognised short-document feature mismatch falls back to one span.
   * - Section
     - `get_section_spans`
     - `DocumentSection`
     - Bundled classifier or regular expressions
     - Some coordinates are recovered from segmented strings, not the separate
       lossless-tree contract.
   * - Page
     - `get_pages`
     - Text only
     - Bundled line-window classifier
     - Input lines are rejoined with `\n`.
   * - Title
     - `get_titles`
     - Text only
     - Bundled line-window classifier
     - Returns detected title text rather than its source span.

Sentence spans
--------------

::

    from lexnlp.nlp.en.segments.sentences import get_sentence_span

    text = "Section 2. U.S. law applies."
    for start, end, sentence in get_sentence_span(text):
        assert sentence == text[start:end]

Offsets are half-open Python string indices, not UTF-8 bytes or PDF
coordinates.  Boundary detection uses a small normalised view but returned
text is sliced from the original string.  Post-processing trims outer
whitespace and filters some OCR-like fragments, so the spans are not a
lossless partition of every character.

`pre_process_document` removes page markers and other lines.  If it is called
first, later coordinates refer to that processed string rather than the
original input.

Optional Segment Any Text adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The additive `SaTSentenceSegmenter` can be used with the new hierarchy, but is
not the compatibility default::

    from lexnlp.nlp.en.segments import SaTSentenceSegmenter

    segmenter = SaTSentenceSegmenter(
        make_pinned_sat_model(),
        backend_id="segment-any-text/sat-3l-sm@pinned-revision/split-v1",
    )
    spans = list(segmenter(text))
    assert "".join(part for _, _, part in spans) == text

LexNLP neither imports `wtpsplit` nor downloads weights.  The adapter requires
a caller-pinned backend identity, forwards explicit `split_kwargs`, defaults
`split_on_input_newlines` to `False` and accepts only an exact consecutive
partition.  The hierarchy invokes it per paragraph, not through upstream
multi-document batching.  It is an unpromoted challenger until a held-out legal
comparison supports a declared use case.

Paragraph spans
---------------

::

    from lexnlp.nlp.en.segments.paragraphs import get_paragraph_spans

    text = "First paragraph.\r\n\r\nSecond paragraph."
    for start, end, paragraph in get_paragraph_spans(text):
        assert paragraph == text[start:end]

Original line endings within each returned slice are retained.  A document
with no predicted break is returned as one span.  The same result is used as a
compatibility fallback for a recognised feature-count mismatch, so the return
type cannot distinguish a confident one-paragraph prediction from that
fallback.

Sections
--------

::

    from lexnlp.nlp.en.segments.sections import get_section_spans

    for section in get_section_spans(text, use_ml=False, safe_failure=False):
        print(section.start, section.end, section.title, section.level)

`use_ml=False` selects the regular-expression detector.  `level` is a relative
scan-time hint and `abs_level` identifies the first matching parser rule; they
are not a validated clause hierarchy.

The page and machine-learning section paths split lines and rejoin with `\n`,
so they can normalise CRLF and other endings.  Section spans then locate those
pieces sequentially in the original input.  Do not use these coordinates as a
universal provenance layer for mixed line endings, repeated text, PDF layout
or OCR.

Unlike paragraph segmentation, `get_pages` and the machine-learning
`get_sections` path can yield no result when no break is predicted.  Callers
must decide whether that means one unsplit unit in their application.

Failure behaviour
-----------------

`get_sections`, `get_sections_re`, `get_section_spans` and `get_titles` use
LexNLP's `safe_failure` wrapper.  Ordinary exceptions are suppressed by
default, after which iteration ends.  During data-quality checks or migrations,
pass `safe_failure=False` so malformed input or model-schema problems remain
observable.  This keyword is consumed by the wrapper.

Choosing an API
---------------

* Use legacy sentence/paragraph spans for compatibility annotations and verify
  `text[start:end] == returned_text`.
* Treat `DocumentSection` as a best-effort English structural hint.
* Preserve page information from the extraction layer; `get_pages` cannot
  recover geometry or reading order.
* Apply downstream limits when using legacy functions.  They do not guarantee
  a token budget, overlap or retrieval chunk size.
* Use :ref:`lossless_segmentation_api` when exact coverage, hierarchy manifests,
  strict budgets or authenticated final embedding payloads are required.
* Freeze existing outputs before changing a legacy model or threshold.

The additive alpha API does not retroactively alter these return shapes or
failure behaviours.
