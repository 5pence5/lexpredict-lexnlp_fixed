"""Coverage tests for :mod:`lexnlp.nlp.en.segments.sentences` missing lines."""

import random

from nltk.tokenize.punkt import PunktSentenceTokenizer

from lexnlp.nlp.en.segments.sentences import (
    _trim_span,
    build_sentence_model,
    pre_process_document,
)


def test_pre_process_document_falsy_returns_input() -> None:
    assert pre_process_document("") == ""
    assert pre_process_document(None) is None  # type: ignore[arg-type]


def test_pre_process_document_strips_page_number_line() -> None:
    assert pre_process_document("hello\n123\nworld") == "hello\n\nworld"
    assert pre_process_document("hello world") == "hello world"


def test_trim_span_none_without_content() -> None:
    assert _trim_span("   ", (0, 3)) is None
    assert _trim_span("", (5, 5)) is None


def test_trim_span_pins_offsets() -> None:
    assert _trim_span("  hi  ", (10, 16)) == (12, 14)


def _training_corpus() -> str:
    sents = [
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore every morning",
        " Elm Blvd office opens at nine",
        "Mr Johnson arrived late yesterday",
        "The contract was signed in Austin",
        "Dogs bark loudly during thunderstorms",
        "Rain fell steadily across the valley",
        "He reads novels every evening",
        "The engineer fixed the broken bridge",
        "Birds migrate south before winter starts",
    ]
    rng = random.Random(7)
    return " ".join(rng.choice(sents) + "." for _ in range(400))


def test_build_sentence_model_with_extra_abbrevs() -> None:
    model = build_sentence_model(_training_corpus(), extra_abbrevs=["Blvd.", "Prof."])
    assert isinstance(model, PunktSentenceTokenizer)
    # "prof" never occurs in the training text, so its presence proves the
    # extra_abbrevs loop ran.
    assert "blvd" in model._params.abbrev_types
    assert "prof" in model._params.abbrev_types
    # The injected abbreviation blocks a sentence split after "Blvd.".
    assert model.tokenize("The office is on Elm Blvd. It opened in 1999.") == [
        "The office is on Elm Blvd. It opened in 1999."
    ]


def test_build_sentence_model_without_extra_abbrevs() -> None:
    model = build_sentence_model(_training_corpus())
    assert isinstance(model, PunktSentenceTokenizer)
    assert model.tokenize("Dogs bark loudly. Rain fell steadily.") == [
        "Dogs bark loudly.",
        "Rain fell steadily.",
    ]
