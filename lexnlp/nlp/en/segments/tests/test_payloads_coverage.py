"""Coverage tests for lexnlp.nlp.en.segments.payloads validation and error paths."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from lexnlp.nlp.en.segments.chunks import DocumentChunk, chunk_document
from lexnlp.nlp.en.segments.hierarchy import SegmentKind
from lexnlp.nlp.en.segments.payloads import (
    EMBEDDING_PAYLOAD_SERIALIZATION_VERSION,
    ContextFragment,
    EmbeddingPayload,
    PayloadBudgetExceeded,
    render_embedding_payload,
)


def _whole_span(text: str):
    if text:
        yield 0, len(text), text


def _chunk(text: str = "Hello world payload test") -> DocumentChunk:
    return chunk_document(text, max_chars=100, sentence_segmenter=_whole_span)[0]


def _valid_payload(text: str = "Hello world payload test", **kwargs) -> EmbeddingPayload:
    chunk = _chunk(text)
    args = {"token_counter": len, "tokenizer_id": "tests.tok.v1", "max_tokens": len(text)}
    args.update(kwargs)
    return render_embedding_payload(chunk, **args)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


class TestGuards:
    def test_blank_text_fragment_rejected(self) -> None:
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            ContextFragment("heading", "   ", "section:0:4")

    def test_non_string_text_fragment_rejected(self) -> None:
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            ContextFragment("heading", 123, "section:0:4")  # type: ignore[arg-type]

    def test_blank_tokenizer_id_rejected(self) -> None:
        chunk = _chunk()
        with pytest.raises(ValueError, match="tokenizer_id must be a non-empty string"):
            render_embedding_payload(chunk, token_counter=len, tokenizer_id="  ", max_tokens=100)

    def test_non_positive_max_tokens_rejected(self) -> None:
        chunk = _chunk()
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            render_embedding_payload(chunk, token_counter=len, tokenizer_id="tests.tok.v1", max_tokens=0)

    def test_bool_max_tokens_rejected(self) -> None:
        chunk = _chunk()
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            render_embedding_payload(chunk, token_counter=len, tokenizer_id="tests.tok.v1", max_tokens=True)


class TestEmbeddingPayloadValidation:
    def test_wrong_serialization_version_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="serialization_version must be"):
            dataclasses.replace(payload, serialization_version="bogus.v9")
        assert payload.serialization_version == EMBEDDING_PAYLOAD_SERIALIZATION_VERSION

    def test_negative_token_count_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="token_count must be a non-negative integer"):
            dataclasses.replace(payload, token_count=-1)

    def test_bool_token_count_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="token_count must be a non-negative integer"):
            dataclasses.replace(payload, token_count=True)

    def test_token_count_above_max_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="token_count cannot exceed max_tokens"):
            dataclasses.replace(payload, token_count=payload.max_tokens + 1)

    def test_non_fragment_context_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(TypeError, match="must contain ContextFragment values"):
            dataclasses.replace(payload, context_fragments=("nope",))

    def test_non_string_source_text_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(TypeError, match="source_text and text must be strings"):
            dataclasses.replace(payload, source_text=123)

    def test_malformed_sha_rejected(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="text_sha256 must be a lowercase SHA-256 digest"):
            dataclasses.replace(payload, text_sha256="not-a-digest")

    def test_text_must_match_serialization(self) -> None:
        payload = _valid_payload()
        with pytest.raises(ValueError, match="versioned payload serialization"):
            dataclasses.replace(payload, text="something else entirely", text_sha256="a" * 64)

    def test_sha_must_match_exact_text(self) -> None:
        payload = _valid_payload()
        wrong = "0" * 64
        assert wrong != _digest(payload.text)
        with pytest.raises(ValueError, match="text_sha256 does not match"):
            dataclasses.replace(payload, text_sha256=wrong)

    def test_valid_payload_identity(self) -> None:
        payload = _valid_payload()
        assert payload.text == payload.source_text
        assert payload.text_sha256 == _digest(payload.text)
        assert len(payload.payload_metadata_sha256) == 64
        assert payload.payload_id.startswith(f"{payload.chunk_id}:payload:sha256:")


class TestRenderGuards:
    def test_non_chunk_rejected(self) -> None:
        with pytest.raises(TypeError, match="chunk must be a DocumentChunk"):
            render_embedding_payload(
                "nope",
                token_counter=len,
                tokenizer_id="t",
                max_tokens=10,  # type: ignore[arg-type]
            )

    def test_non_callable_counter_rejected(self) -> None:
        with pytest.raises(TypeError, match="token_counter must be callable"):
            render_embedding_payload(
                _chunk(),
                token_counter="len",
                tokenizer_id="t",
                max_tokens=100,  # type: ignore[arg-type]
            )

    def test_non_bool_ancestry_flag_rejected(self) -> None:
        with pytest.raises(TypeError, match="include_ancestry_labels must be a boolean"):
            render_embedding_payload(
                _chunk(),
                token_counter=len,
                tokenizer_id="t",
                max_tokens=100,
                include_ancestry_labels="yes",  # type: ignore[arg-type]
            )

    def test_non_fragment_supplied_context_rejected(self) -> None:
        with pytest.raises(TypeError, match="must contain ContextFragment values"):
            render_embedding_payload(
                _chunk(),
                token_counter=len,
                tokenizer_id="t",
                max_tokens=100,
                context_fragments=("nope",),  # type: ignore[list-item]
                include_ancestry_labels=False,
            )

    def test_heading_owned_by_paragraph_rejected(self) -> None:
        chunk = _chunk()
        paragraph = next(ref for ref in chunk.content_provenance.segments if ref.kind is SegmentKind.PARAGRAPH)
        fragment = ContextFragment("heading", "Header", paragraph.segment_id)
        with pytest.raises(ValueError, match="heading context must be owned"):
            render_embedding_payload(
                chunk,
                token_counter=len,
                tokenizer_id="t",
                max_tokens=100,
                context_fragments=(fragment,),
                include_ancestry_labels=False,
            )

    def test_counter_string_result_rejected(self) -> None:
        with pytest.raises(TypeError, match="token_counter must return an integer"):
            render_embedding_payload(
                _chunk(),
                token_counter=lambda text: "5",  # type: ignore[return-value]
                tokenizer_id="t",
                max_tokens=100,
            )

    def test_counter_bool_result_rejected(self) -> None:
        with pytest.raises(TypeError, match="token_counter must return an integer"):
            render_embedding_payload(
                _chunk(),
                token_counter=lambda text: True,  # type: ignore[return-value]
                tokenizer_id="t",
                max_tokens=100,
            )

    def test_counter_negative_result_rejected(self) -> None:
        with pytest.raises(ValueError, match="token_counter cannot return a negative value"):
            render_embedding_payload(_chunk(), token_counter=lambda text: -1, tokenizer_id="t", max_tokens=100)

    def test_budget_overflow_raises_with_evidence(self) -> None:
        chunk = _chunk()
        with pytest.raises(PayloadBudgetExceeded) as caught:
            render_embedding_payload(chunk, token_counter=len, tokenizer_id="t", max_tokens=1)
        assert caught.value.actual == len(chunk.text)
        assert caught.value.maximum == 1
        assert caught.value.chunk_id == chunk.chunk_id
