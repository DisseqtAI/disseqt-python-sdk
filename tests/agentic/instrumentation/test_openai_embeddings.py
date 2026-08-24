"""
Tests for the OpenAI embeddings instrumentor + shared canonical layer.

Covers the extended attribute set (dimensions_requested, encoding_format,
user, count, dimensions_actual) that Phase A of #16 introduced. Same
mocked-HTTP recipe as the chat tests.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402
from openai.types import CreateEmbeddingResponse, Embedding  # noqa: E402
from openai.types.create_embedding_response import Usage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.instrumentation._embeddings import (  # noqa: E402
    from_openai_request,
    from_openai_response,
)
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_embedding_response(vectors: list[list[float]]) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model="text-embedding-3-small",
        data=[
            Embedding(index=i, embedding=vec, object="embedding") for i, vec in enumerate(vectors)
        ],
        usage=Usage(prompt_tokens=8, total_tokens=8),
    )


class TestOpenAIEmbeddings:
    def test_captures_extended_request_and_response_attrs(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            # Two 1536-dim vectors — realistic gpt-3-small default.
            fake = _fake_embedding_response([[0.1] * 1536, [0.2] * 1536])
            with patch.object(client.embeddings, "_post", return_value=fake, create=True):
                client.embeddings.create(
                    model="text-embedding-3-small",
                    input=["hello", "world"],
                    dimensions=1536,
                    encoding_format="float",
                    user="u_test",
                )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.embeddings.create")
        attrs = json.loads(span.attributes_json)

        # Existing captures still work.
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "text-embedding-3-small"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "embeddings"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 8

        # New captures via canonical layer.
        assert attrs[AgenticAttributes.EMBEDDINGS_INPUT_COUNT] == 2
        assert attrs[AgenticAttributes.EMBEDDINGS_DIMENSIONS_REQUESTED] == 1536
        assert attrs[AgenticAttributes.EMBEDDINGS_ENCODING_FORMAT] == "float"
        assert attrs[AgenticAttributes.REQUEST_USER] == "u_test"
        assert attrs[AgenticAttributes.EMBEDDINGS_COUNT] == 2
        assert attrs[AgenticAttributes.EMBEDDINGS_DIMENSIONS_ACTUAL] == 1536

    def test_single_string_input_counts_as_one(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = _fake_embedding_response([[0.5] * 3])
            with patch.object(client.embeddings, "_post", return_value=fake, create=True):
                client.embeddings.create(
                    model="text-embedding-3-small",
                    input="just one string",
                )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.embeddings.create")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.EMBEDDINGS_INPUT_COUNT] == 1
        assert attrs[AgenticAttributes.EMBEDDINGS_COUNT] == 1
        assert attrs[AgenticAttributes.EMBEDDINGS_DIMENSIONS_ACTUAL] == 3


class TestCanonicalAdapters:
    """Direct tests of the adapters — provider-independent, no wrapper glue."""

    def test_from_openai_request_reads_all_fields(self):
        req = from_openai_request(
            {
                "model": "text-embedding-3-large",
                "input": ["a", "b", "c"],
                "dimensions": 3072,
                "encoding_format": "base64",
                "user": "u_1",
            }
        )
        assert req["model"] == "text-embedding-3-large"
        assert req["input_count"] == 3
        assert req["dimensions_requested"] == 3072
        assert req["encoding_format"] == "base64"
        assert req["user"] == "u_1"

    def test_from_openai_request_missing_optional_fields(self):
        req = from_openai_request({"model": "m", "input": "one"})
        assert req["input_count"] == 1
        assert req["dimensions_requested"] is None
        assert req["encoding_format"] is None
        assert req["user"] is None

    def test_from_openai_response_measures_actual_dimensions(self):
        resp = from_openai_response(_fake_embedding_response([[0.1] * 512, [0.2] * 512]))
        assert resp["count"] == 2
        assert resp["dimensions_actual"] == 512
        assert resp["input_tokens"] == 8
        assert resp["total_tokens"] == 8

    def test_from_openai_response_base64_leaves_actual_dims_none(self):
        # When encoding_format="base64" the embedding is a str, not a list.
        # Pydantic's Embedding model refuses str, so use a dict-shaped
        # response — the adapter's read() tolerates both.
        resp = from_openai_response(
            {
                "model": "text-embedding-3-small",
                "data": [{"embedding": "Zm9vYmFy" * 100, "index": 0}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        )
        assert resp["count"] == 1
        assert resp["dimensions_actual"] is None
