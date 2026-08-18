"""Tests for the Mistral instrumentor."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("mistralai")

from mistralai.client import Mistral  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_http_response() -> httpx.Response:
    """Return a real httpx.Response mistralai will happily deserialize."""
    body = {
        "id": "cmpl-mistral-fake",
        "object": "chat.completion",
        "model": "mistral-large-latest",
        "created": 0,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Paris."},
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )


class TestMistralChat:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("mistralai", recording_client)
        try:
            client = Mistral(api_key="fake")
            with patch.object(client.chat, "do_request", return_value=_fake_http_response()):
                result = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": "capital of France?"}],
                    temperature=0.4,
                )
        finally:
            uninstrument("mistralai")

        assert result.choices[0].message.content == "Paris."

        span = find_span(recording_client, "mistral.chat.complete")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.REQUEST_MODEL] == "mistral-large-latest"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "mistral_ai"
        assert attrs[AgenticAttributes.REQUEST_TEMPERATURE] == 0.4
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.RESPONSE_ID] == "cmpl-mistral-fake"

        assert attrs[GenAIAttributes.SYSTEM] == "mistral_ai"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
