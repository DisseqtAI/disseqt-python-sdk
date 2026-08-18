"""Tests for the Cohere v2 instrumentor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("cohere")

import cohere  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_v2_response():
    """
    Cohere v2 response shape (mocked out; the real V2ChatResponse pydantic
    model validates fields we don't need here).
    """
    text_block = MagicMock(text="Paris.")
    message = MagicMock(role="assistant", content=[text_block])
    tokens = MagicMock(input_tokens=7, output_tokens=2)
    usage = MagicMock(tokens=tokens)
    response = MagicMock(
        id="cohere-fake",
        finish_reason="COMPLETE",
        message=message,
        usage=usage,
    )
    return response


class TestCohereChat:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("cohere", recording_client)
        try:
            client = cohere.ClientV2(api_key="fake")
            fake = _fake_v2_response()
            # V2Client.chat calls self._raw_client.chat(...).data — wrap in HttpResponse-shape.
            http_response = MagicMock(data=fake)
            with patch.object(client._raw_client, "chat", return_value=http_response, create=True):
                result = client.chat(
                    model="command-r-plus",
                    messages=[{"role": "user", "content": "capital of France?"}],
                    temperature=0.5,
                )
        finally:
            uninstrument("cohere")

        assert result.message.content[0].text == "Paris."

        span = find_span(recording_client, "cohere.chat")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.REQUEST_MODEL] == "command-r-plus"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "cohere"
        assert attrs[AgenticAttributes.REQUEST_TEMPERATURE] == 0.5
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.RESPONSE_ID] == "cohere-fake"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "COMPLETE"
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "Paris."}
        ]

        assert attrs[GenAIAttributes.SYSTEM] == "cohere"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == ["COMPLETE"]
