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

        # TP-2128 round-2 P2 #2.1: Mistral streams via a separate
        # method (Chat.stream) so set_common_chat_request can't read
        # stream= from kwargs. Non-streaming path must still emit
        # gen_ai.request.is_stream=False so dashboards filtering on
        # this attribute don't misclassify Mistral spans.
        assert attrs[GenAIAttributes.REQUEST_IS_STREAM] is False

    def test_stream_span_sets_is_stream_true(self, recording_client):
        """
        TP-2128 round-2 P2 #2.1: Mistral Chat.stream must set
        gen_ai.request.is_stream=True — mirrors the Cohere fix from
        Appendix A.3.

        Direct-invoke the wrapper factory (rather than round-tripping
        through client.chat.stream) so the test doesn't have to fake
        the mistralai EventStream / httpx layers. Verifies the specific
        contract we care about: the span attribute gets stamped.
        """
        from types import SimpleNamespace

        from disseqt_agentic_sdk.instrumentation.mistral.instrumentor import (
            MistralInstrumentor,
            _sync_stream,
        )

        completion_event = SimpleNamespace(
            data=SimpleNamespace(
                id="cmpl-mistral-stream",
                model="mistral-large-latest",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )
        )

        instrumentor = MistralInstrumentor()
        instrumentor._client = recording_client
        wrapper_fn = _sync_stream(instrumentor)

        stream = wrapper_fn(
            wrapped=lambda *a, **kw: iter([completion_event]),
            instance=None,
            args=(),
            kwargs={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        # Drain so on_finish → finalize → span end fires.
        list(stream)

        span = find_span(recording_client, "mistral.chat.stream")
        attrs = json.loads(span.attributes_json)
        assert attrs[GenAIAttributes.REQUEST_IS_STREAM] is True
