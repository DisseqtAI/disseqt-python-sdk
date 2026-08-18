"""Tests for the Groq instrumentor (OpenAI-shape SDK)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("groq")

from groq import Groq  # noqa: E402
from groq.types.chat.chat_completion import (  # noqa: E402
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from groq.types.completion_usage import CompletionUsage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-groq-fake",
        model="llama-3.1-70b-versatile",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                logprobs=None,
                message=ChatCompletionMessage(role="assistant", content="Paris."),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=7,
            completion_tokens=2,
            total_tokens=9,
            prompt_time=0.0,
            completion_time=0.0,
            queue_time=0.0,
            total_time=0.0,
        ),
    )


class TestGroqChat:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("groq", recording_client)
        try:
            client = Groq(api_key="fake")
            fake = _fake_chat_completion()
            with patch.object(client.chat.completions, "_post", return_value=fake, create=True):
                result = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[{"role": "user", "content": "capital of France?"}],
                    temperature=0.3,
                )
        finally:
            uninstrument("groq")

        assert result.choices[0].message.content == "Paris."

        span = find_span(recording_client, "groq.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        assert attrs[AgenticAttributes.REQUEST_MODEL] == "llama-3.1-70b-versatile"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "groq"
        assert attrs[AgenticAttributes.REQUEST_TEMPERATURE] == 0.3
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.RESPONSE_ID] == "chatcmpl-groq-fake"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "stop"

        assert attrs[GenAIAttributes.SYSTEM] == "groq"
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "llama-3.1-70b-versatile"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == ["stop"]

    def test_records_error_on_exception(self, recording_client):
        instrument("groq", recording_client)
        try:
            client = Groq(api_key="fake")
            with patch.object(
                client.chat.completions, "_post", side_effect=RuntimeError("boom"), create=True
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=[{"role": "user", "content": "x"}],
                    )
        finally:
            uninstrument("groq")

        span = find_span(recording_client, "groq.chat.completions.create")
        assert span.status_code == "ERROR"
