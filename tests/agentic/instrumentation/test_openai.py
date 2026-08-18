"""
End-to-end tests for the OpenAI instrumentor.

Patches openai's underlying HTTP path so tests never hit the network,
then invokes `openai.chat.completions.create(...)` through the real SDK.
The instrumentation should emit exactly one DisseqtSpan into the
recording buffer, populated with both `agentic.*` and `gen_ai.*` attrs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from openai import OpenAI  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion import Choice  # noqa: E402
from openai.types.completion_usage import CompletionUsage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-fake",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Paris."),
            )
        ],
        usage=CompletionUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
    )


class TestOpenAIChat:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = _fake_chat_completion()
            with patch.object(client.chat.completions, "_post", return_value=fake, create=True):
                # openai's Completions.create calls self._post under the hood.
                result = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "capital of France?"}],
                    temperature=0.2,
                )
        finally:
            uninstrument("openai")

        # Round-trip returned the fake response unchanged.
        assert result.choices[0].message.content == "Paris."

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        # agentic.* — our native names
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "openai"
        assert attrs[AgenticAttributes.REQUEST_TEMPERATURE] == 0.2
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.USAGE_TOTAL_TOKENS] == 9
        assert attrs[AgenticAttributes.RESPONSE_ID] == "chatcmpl-fake"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "stop"

        # gen_ai.* — dual-emitted for OTel-compatible tooling
        assert attrs[GenAIAttributes.SYSTEM] == "openai"
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[GenAIAttributes.USAGE_TOTAL_TOKENS] == 9
        assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == ["stop"]

    def test_records_error_on_exception(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with patch.object(
                client.chat.completions, "_post", side_effect=RuntimeError("boom"), create=True
            ):
                with pytest.raises(RuntimeError, match="boom"):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "x"}],
                    )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        assert span.status_code == "ERROR"
        attrs = json.loads(span.attributes_json)
        assert attrs.get(AgenticAttributes.ERROR_MESSAGE) == "boom"
        assert attrs.get(AgenticAttributes.ERROR_TYPE) == "RuntimeError"

    def test_streaming_aggregates_deltas(self, recording_client):
        """Streaming: token deltas should accumulate into output_messages."""
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            # Build a fake stream: three text deltas + a final chunk with usage.
            def _chunk(text=None, finish=None, usage=None, has_choice=True):
                ch = MagicMock()
                ch.id = "chatcmpl-stream"
                ch.model = "gpt-4o-mini"
                ch.usage = usage
                if has_choice:
                    choice = MagicMock()
                    choice.delta = MagicMock(role="assistant", content=text)
                    choice.finish_reason = finish
                    ch.choices = [choice]
                else:
                    ch.choices = []
                return ch

            usage_obj = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            fake_stream = iter(
                [
                    _chunk(text="Par"),
                    _chunk(text="is"),
                    _chunk(text=".", finish="stop"),
                    _chunk(usage=usage_obj, has_choice=False),
                ]
            )
            with patch.object(
                client.chat.completions, "_post", return_value=fake_stream, create=True
            ):
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "x"}],
                    stream=True,
                )
                chunks = list(stream)
        finally:
            uninstrument("openai")

        assert len(chunks) == 4
        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        outputs = attrs[AgenticAttributes.OUTPUT_MESSAGES]
        assert outputs == [{"role": "assistant", "content": "Paris."}]
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "stop"
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 3


class TestOpenAIParentLinkage:
    def test_auto_span_nests_under_user_trace(self, recording_client):
        """If the user has opened a trace, the auto-span parents to it."""
        from disseqt_agentic_sdk import SpanKind, start_trace

        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = _fake_chat_completion()
            with start_trace(recording_client, name="user_trace") as trace:
                with trace.start_span("agent", SpanKind.AGENT_EXEC) as agent_span:
                    parent_span_id = agent_span.span_id
                    trace_id = trace.trace_id
                    with patch.object(
                        client.chat.completions, "_post", return_value=fake, create=True
                    ):
                        client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "x"}],
                        )
        finally:
            uninstrument("openai")

        auto_span = find_span(recording_client, "openai.chat.completions.create")
        assert auto_span.trace_id == trace_id
        assert auto_span.parent_span_id == parent_span_id
        assert auto_span.root is False
