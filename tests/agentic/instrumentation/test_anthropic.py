"""Tests for the Anthropic instrumentor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from anthropic import Anthropic  # noqa: E402
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage  # noqa: E402

from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _fake_message() -> Message:
    return Message(
        id="msg-fake",
        type="message",
        role="assistant",
        model="claude-3-5-haiku-latest",
        content=[TextBlock(type="text", text="Paris.")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=7, output_tokens=2),
    )


class TestAnthropicMessages:
    def test_records_span_with_dual_attrs(self, recording_client):
        instrument("anthropic", recording_client)
        try:
            client = Anthropic(api_key="fake")
            fake = _fake_message()
            with patch.object(client.messages, "_post", return_value=fake, create=True):
                result = client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=64,
                    messages=[{"role": "user", "content": "capital of France?"}],
                )
        finally:
            uninstrument("anthropic")

        assert result.content[0].text == "Paris."

        span = find_span(recording_client, "anthropic.messages.create")
        attrs = json.loads(span.attributes_json)

        # agentic.*
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "claude-3-5-haiku-latest"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "anthropic"
        assert attrs[AgenticAttributes.REQUEST_MAX_TOKENS] == 64
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.USAGE_TOTAL_TOKENS] == 9
        assert attrs[AgenticAttributes.RESPONSE_ID] == "msg-fake"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "end_turn"
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "Paris."}
        ]

        # gen_ai.*
        assert attrs[GenAIAttributes.SYSTEM] == "anthropic"
        assert attrs[GenAIAttributes.REQUEST_MODEL] == "claude-3-5-haiku-latest"
        assert attrs[GenAIAttributes.OPERATION_NAME] == "chat"
        assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == ["end_turn"]

    def test_streaming_aggregates_events(self, recording_client):
        instrument("anthropic", recording_client)
        try:
            client = Anthropic(api_key="fake")

            # Fake stream event sequence.
            def _evt(**fields):
                m = MagicMock()
                for k, v in fields.items():
                    setattr(m, k, v)
                return m

            message_start = _evt(
                type="message_start",
                message=_evt(
                    id="msg-stream",
                    model="claude-3-5-haiku-latest",
                    usage=_evt(input_tokens=5, output_tokens=0),
                ),
            )
            delta1 = _evt(
                type="content_block_delta",
                delta=_evt(type="text_delta", text="Par"),
            )
            delta2 = _evt(
                type="content_block_delta",
                delta=_evt(type="text_delta", text="is."),
            )
            message_delta = _evt(
                type="message_delta",
                delta=_evt(stop_reason="end_turn"),
                usage=_evt(output_tokens=3),
            )
            fake_stream = iter([message_start, delta1, delta2, message_delta])

            with patch.object(client.messages, "_post", return_value=fake_stream, create=True):
                stream = client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=64,
                    messages=[{"role": "user", "content": "x"}],
                    stream=True,
                )
                chunks = list(stream)
        finally:
            uninstrument("anthropic")

        assert len(chunks) == 4
        span = find_span(recording_client, "anthropic.messages.create")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "Paris."}
        ]
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 3
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "end_turn"


class TestAnthropicToolCalls:
    def test_captures_tool_use_block(self, recording_client):
        # Anthropic embeds tool calls as `content` blocks with type="tool_use".
        # `input` arrives as a parsed dict; we normalize to a JSON string.
        instrument("anthropic", recording_client)
        try:
            client = Anthropic(api_key="fake")
            fake = Message(
                id="msg-tools",
                type="message",
                role="assistant",
                model="claude-3-5-haiku-latest",
                content=[
                    ToolUseBlock(
                        type="tool_use",
                        id="toolu_01ABC",
                        name="get_weather",
                        input={"location": "Paris", "unit": "celsius"},
                    ),
                ],
                stop_reason="tool_use",
                stop_sequence=None,
                usage=Usage(input_tokens=12, output_tokens=8),
            )
            weather_tool = {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
            with patch.object(client.messages, "_post", return_value=fake, create=True):
                client.messages.create(
                    model="claude-3-5-haiku-latest",
                    max_tokens=64,
                    messages=[{"role": "user", "content": "weather in Paris?"}],
                    tools=[weather_tool],
                )
        finally:
            uninstrument("anthropic")

        span = find_span(recording_client, "anthropic.messages.create")
        attrs = json.loads(span.attributes_json)

        req_tools = json.loads(attrs[AgenticAttributes.REQUEST_TOOLS])
        assert req_tools[0]["name"] == "get_weather"

        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        assert calls[0]["id"] == "toolu_01ABC"
        assert calls[0]["name"] == "get_weather"
        # arguments is a JSON string; dict ordering isn't guaranteed so parse.
        assert json.loads(calls[0]["arguments"]) == {"location": "Paris", "unit": "celsius"}
        assert attrs[AgenticAttributes.TOOL_NAME] == "get_weather"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "toolu_01ABC"
