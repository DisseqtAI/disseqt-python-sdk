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


class TestCohereAsyncStream:
    def test_async_stream_does_not_await_async_generator(self, recording_client):
        """
        Regression: Cohere's ``AsyncV2Client.chat_stream`` is an async
        **generator function**, not a coroutine — calling it returns an
        async-generator object immediately. If the wrapper does
        ``await wrapped(...)`` the call raises ``TypeError: object
        async_generator can't be used in 'await' expression``. This test
        uses a real async-generator replacement (not ``AsyncMock``, which
        would happily be awaited and hide the bug).
        """
        import asyncio

        from cohere.v2.client import AsyncV2Client

        events = [
            MagicMock(type="message-start", id="cohere-async-stream"),
            MagicMock(
                type="content-delta",
                delta=MagicMock(message=MagicMock(content=MagicMock(text="Paris."))),
            ),
            MagicMock(
                type="message-end",
                delta=MagicMock(finish_reason="COMPLETE"),
            ),
        ]

        async def fake_chat_stream(self, *args, **kwargs):
            for evt in events:
                yield evt

        original = AsyncV2Client.chat_stream
        AsyncV2Client.chat_stream = fake_chat_stream
        try:
            instrument("cohere", recording_client)
            try:
                aclient = cohere.AsyncClientV2(api_key="fake")

                async def _consume() -> list:
                    stream = aclient.chat_stream(
                        model="command-r-plus",
                        messages=[{"role": "user", "content": "x"}],
                    )
                    out = []
                    async for chunk in stream:
                        out.append(chunk)
                    return out

                chunks = asyncio.run(_consume())
            finally:
                uninstrument("cohere")
        finally:
            AsyncV2Client.chat_stream = original

        # If the wrapper had awaited the async-gen function, asyncio.run
        # would have raised TypeError long before we got here.
        assert len(chunks) == 3


class TestCohereStreamingToolCalls:
    def test_stream_captures_tool_call_events(self, recording_client):
        """
        Regression: Cohere v2 streaming emits `tool-call-start` +
        `tool-call-delta` events; before TP-2128 P1 #1.8 only
        `message-start`/`content-delta`/`message-end` were handled, so
        a streamed tool-calling turn recorded zero tool_calls even
        though real ones were emitted.
        """
        # MagicMock treats `.name` as its own bookkeeping — use SimpleNamespace
        # for anything with a `.name` we care about, so it round-trips as a
        # real attribute the adapter can read.
        from types import SimpleNamespace

        from cohere.v2.client import V2Client

        def _tc_start_event():
            fn = SimpleNamespace(name="get_weather", arguments='{"loc')
            tc = SimpleNamespace(id="tc_1", function=fn)
            return SimpleNamespace(
                type="tool-call-start",
                index=0,
                delta=SimpleNamespace(message=SimpleNamespace(tool_calls=[tc])),
            )

        def _tc_delta_event():
            fn = SimpleNamespace(name=None, arguments='":"Paris"}')
            tc = SimpleNamespace(id=None, function=fn)
            return SimpleNamespace(
                type="tool-call-delta",
                index=0,
                delta=SimpleNamespace(message=SimpleNamespace(tool_calls=[tc])),
            )

        events = [
            SimpleNamespace(type="message-start", id="cohere-stream-tools"),
            _tc_start_event(),
            _tc_delta_event(),
            SimpleNamespace(type="tool-call-end", index=0),
            SimpleNamespace(
                type="message-end",
                delta=SimpleNamespace(
                    finish_reason="TOOL_CALL",
                    usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=5, output_tokens=7)),
                ),
            ),
        ]

        def fake_chat_stream(self, *args, **kwargs):
            yield from events

        original = V2Client.chat_stream
        V2Client.chat_stream = fake_chat_stream
        try:
            instrument("cohere", recording_client)
            try:
                client = cohere.ClientV2(api_key="fake")
                stream = client.chat_stream(
                    model="command-r-plus",
                    messages=[{"role": "user", "content": "weather?"}],
                    tools=[{"type": "function", "function": {"name": "get_weather"}}],
                )
                list(stream)  # drain
            finally:
                uninstrument("cohere")
        finally:
            V2Client.chat_stream = original

        span = find_span(recording_client, "cohere.chat_stream")
        attrs = json.loads(span.attributes_json)
        # Before the fix: KeyError — TOOL_CALLS attr was never set.
        # After: the reassembled tool call lands with both fragments joined.
        assert AgenticAttributes.TOOL_CALLS in attrs
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        assert calls[0]["id"] == "tc_1"
        assert calls[0]["name"] == "get_weather"
        # Args fragments '{"loc' + '":"Paris"}' assembled into valid JSON.
        assert json.loads(calls[0]["arguments"]) == {"loc": "Paris"}


class TestCohereToolCalls:
    def test_captures_tool_calls_on_message(self, recording_client):
        # Cohere v2 puts tool calls on `response.message.tool_calls` in the
        # OpenAI-shape (id, function.name, function.arguments).
        instrument("cohere", recording_client)
        try:
            client = cohere.ClientV2(api_key="fake")

            fn = MagicMock()
            fn.name = "get_weather"
            fn.arguments = '{"location":"Paris"}'
            tc = MagicMock(id="tool_c1", function=fn)
            message = MagicMock(role="assistant", content=[MagicMock(text="")], tool_calls=[tc])
            tokens = MagicMock(input_tokens=9, output_tokens=5)
            usage = MagicMock(tokens=tokens)
            fake = MagicMock(
                id="cohere-tools",
                finish_reason="TOOL_CALL",
                message=message,
                usage=usage,
            )
            http_response = MagicMock(data=fake)
            tools_arg = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]
            with patch.object(client._raw_client, "chat", return_value=http_response, create=True):
                client.chat(
                    model="command-r-plus",
                    messages=[{"role": "user", "content": "weather?"}],
                    tools=tools_arg,
                )
        finally:
            uninstrument("cohere")

        span = find_span(recording_client, "cohere.chat")
        attrs = json.loads(span.attributes_json)

        req_tools = json.loads(attrs[AgenticAttributes.REQUEST_TOOLS])
        assert req_tools[0]["function"]["name"] == "get_weather"

        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert calls == [
            {"id": "tool_c1", "name": "get_weather", "arguments": '{"location":"Paris"}'}
        ]
        assert attrs[AgenticAttributes.TOOL_NAME] == "get_weather"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "tool_c1"
