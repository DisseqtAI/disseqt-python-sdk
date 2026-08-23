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
from openai.types.chat.chat_completion_message_tool_call import (  # noqa: E402
    ChatCompletionMessageToolCall,
    Function,
)
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


def _fake_chat_completion_with_tools() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-tools",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc123",
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments='{"location":"Paris","unit":"celsius"}',
                            ),
                        ),
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


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

    def test_cancelled_await_still_finalizes_span(self, recording_client):
        """
        Regression: `except Exception` in the async wrapper missed
        `asyncio.CancelledError` (a BaseException), so a cancelled
        `await client.chat.completions.create(...)` leaked the span
        with no error status recorded. Fix widens to `except BaseException`.

        Simulate cancellation with asyncio.wait_for + a slow patched
        HTTP path.
        """
        import asyncio

        from openai import AsyncOpenAI

        instrument("openai", recording_client)
        try:
            aclient = AsyncOpenAI(api_key="fake")

            async def _slow_post(*_a, **_kw):
                await asyncio.sleep(10)
                return _fake_chat_completion()

            with patch.object(
                aclient.chat.completions,
                "_post",
                side_effect=_slow_post,
                create=True,
            ):

                async def _drive():
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(
                            aclient.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": "x"}],
                            ),
                            timeout=0.05,
                        )

                asyncio.run(_drive())
        finally:
            uninstrument("openai")

        # The span must be present (scope.__exit__ ran because
        # `except BaseException` caught the CancelledError) and must be
        # marked ERROR so downstream dashboards see the failed call.
        span = find_span(recording_client, "openai.chat.completions.create")
        assert span.status_code == "ERROR"

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


class TestOpenAIToolCalls:
    def test_captures_request_tools_and_response_tool_calls(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            fake = _fake_chat_completion_with_tools()
            with patch.object(client.chat.completions, "_post", return_value=fake, create=True):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                    tools=[_WEATHER_TOOL],
                )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        # Request-side: tools schema captured as JSON string.
        req_tools = json.loads(attrs[AgenticAttributes.REQUEST_TOOLS])
        assert req_tools[0]["function"]["name"] == "get_weather"
        assert attrs[GenAIAttributes.REQUEST_TOOLS] == attrs[AgenticAttributes.REQUEST_TOOLS]

        # Response-side: canonical tool_calls list.
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert calls == [
            {
                "id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"location":"Paris","unit":"celsius"}',
            }
        ]
        assert attrs[GenAIAttributes.TOOL_CALLS] == calls

        # First-call convenience columns for the enriched-table lookup.
        assert attrs[AgenticAttributes.TOOL_NAME] == "get_weather"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "call_abc123"
        assert attrs[AgenticAttributes.TOOL_ARGS] == '{"location":"Paris","unit":"celsius"}'
        assert attrs[GenAIAttributes.TOOL_NAME] == "get_weather"

    def test_streaming_aggregates_tool_calls(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            def _tc_chunk(index, id_=None, name=None, args_frag=None, finish=None):
                ch = MagicMock()
                ch.id = "chatcmpl-tools-stream"
                ch.model = "gpt-4o-mini"
                ch.usage = None
                choice = MagicMock()
                fn = MagicMock(name=name, arguments=args_frag)
                # MagicMock intercepts `.name`; set explicitly.
                fn.name = name
                fn.arguments = args_frag
                tc = MagicMock(index=index, id=id_, function=fn)
                choice.delta = MagicMock(
                    role="assistant" if index == 0 else None,
                    content=None,
                    tool_calls=[tc],
                )
                choice.finish_reason = finish
                ch.choices = [choice]
                return ch

            usage_obj = MagicMock(prompt_tokens=10, completion_tokens=6, total_tokens=16)
            final_chunk = MagicMock()
            final_chunk.id = "chatcmpl-tools-stream"
            final_chunk.model = "gpt-4o-mini"
            final_chunk.usage = usage_obj
            final_chunk.choices = []

            fake_stream = iter(
                [
                    _tc_chunk(0, id_="call_xyz", name="get_weather", args_frag='{"loc'),
                    _tc_chunk(0, args_frag='ation":"Paris"}', finish="tool_calls"),
                    final_chunk,
                ]
            )
            with patch.object(
                client.chat.completions, "_post", return_value=fake_stream, create=True
            ):
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "weather?"}],
                    stream=True,
                    tools=[_WEATHER_TOOL],
                )
                list(stream)
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert calls == [
            {"id": "call_xyz", "name": "get_weather", "arguments": '{"location":"Paris"}'}
        ]
        assert attrs[AgenticAttributes.TOOL_NAME] == "get_weather"


class TestOpenAINMultiChoice:
    """
    TP-2128 P2 #2.7: n>1 responses used to mix tool_calls across
    choices (flattened list from every choice) while finish_reason kept
    coming from choice 0 only. That misattributed which choice owned a
    given tool call and broke the AGENT_EXEC plan-coherence validator.
    Now tool_calls come from choice 0 exclusively — matches how
    RESPONSE_FINISH_REASON (singular) and TOOL_NAME/TOOL_CALL_ID/
    TOOL_ARGS already behave. RESPONSE_FINISH_REASONS (plural) still
    lists every choice.
    """

    def _build_multi_choice_response(self):
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )
        from openai.types.completion_usage import CompletionUsage

        def _tc(id_, name, args):
            return ChatCompletionMessageToolCall(
                id=id_, type="function", function=Function(name=name, arguments=args)
            )

        return ChatCompletion(
            id="chatcmpl-multi",
            model="gpt-4o-mini",
            object="chat.completion",
            created=0,
            choices=[
                Choice(
                    index=0,
                    finish_reason="tool_calls",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[_tc("call_0", "get_weather", '{"loc":"Paris"}')],
                    ),
                ),
                Choice(
                    index=1,
                    finish_reason="tool_calls",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[_tc("call_1", "get_stock", '{"sym":"AAPL"}')],
                    ),
                ),
                Choice(
                    index=2,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content="plain text"),
                ),
            ],
            usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    def test_tool_calls_come_from_choice_zero_only(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with patch.object(
                client.chat.completions,
                "_post",
                return_value=self._build_multi_choice_response(),
                create=True,
            ):
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    n=3,
                    messages=[{"role": "user", "content": "give me options"}],
                )
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)

        # Only choice 0's tool call lands — choice 1's is dropped.
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_0"
        assert calls[0]["name"] == "get_weather"

        # Convenience columns match choice 0.
        assert attrs[AgenticAttributes.TOOL_NAME] == "get_weather"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "call_0"

        # Plural finish_reasons still lists every choice.
        assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == [
            "tool_calls",
            "tool_calls",
            "stop",
        ]
        # Singular keeps choice-0 value (back-compat).
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "tool_calls"

        # output_messages still lists every choice — content is
        # per-choice and callers reading this array shouldn't lose the
        # multi-choice fan-out.
        msgs = attrs[AgenticAttributes.OUTPUT_MESSAGES]
        assert len(msgs) == 3


class TestOpenAIStreamingMultiChoice:
    """
    Streaming n>1: tool-call deltas from non-zero choices must be
    ignored. Their fragment indexes restart at 0 for each choice and
    would collide into choice 0's slot, corrupting the accumulated
    arguments. TP-2128 P2 #2.7 (streaming side).
    """

    def test_only_choice_zero_tool_calls_accumulate(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            def _chunk_for_choice(choice_index, tc_id, tc_name, args_frag, finish=None):
                ch = MagicMock()
                ch.id = "chatcmpl-n2-stream"
                ch.model = "gpt-4o-mini"
                ch.usage = None
                choice = MagicMock()
                choice.index = choice_index
                fn = MagicMock()
                fn.name = tc_name
                fn.arguments = args_frag
                tc = MagicMock(index=0, id=tc_id, function=fn)
                choice.delta = MagicMock(
                    role="assistant" if tc_id else None,
                    content=None,
                    tool_calls=[tc] if tc_name or args_frag else None,
                )
                choice.finish_reason = finish
                ch.choices = [choice]
                return ch

            # Choice 0 tool call: get_weather → {"loc":"Paris"}
            # Choice 1 tool call: DIFFERENT tool that MUST be ignored.
            fake_stream = iter(
                [
                    _chunk_for_choice(0, "call_c0", "get_weather", '{"loc'),
                    _chunk_for_choice(1, "call_c1", "get_stock", '{"sym'),  # ignore
                    _chunk_for_choice(0, None, None, 'ation":"Paris"}', finish="tool_calls"),
                    _chunk_for_choice(1, None, None, '":"AAPL"}', finish="tool_calls"),  # ignore
                ]
            )
            with patch.object(
                client.chat.completions, "_post", return_value=fake_stream, create=True
            ):
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    n=2,
                    messages=[{"role": "user", "content": "two answers"}],
                    stream=True,
                    tools=[_WEATHER_TOOL],
                )
                list(stream)
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.chat.completions.create")
        attrs = json.loads(span.attributes_json)
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        # Only choice-0's call. If choice-1's deltas leaked in, we'd see
        # either 2 tool calls or a corrupted arguments string.
        assert calls == [
            {"id": "call_c0", "name": "get_weather", "arguments": '{"location":"Paris"}'}
        ]


class TestOpenAILegacyCompletionsStreaming:
    """
    TP-2128 P2 #2.5: legacy /v1/completions streaming path used to wire
    on_chunk/on_finish to no-ops, so a streamed legacy completion
    recorded ZERO response attrs (no model, no id, no completion text,
    no tokens, no finish_reason). Now a dedicated accumulator handles
    the ``choices[i].text`` shape.
    """

    def test_streaming_records_response_attrs(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            def _chunk(text=None, finish=None, usage=None):
                ch = MagicMock()
                ch.id = "cmpl-legacy-stream"
                ch.model = "gpt-3.5-turbo-instruct"
                ch.usage = usage
                choice = MagicMock()
                choice.index = 0
                choice.text = text
                choice.finish_reason = finish
                ch.choices = [choice]
                return ch

            usage_obj = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
            final = MagicMock()
            final.id = "cmpl-legacy-stream"
            final.model = "gpt-3.5-turbo-instruct"
            final.usage = usage_obj
            final.choices = []

            fake_stream = iter(
                [
                    _chunk(text="Once "),
                    _chunk(text="upon "),
                    _chunk(text="a time.", finish="stop"),
                    final,
                ]
            )

            with patch.object(client.completions, "_post", return_value=fake_stream, create=True):
                stream = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Continue the story: Once",
                    stream=True,
                )
                list(stream)
        finally:
            uninstrument("openai")

        span = find_span(recording_client, "openai.completions.create")
        attrs = json.loads(span.attributes_json)

        # Before the fix: none of these were set.
        assert attrs[AgenticAttributes.RESPONSE_MODEL] == "gpt-3.5-turbo-instruct"
        assert attrs[AgenticAttributes.RESPONSE_ID] == "cmpl-legacy-stream"
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "stop"
        assert attrs[AgenticAttributes.OUTPUT_MESSAGES] == [
            {"role": "assistant", "content": "Once upon a time."}
        ]
        assert attrs[GenAIAttributes.COMPLETION] == "Once upon a time."
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 3
        assert attrs[AgenticAttributes.USAGE_TOTAL_TOKENS] == 8


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
