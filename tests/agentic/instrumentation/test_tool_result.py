"""
Tests for Lane B: agent_span + record_tool_result.

The four tool validators (tool-failure-rate, tool-call-accuracy,
plan-optimality, plan-coherence) only fire on AGENT_EXEC spans and
read ``agentic.tool_calls``. Lane B makes sure the auto-captured
planned calls (from nested MODEL_EXEC spans) plus user-supplied
execution results end up on the enclosing AGENT_EXEC span.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

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

from disseqt_agentic_sdk import agent_span, record_tool_result  # noqa: E402
from disseqt_agentic_sdk.instrumentation import instrument, uninstrument  # noqa: E402
from disseqt_agentic_sdk.semantics import AgenticAttributes  # noqa: E402
from tests.agentic.instrumentation.conftest import find_span  # noqa: E402


def _tool_use_response(call_id: str = "call_abc") -> ChatCompletion:
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
                            id=call_id,
                            type="function",
                            function=Function(
                                name="get_weather",
                                arguments='{"location":"Paris"}',
                            ),
                        ),
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


def _plain_response() -> ChatCompletion:
    return ChatCompletion(
        id="c",
        model="gpt-4o-mini",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="ok"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class TestLaneB:
    def test_planned_tool_calls_bubble_up_to_agent_span(self, recording_client):
        # No record_tool_result — the plan from a nested MODEL_EXEC should
        # still land on the enclosing AGENT_EXEC span.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with agent_span(recording_client, "weather_agent"):
                with patch.object(
                    client.chat.completions,
                    "_post",
                    return_value=_tool_use_response("call_abc"),
                    create=True,
                ):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "weather in Paris?"}],
                    )
        finally:
            uninstrument("openai")

        agent = find_span(recording_client, "weather_agent")
        attrs = json.loads(agent.attributes_json)
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_abc"
        assert calls[0]["name"] == "get_weather"
        # No status — the user didn't call record_tool_result.
        assert "status" not in calls[0]

    def test_record_tool_result_fuses_plan_and_execution(self, recording_client):
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with agent_span(recording_client, "weather_agent"):
                with patch.object(
                    client.chat.completions,
                    "_post",
                    return_value=_tool_use_response("call_abc"),
                    create=True,
                ):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "weather?"}],
                    )
                # User's tool runs; they record the outcome.
                record_tool_result(
                    "call_abc",
                    result="sunny, 22C",
                    status="success",
                )
        finally:
            uninstrument("openai")

        agent = find_span(recording_client, "weather_agent")
        attrs = json.loads(agent.attributes_json)
        calls = attrs[AgenticAttributes.TOOL_CALLS]
        assert len(calls) == 1
        # Plan fields (from bubble-up) + result fields (from record_tool_result)
        # coexist on the same entry.
        assert calls[0] == {
            "id": "call_abc",
            "name": "get_weather",
            "arguments": '{"location":"Paris"}',
            "result": "sunny, 22C",
            "status": "success",
        }

    def test_multiple_tools_across_multiple_llm_calls(self, recording_client):
        # Two model calls, each proposes a different tool; user records both.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with agent_span(recording_client, "multi_tool_agent"):
                with patch.object(
                    client.chat.completions,
                    "_post",
                    side_effect=[
                        _tool_use_response("call_1"),
                        _tool_use_response("call_2"),
                    ],
                    create=True,
                ):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "a"}],
                    )
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "b"}],
                    )
                record_tool_result("call_1", result="ok1", status="success")
                record_tool_result("call_2", result="ok2", status="failure")
        finally:
            uninstrument("openai")

        agent = find_span(recording_client, "multi_tool_agent")
        attrs = json.loads(agent.attributes_json)
        calls = sorted(attrs[AgenticAttributes.TOOL_CALLS], key=lambda c: c["id"])
        assert [c["id"] for c in calls] == ["call_1", "call_2"]
        assert calls[0]["status"] == "success"
        assert calls[1]["status"] == "failure"

    def test_record_tool_result_outside_agent_span_warns(self, recording_client):
        """
        No agent_span active → helper should log a WARNING and no-op.
        TP-2128 P5: the old version of this test only asserted "doesn't
        raise" — deleting the warning call left the test passing. Now
        we patch the module logger and assert the warning fired with
        the expected call_id so a regression is caught.
        """
        from disseqt_agentic_sdk.instrumentation import _tool_result as tr_mod

        with patch.object(tr_mod, "_logger") as fake_logger:
            record_tool_result("call_xxx", result="ok", status="success")
            fake_logger.warning.assert_called_once()
            msg = fake_logger.warning.call_args.args[0]
            assert "outside agent_span" in msg
            # call_id is passed as a subsequent positional arg to the
            # logger.warning(fmt, *args) call.
            assert "call_xxx" in fake_logger.warning.call_args.args

    def test_agent_span_with_no_tool_calls_does_not_emit_empty_key(self, recording_client):
        # No LLM call inside, no record_tool_result — the span should have
        # no agentic.tool_calls attribute at all (not an empty list).
        instrument("openai", recording_client)
        try:
            with agent_span(recording_client, "quiet_agent"):
                pass
        finally:
            uninstrument("openai")

        agent = find_span(recording_client, "quiet_agent")
        attrs = json.loads(agent.attributes_json)
        assert AgenticAttributes.TOOL_CALLS not in attrs

    def test_concurrent_agent_spans_are_isolated(self, recording_client):
        # Two asyncio tasks each open their own agent_span, each records
        # a distinct call_id. Neither should leak into the other's span.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")

            async def one_flow(agent_name: str, call_id: str, status: str) -> None:
                with agent_span(recording_client, agent_name):
                    await asyncio.sleep(0)  # force interleaving
                    with patch.object(
                        client.chat.completions,
                        "_post",
                        return_value=_tool_use_response(call_id),
                        create=True,
                    ):
                        client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "x"}],
                        )
                    await asyncio.sleep(0)
                    record_tool_result(call_id, result="ok", status=status)

            async def main() -> None:
                await asyncio.gather(
                    one_flow("alpha_agent", "call_a", "success"),
                    one_flow("beta_agent", "call_b", "failure"),
                )

            asyncio.run(main())
        finally:
            uninstrument("openai")

        alpha = find_span(recording_client, "alpha_agent")
        beta = find_span(recording_client, "beta_agent")
        alpha_calls = json.loads(alpha.attributes_json)[AgenticAttributes.TOOL_CALLS]
        beta_calls = json.loads(beta.attributes_json)[AgenticAttributes.TOOL_CALLS]
        # Each agent span sees only its own tool call.
        assert [c["id"] for c in alpha_calls] == ["call_a"]
        assert alpha_calls[0]["status"] == "success"
        assert [c["id"] for c in beta_calls] == ["call_b"]
        assert beta_calls[0]["status"] == "failure"

    def test_record_after_agent_span_exit_warns_and_no_ops(self, recording_client):
        """
        TP-2128 P2 #2.2: if the user captures a live aggregator reference
        (e.g. inside a background task) and calls record_tool_result on it
        after the enclosing agent_span exits, the outcome cannot land on
        the AGENT_EXEC span. Warn loudly instead of silently mutating a
        flushed dict.
        """
        from disseqt_agentic_sdk.instrumentation import _tool_result as tr_mod
        from disseqt_agentic_sdk.instrumentation._tool_result import (
            _current_agg,
            _ToolCallAggregator,
        )

        captured: list[_ToolCallAggregator] = []

        instrument("openai", recording_client)
        try:
            with agent_span(recording_client, "leaky_agent"):
                agg = _current_agg.get()
                assert agg is not None
                captured.append(agg)
        finally:
            uninstrument("openai")

        assert captured[0].closed is True

        # Patch the module logger to capture warnings directly (the SDK
        # logger uses disseqt_logging and may be silent by default).
        with patch.object(tr_mod, "_logger") as fake_logger:
            captured[0].add_result("stale_call", result="late", status="success")
            fake_logger.warning.assert_called_once()
            msg = fake_logger.warning.call_args.args[0]
            assert "flushed aggregator" in msg
            assert "stale_call" in fake_logger.warning.call_args.args

        # Data must NOT have landed in the closed aggregator dict.
        assert "stale_call" not in captured[0]._calls

        # add_planned is a silent no-op — normal for fire-and-forget async
        # LLM calls, not user misuse.
        with patch.object(tr_mod, "_logger") as fake_logger:
            captured[0].add_planned([{"id": "late_plan", "name": "x"}])
            fake_logger.warning.assert_not_called()
        assert "late_plan" not in captured[0]._calls

    def test_plain_llm_call_inside_agent_span_leaves_tool_calls_absent(self, recording_client):
        # LLM call inside agent_span but no tool_calls in the response —
        # the AGENT_EXEC span should still have no tool_calls attribute.
        instrument("openai", recording_client)
        try:
            client = OpenAI(api_key="fake")
            with agent_span(recording_client, "chat_agent"):
                with patch.object(
                    client.chat.completions,
                    "_post",
                    return_value=_plain_response(),
                    create=True,
                ):
                    client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "hi"}],
                    )
        finally:
            uninstrument("openai")

        agent = find_span(recording_client, "chat_agent")
        attrs = json.loads(agent.attributes_json)
        assert AgenticAttributes.TOOL_CALLS not in attrs
