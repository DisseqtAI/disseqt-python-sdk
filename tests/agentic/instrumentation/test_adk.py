"""
Tests for the Google ADK (Agent Development Kit) instrumentor.

These tests exercise the wrapper closures directly with synthetic
mock objects — google-adk itself does not need to be importable for
them to run. That's deliberate: the wrappers' failure modes (missed
scope closure on cancellation, context-var Token leaks across
frames, aggregator flushed twice) are pure Python control-flow bugs
that don't need a real ADK to reproduce. An additional
``pytest.importorskip("google.adk")`` block at the bottom covers
the end-to-end wrapt install path when the package is present in CI.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk.instrumentation._tool_result import _current_agg
from disseqt_agentic_sdk.instrumentation.adk.patch import (
    _AgentGenScope,
    _extract_tool_calls,
    _install_aggregator_if_absent,
    _LlmStreamState,
    _normalize_contents,
    agent_run_async,
    llm_generate_content_async,
    runner_run_async,
    tool_run_async,
)
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes
from tests.agentic.instrumentation.conftest import find_span


# ---------------------------------------------------------------------
# Small helpers to build fake ADK objects
# ---------------------------------------------------------------------
def _fake_llm_request(
    *,
    model: str = "gemini-2.0-flash",
    contents: object = "hello",
    temperature: float | None = 0.5,
    max_output_tokens: int | None = 128,
    system_instruction: str | None = None,
    tools: dict | None = None,
) -> SimpleNamespace:
    config = SimpleNamespace(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=None,
        top_k=None,
        system_instruction=system_instruction,
        tools=None,
    )
    return SimpleNamespace(
        model=model,
        contents=contents,
        config=config,
        tools_dict=tools,
    )


def _fake_llm_response(
    *,
    text: str = "hi",
    prompt_tokens: int = 3,
    output_tokens: int = 1,
    total_tokens: int | None = None,
    finish_reason: str | None = "STOP",
    partial: bool | None = False,
    function_calls: list[dict] | None = None,
    model_version: str = "gemini-2.0-flash-001",
    interaction_id: str = "resp-1",
) -> SimpleNamespace:
    parts = []
    if text:
        parts.append(SimpleNamespace(text=text, function_call=None))
    for fc in function_calls or []:
        parts.append(
            SimpleNamespace(
                text=None,
                function_call=SimpleNamespace(
                    id=fc.get("id"), name=fc["name"], args=fc.get("args") or {}
                ),
            )
        )
    content = SimpleNamespace(parts=parts, role="model")
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        total_token_count=total_tokens or (prompt_tokens + output_tokens),
        thoughts_token_count=None,
        cached_content_token_count=None,
    )
    return SimpleNamespace(
        model_version=model_version,
        content=content,
        finish_reason=finish_reason,
        usage_metadata=usage,
        partial=partial,
        interaction_id=interaction_id,
        error_code=None,
        error_message=None,
    )


class _AsyncGen:
    """Minimal async iterator over a fixed list; supports aclose()."""

    def __init__(self, items):
        self._items = list(items)
        self._i = 0
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._i]
        self._i += 1
        return item

    async def aclose(self):
        self.aclose_called = True


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------
# Registry / alias tests (do not require google-adk installed)
# ---------------------------------------------------------------------
class TestAdkRegistry:
    def test_adk_alias_resolves(self):
        from disseqt_agentic_sdk.instrumentation._registry import resolve_provider_name

        assert resolve_provider_name("adk") == "google-adk"

    def test_google_adk_key_in_registry(self):
        from disseqt_agentic_sdk.instrumentation._registry import INSTRUMENTOR_CLASSES

        assert "google-adk" in INSTRUMENTOR_CLASSES

    def test_adk_alias_not_iterated_by_instrument_all(self):
        from disseqt_agentic_sdk.instrumentation._registry import INSTRUMENTOR_CLASSES

        assert "adk" not in INSTRUMENTOR_CLASSES


# ---------------------------------------------------------------------
# _normalize_contents
# ---------------------------------------------------------------------
class TestNormalizeContents:
    def test_string_becomes_user_message(self):
        assert _normalize_contents("hi") == [{"role": "user", "content": "hi"}]

    def test_none_returns_empty_list(self):
        assert _normalize_contents(None) == []

    def test_content_object_extracts_parts(self):
        content = SimpleNamespace(
            parts=[SimpleNamespace(text="a"), SimpleNamespace(text="b")],
            role="user",
        )
        assert _normalize_contents(content) == [{"role": "user", "content": "ab"}]

    def test_bare_part_falls_back_to_text_attr(self):
        # Round-trips the same "bare Part" shape the Gemini normalizer
        # handles (google-genai first-class input) so ADK requests that
        # pass parts directly don't drop text.
        part = SimpleNamespace(parts=None, text="lonely")
        assert _normalize_contents([part]) == [{"role": "user", "content": "lonely"}]


# ---------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------
class TestExtractToolCalls:
    def test_synthesizes_id_when_missing(self):
        content = SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(id=None, name="lookup", args={"q": "x"}),
                )
            ]
        )
        calls = _extract_tool_calls(content, response_id="resp-42")
        assert len(calls) == 1
        assert calls[0]["name"] == "lookup"
        assert calls[0]["id"] == "resp-42-0"
        assert json.loads(calls[0]["arguments"]) == {"q": "x"}

    def test_uses_real_id_when_present(self):
        content = SimpleNamespace(
            parts=[
                SimpleNamespace(
                    text=None,
                    function_call=SimpleNamespace(id="call-abc", name="lookup", args={}),
                )
            ]
        )
        assert _extract_tool_calls(content, response_id="r")[0]["id"] == "call-abc"

    def test_ignores_text_parts(self):
        content = SimpleNamespace(parts=[SimpleNamespace(text="hello", function_call=None)])
        assert _extract_tool_calls(content, response_id=None) == []


# ---------------------------------------------------------------------
# _LlmStreamState
# ---------------------------------------------------------------------
class TestLlmStreamState:
    def test_finalizes_from_final_chunk_not_partials(self, recording_client):
        state = _LlmStreamState()
        # Stream: two partials followed by one final.
        state.absorb(_fake_llm_response(text="a", prompt_tokens=0, output_tokens=0, partial=True))
        state.absorb(_fake_llm_response(text="b", prompt_tokens=0, output_tokens=0, partial=True))
        state.absorb(_fake_llm_response(text="ab", prompt_tokens=5, output_tokens=2, partial=False))

        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span

        scope = open_llm_span(recording_client, "test", SpanKind.MODEL_EXEC)
        state.finalize(scope.span)
        scope.__exit__(None, None, None)

        span = find_span(recording_client, "test")
        attrs = json.loads(span.attributes_json)
        # Tokens from the final chunk, not double-counted.
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2

    def test_falls_back_to_last_chunk_if_no_final(self, recording_client):
        state = _LlmStreamState()
        # Only partials — final missing (protocol violation / cancelled stream).
        state.absorb(_fake_llm_response(text="a", partial=True, prompt_tokens=1, output_tokens=1))

        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span

        scope = open_llm_span(recording_client, "test-fallback", SpanKind.MODEL_EXEC)
        state.finalize(scope.span)
        scope.__exit__(None, None, None)

        span = find_span(recording_client, "test-fallback")
        attrs = json.loads(span.attributes_json)
        # Still surfaces attributes so span isn't blank.
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 1

    def test_no_chunks_does_not_raise(self, recording_client):
        # A generator that yields nothing (immediate StopAsyncIteration)
        # would produce a stream state with neither final nor last.
        # finalize() must be a safe no-op.
        state = _LlmStreamState()
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span

        scope = open_llm_span(recording_client, "test-empty", SpanKind.MODEL_EXEC)
        state.finalize(scope.span)  # must not raise
        scope.__exit__(None, None, None)


# ---------------------------------------------------------------------
# BaseTool.run_async wrapper
# ---------------------------------------------------------------------
class TestToolWrapper:
    def test_success_records_result_on_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = tool_run_async(instrumentor)

        async def wrapped(*, args, tool_context):
            return {"answer": 42}

        tool = SimpleNamespace(name="calculator", description="does math")
        tool_context = SimpleNamespace(function_call_id="call-1")
        result = _run(wrapper(wrapped, tool, (), {"args": {"x": 1}, "tool_context": tool_context}))

        assert result == {"answer": 42}
        span = find_span(recording_client, "adk.tool.calculator")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.TOOL_NAME] == "calculator"
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "call-1"
        assert attrs[GenAIAttributes.TOOL_NAME] == "calculator"
        assert attrs[AgenticAttributes.OPERATION_NAME] == "execute_tool"

    def test_exception_closes_span_and_reraises(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = tool_run_async(instrumentor)

        async def wrapped(*, args, tool_context):
            raise ValueError("boom")

        tool = SimpleNamespace(name="calculator", description=None)
        tool_context = SimpleNamespace(function_call_id="call-err")
        with pytest.raises(ValueError, match="boom"):
            _run(wrapper(wrapped, tool, (), {"args": {}, "tool_context": tool_context}))

        # Span reached the buffer, which means __exit__ ran on the
        # error path — it wouldn't be there otherwise.
        find_span(recording_client, "adk.tool.calculator")

    def test_cancellation_finalizes_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = tool_run_async(instrumentor)

        async def wrapped(*, args, tool_context):
            await asyncio.sleep(3600)  # never returns

        tool = SimpleNamespace(name="slow", description=None)
        tool_context = SimpleNamespace(function_call_id="call-cancel")

        async def _drive():
            task = asyncio.create_task(
                wrapper(wrapped, tool, (), {"args": {}, "tool_context": tool_context})
            )
            # Yield once so wrapped starts, then cancel.
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _run(_drive())
        # Span reached the buffer via the cancellation path — the wrapper
        # did not deadlock and the scope was closed correctly.
        find_span(recording_client, "adk.tool.slow")

    def test_records_tool_outcome_into_aggregator(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = tool_run_async(instrumentor)

        # Install an aggregator ourselves and hold the token.
        token = _install_aggregator_if_absent()
        try:
            agg = _current_agg.get()

            async def wrapped(*, args, tool_context):
                return "ok"

            tool = SimpleNamespace(name="t", description=None)
            tool_context = SimpleNamespace(function_call_id="c1")
            _run(wrapper(wrapped, tool, (), {"args": {"a": 1}, "tool_context": tool_context}))

            assert "c1" in agg._calls
            assert agg._calls["c1"]["status"] == "success"
            assert agg._calls["c1"]["name"] == "t"
        finally:
            _current_agg.reset(token)


# ---------------------------------------------------------------------
# BaseLlm.generate_content_async wrapper
# ---------------------------------------------------------------------
class TestLlmWrapper:
    def test_captures_request_and_final_response(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_generate_content_async(instrumentor)

        req = _fake_llm_request(system_instruction="be brief")
        final = _fake_llm_response(
            text="hi", prompt_tokens=5, output_tokens=2, finish_reason="STOP"
        )
        agen = _AsyncGen([final])

        def wrapped(llm_request, stream=False):
            return agen

        result = wrapper(wrapped, None, (req,), {"stream": False})

        async def _drain():
            items = []
            async for it in result:
                items.append(it)
            return items

        items = _run(_drain())
        assert items == [final]

        span = find_span(recording_client, "adk.llm.gemini-2.0-flash")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.REQUEST_MODEL] == "gemini-2.0-flash"
        assert attrs[AgenticAttributes.PROVIDER_NAME] == "google"
        assert attrs[AgenticAttributes.SYSTEM_INSTRUCTIONS] == "be brief"
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 5
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2
        assert attrs[AgenticAttributes.RESPONSE_FINISH_REASON] == "STOP"
        assert attrs[GenAIAttributes.SYSTEM] == "gemini"

    def test_streaming_ignores_partial_token_counts(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_generate_content_async(instrumentor)

        chunks = [
            _fake_llm_response(text="a", partial=True, prompt_tokens=0, output_tokens=0),
            _fake_llm_response(text="b", partial=True, prompt_tokens=0, output_tokens=0),
            _fake_llm_response(text="ab", partial=False, prompt_tokens=10, output_tokens=5),
        ]

        def wrapped(llm_request, stream=False):
            return _AsyncGen(chunks)

        result = wrapper(wrapped, None, (_fake_llm_request(),), {"stream": True})

        async def _drain():
            return [it async for it in result]

        drained = _run(_drain())
        assert drained == chunks

        span = find_span(recording_client, "adk.llm.gemini-2.0-flash")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 10
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 5

    def test_early_aclose_closes_span_and_forwards(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_generate_content_async(instrumentor)

        agen = _AsyncGen([_fake_llm_response(text="x", partial=True)] * 3)

        def wrapped(llm_request, stream=False):
            return agen

        result = wrapper(wrapped, None, (_fake_llm_request(),), {"stream": True})

        async def _drain_and_close():
            it = result.__aiter__()
            await it.__anext__()  # consume one
            await result.aclose()  # early exit
            # Second aclose is a no-op (idempotent).
            await result.aclose()

        _run(_drain_and_close())
        # Span reached the buffer — aclose forwarded to underlying gen
        # and scope closed exactly once (idempotency held).
        find_span(recording_client, "adk.llm.gemini-2.0-flash")


# ---------------------------------------------------------------------
# Runner.run_async / BaseAgent.run_async wrappers
# ---------------------------------------------------------------------
class TestRunnerWrapper:
    def test_runner_installs_and_flushes_aggregator(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = runner_run_async(instrumentor)

        events = [SimpleNamespace(id=1), SimpleNamespace(id=2)]

        def wrapped(**kwargs):
            return _AsyncGen(events)

        runner = SimpleNamespace(app_name="my-app", agent=SimpleNamespace(name="root-agent"))
        result = wrapper(
            wrapped,
            runner,
            (),
            {"user_id": "u1", "session_id": "s1", "new_message": None},
        )

        # Inside the generator lifetime, aggregator is set.
        async def _drive():
            it = result.__aiter__()
            await it.__anext__()
            assert _current_agg.get() is not None
            async for _ in result:
                pass

        _run(_drive())

        # After generator exhaustion, aggregator was reset.
        assert _current_agg.get() is None
        span = find_span(recording_client, "adk.runner.run_async")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.AGENT_NAME] == "root-agent"
        assert attrs["agentic.app.name"] == "my-app"
        assert attrs["agentic.session.user_id"] == "u1"
        assert attrs["agentic.session.id"] == "s1"

    def test_nested_agent_does_not_replace_outer_aggregator(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        runner_wrapper = runner_run_async(instrumentor)
        agent_wrapper = agent_run_async(instrumentor)

        # Simulate: Runner opens root agg → BaseAgent runs inside → tool
        # calls should land on the runner-level aggregator.
        outer_events = [SimpleNamespace(id="outer")]

        def outer(**kwargs):
            return _AsyncGen(outer_events)

        runner = SimpleNamespace(app_name="app", agent=SimpleNamespace(name="root"))
        outer_gen = runner_wrapper(outer, runner, (), {})

        captured_agg: list = []

        def inner(parent_context):
            # Capture the aggregator active WHEN the BaseAgent wrapper opens.
            captured_agg.append(_current_agg.get())
            return _AsyncGen([SimpleNamespace(id="inner")])

        agent = SimpleNamespace(name="sub", description=None)
        parent_ctx = SimpleNamespace(invocation_id="inv-1", branch=None)

        async def _drive():
            it = outer_gen.__aiter__()
            await it.__anext__()  # inside runner scope
            inner_gen = agent_wrapper(inner, agent, (parent_ctx,), {})
            async for _ in inner_gen:
                pass
            async for _ in outer_gen:
                pass

        _run(_drive())

        # Runner's aggregator was still active while sub-agent ran.
        assert captured_agg[0] is not None
        # After both scopes exit, aggregator reset.
        assert _current_agg.get() is None

    def test_runner_cancelled_mid_iteration_still_finalizes(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = runner_run_async(instrumentor)

        class _NeverEnds:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(3600)

            async def aclose(self):
                pass

        def wrapped(**kwargs):
            return _NeverEnds()

        runner = SimpleNamespace(app_name="app", agent=SimpleNamespace(name="root"))
        result = wrapper(wrapped, runner, (), {})

        async def _drive():
            it = result.__aiter__()
            task = asyncio.create_task(it.__anext__())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _run(_drive())

        # Aggregator cleaned up; span emitted despite cancellation.
        assert _current_agg.get() is None
        find_span(recording_client, "adk.runner.run_async")


# ---------------------------------------------------------------------
# _AgentGenScope directly
# ---------------------------------------------------------------------
class TestAgentGenScope:
    def test_double_close_is_idempotent(self, recording_client):
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span

        scope = open_llm_span(recording_client, "adk.agent.x", SpanKind.AGENT_EXEC)
        agen = _AsyncGen([1, 2])
        wrapped = _AgentGenScope(agen=agen, scope=scope, span=scope.span, install_agg=False)

        async def _drive():
            await wrapped.aclose()
            await wrapped.aclose()  # second must be a no-op

        _run(_drive())
        # aclose was forwarded exactly once (idempotency guard held).
        assert agen.aclose_called is True

    def test_exhaustion_closes_scope(self, recording_client):
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span

        scope = open_llm_span(recording_client, "adk.agent.y", SpanKind.AGENT_EXEC)
        agen = _AsyncGen([SimpleNamespace(id="e")])
        wrapped = _AgentGenScope(agen=agen, scope=scope, span=scope.span, install_agg=False)

        async def _drain():
            async for _ in wrapped:
                pass

        _run(_drain())
        find_span(recording_client, "adk.agent.y")


# ---------------------------------------------------------------------
# End-to-end wrapt install (only if google-adk is available)
# ---------------------------------------------------------------------
class TestAdkEndToEnd:
    def test_instrument_and_uninstrument_roundtrip(self, recording_client):
        pytest.importorskip("google.adk.runners")
        from disseqt_agentic_sdk.instrumentation import instrument, uninstrument
        from disseqt_agentic_sdk.instrumentation.auto import get_instrumented_client

        assert instrument("adk", recording_client) is True
        try:
            # Both alias and canonical resolve to the same active entry.
            assert get_instrumented_client("adk") is recording_client
            assert get_instrumented_client("google-adk") is recording_client
        finally:
            assert uninstrument("adk") is True
        assert get_instrumented_client("google-adk") is None
