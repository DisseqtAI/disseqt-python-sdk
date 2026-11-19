"""
Tests for the Google ADK (Agent Development Kit) instrumentor.

These tests exercise the wrapper closures directly with synthetic
mock objects — google-adk itself does not need to be importable for
them to run. That's deliberate: the wrappers' failure modes (missed
scope closure on cancellation, context-var Token leaks across
frames, aggregator flushed twice) are pure Python control-flow bugs
that don't need a real ADK to reproduce. ``test_adk_e2e.py`` covers
the same behavior through a real ``InMemoryRunner``.
"""

from __future__ import annotations

import asyncio
import gc
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk.context import get_current_span, get_current_trace
from disseqt_agentic_sdk.enums import SpanKind, SpanStatus
from disseqt_agentic_sdk.instrumentation._tool_result import _current_agg, _ToolCallAggregator
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span
from disseqt_agentic_sdk.instrumentation.adk.patch import (
    _AdkGenScope,
    _extract_tool_calls,
    _LlmStreamState,
    _normalize_contents,
    _open_adk_span,
    a2a_run_async_impl,
    agent_run_async,
    llm_call_site,
    llm_generate_content_async,
    memory_call_site,
    memory_search_memory,
    runner_run_async,
    tool_call_site,
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


class _Probe(_AsyncGen):
    """An ``_AsyncGen`` that records what is ambient each time it is resumed."""

    def __init__(self, items):
        super().__init__(items)
        self.seen_agg: list = []
        self.seen_span: list = []

    async def __anext__(self):
        self.seen_agg.append(_current_agg.get())
        self.seen_span.append(get_current_span())
        return await super().__anext__()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
        agg = _ToolCallAggregator()
        token = _current_agg.set(agg)
        try:

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

        inner = _Probe([SimpleNamespace(id=1), SimpleNamespace(id=2)])

        def wrapped(**kwargs):
            return inner

        runner = SimpleNamespace(app_name="my-app", agent=SimpleNamespace(name="root-agent"))
        result = wrapper(
            wrapped,
            runner,
            (),
            {"user_id": "u1", "session_id": "s1", "new_message": None},
        )

        between_yields: list = []

        async def _drive():
            async for _ in result:
                # The consumer's own context, between two events.
                between_yields.append((_current_agg.get(), get_current_span()))

        _run(_drive())

        # While ADK code runs, the aggregator and the runner span are ambient...
        assert all(agg is not None for agg in inner.seen_agg)
        assert len(set(map(id, inner.seen_agg))) == 1
        assert all(sp is not None and sp.name == "adk.runner.run_async" for sp in inner.seen_span)
        # ...but nothing is left in the consumer's context between events
        # or afterwards, so a `break` there cannot leak it.
        assert between_yields == [(None, None), (None, None)]
        assert _current_agg.get() is None
        assert get_current_span() is None
        assert get_current_trace() is None

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

        # Simulate: Runner opens root agg -> BaseAgent runs inside -> tool
        # calls should land on the runner-level aggregator.
        agent_inner = _Probe([SimpleNamespace(id="inner")])
        agent = SimpleNamespace(name="sub", description=None)
        parent_ctx = SimpleNamespace(invocation_id="inv-1", branch=None)
        seen_by_runner_body: list = []

        async def runner_body(**kwargs):
            seen_by_runner_body.append(_current_agg.get())
            async for ev in agent_wrapper(lambda ctx: agent_inner, agent, (parent_ctx,), {}):
                yield ev

        runner = SimpleNamespace(app_name="app", agent=SimpleNamespace(name="root"))

        async def _drive():
            async for _ in runner_wrapper(runner_body, runner, (), {}):
                pass

        _run(_drive())

        # The sub-agent ran under the Runner's aggregator, not its own.
        assert seen_by_runner_body[0] is not None
        assert all(agg is seen_by_runner_body[0] for agg in agent_inner.seen_agg)
        # And its span is parented to the runner span, in the same trace.
        root = find_span(recording_client, "adk.runner.run_async")
        sub = find_span(recording_client, "adk.agent.sub")
        assert sub.parent_span_id == root.span_id
        assert sub.trace_id == root.trace_id
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
# _AdkGenScope directly
# ---------------------------------------------------------------------
def _status(span) -> str:
    return span.status_code


class TestAdkGenScope:
    def test_double_close_is_idempotent(self, recording_client):
        scope = _open_adk_span(recording_client, "adk.agent.x", SpanKind.AGENT_EXEC)
        agen = _AsyncGen([1, 2])
        wrapped = _AdkGenScope(agen, scope)

        async def _drive():
            await wrapped.__anext__()
            await wrapped.aclose()
            await wrapped.aclose()  # second must be a no-op
            with pytest.raises(StopAsyncIteration):
                await wrapped.__anext__()

        _run(_drive())
        assert agen.aclose_called is True
        spans = [s for s in recording_client.buffer.spans if s.name == "adk.agent.x"]
        assert len(spans) == 1  # closed exactly once

    def test_exhaustion_closes_scope(self, recording_client):
        scope = _open_adk_span(recording_client, "adk.agent.y", SpanKind.AGENT_EXEC)
        wrapped = _AdkGenScope(_AsyncGen([SimpleNamespace(id="e")]), scope)

        async def _drain():
            async for _ in wrapped:
                pass

        _run(_drain())
        find_span(recording_client, "adk.agent.y")

    def test_async_for_break_finalizes_on_a_live_loop(self, recording_client):
        """
        PR #33 review: ``async for ... break`` (ADK's quickstart pattern)
        must finalize the run while the loop is still alive — no
        ``shutdown_asyncgens()``, no loop turn — as a clean, non-error
        span, and must leave nothing behind for the next run on that
        loop to nest into.
        """
        instrumentor = MagicMock(client=recording_client)
        wrapper = runner_run_async(instrumentor)
        runner = SimpleNamespace(app_name="app", agent=SimpleNamespace(name="root"))

        def start():
            events = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
            return wrapper(lambda **kw: _Probe(events), runner, (), {})

        async def _drive():
            async for _ in start():
                break  # mimics `if event.is_final_response(): break`

            # Still inside the running loop, immediately after the break.
            first = find_span(recording_client, "adk.runner.run_async")
            assert _status(first) == SpanStatus.OK.value
            assert AgenticAttributes.ERROR_MESSAGE not in json.loads(first.attributes_json)
            assert _current_agg.get() is None
            assert get_current_span() is None
            assert get_current_trace() is None

            # A second run on the same loop is its own root, in its own trace.
            async for _ in start():
                pass
            return first

        first = _run(_drive())
        roots = [s for s in recording_client.buffer.spans if s.name == "adk.runner.run_async"]
        assert len(roots) == 2
        second = roots[1]
        assert second.trace_id != first.trace_id
        assert second.parent_span_id is None
        assert second.root is True

    def test_break_with_a_held_reference_finalizes_on_aclose(self, recording_client):
        """
        A consumer that keeps its own reference to the generator and
        breaks is indistinguishable from one that will resume later, so
        the span stays open until ``aclose()`` (or the reference is
        dropped). Either way nothing leaks into the consumer's context,
        and closing is not an error.
        """
        scope = _open_adk_span(recording_client, "adk.agent.held", SpanKind.AGENT_EXEC)
        inner = _AsyncGen([SimpleNamespace(id="a"), SimpleNamespace(id="b")])
        held = _AdkGenScope(inner, scope, install_agg=True)

        async def _drive():
            async for _ in held:
                break
            assert _current_agg.get() is None
            assert get_current_span() is None
            assert not [s for s in recording_client.buffer.spans if s.name == "adk.agent.held"]
            await held.aclose()

        _run(_drive())
        assert inner.aclose_called is True
        assert _status(find_span(recording_client, "adk.agent.held")) == SpanStatus.OK.value

    def test_dropped_reference_finalizes(self, recording_client):
        scope = _open_adk_span(recording_client, "adk.agent.dropped", SpanKind.AGENT_EXEC)
        holder = [_AdkGenScope(_AsyncGen([1, 2]), scope)]

        async def _drive():
            await holder[0].__anext__()

        _run(_drive())
        assert not [s for s in recording_client.buffer.spans if s.name == "adk.agent.dropped"]
        holder.clear()
        gc.collect()
        assert _status(find_span(recording_client, "adk.agent.dropped")) == SpanStatus.OK.value

    def test_generator_exit_from_inner_is_not_an_error(self, recording_client):
        """``aclose()`` injects GeneratorExit; it must not mark the span failed."""

        class _RaisesGeneratorExit:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise GeneratorExit

        scope = _open_adk_span(recording_client, "adk.agent.gexit", SpanKind.AGENT_EXEC)
        wrapped = _AdkGenScope(_RaisesGeneratorExit(), scope)

        async def _drive():
            with pytest.raises(GeneratorExit):
                await wrapped.__anext__()

        _run(_drive())
        span = find_span(recording_client, "adk.agent.gexit")
        assert _status(span) == SpanStatus.OK.value
        assert AgenticAttributes.ERROR_TYPE not in json.loads(span.attributes_json)

    def test_real_error_is_still_an_error(self, recording_client):
        class _Boom:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("model exploded")

        scope = _open_adk_span(recording_client, "adk.agent.boom", SpanKind.AGENT_EXEC)
        wrapped = _AdkGenScope(_Boom(), scope)

        async def _drive():
            with pytest.raises(RuntimeError):
                await wrapped.__anext__()

        _run(_drive())
        span = find_span(recording_client, "adk.agent.boom")
        assert _status(span) == SpanStatus.ERROR.value
        assert json.loads(span.attributes_json)[AgenticAttributes.ERROR_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------
# Concurrency: parent capture + task-isolated context
# ---------------------------------------------------------------------
class TestConcurrentRuns:
    def test_gathered_runs_get_separate_traces(self, recording_client):
        """
        PR #33 review: two ``run_async`` calls in one ``asyncio.gather``
        share a thread, so thread-local "current span" used to fold them
        into one trace. Each run must be its own trace, with its own
        children parented inside it.
        """
        instrumentor = MagicMock(client=recording_client)
        runner_wrapper = runner_run_async(instrumentor)
        tool_wrapper = tool_run_async(instrumentor)

        async def slow_tool(*, args, tool_context):
            await asyncio.sleep(0.01)  # forces the two runs to interleave
            return "ok"

        def make_run(name):
            async def body(**kwargs):
                tool = SimpleNamespace(name=f"tool-{name}", description=None)
                ctx = SimpleNamespace(function_call_id=f"call-{name}")
                await asyncio.sleep(0)
                await tool_wrapper(slow_tool, tool, (), {"args": {}, "tool_context": ctx})
                yield SimpleNamespace(id=name)

            runner = SimpleNamespace(app_name=name, agent=SimpleNamespace(name=name))

            async def consume():
                async for _ in runner_wrapper(body, runner, (), {}):
                    pass

            return consume()

        async def _drive():
            await asyncio.gather(make_run("a"), make_run("b"))

        _run(_drive())

        roots = {
            json.loads(s.attributes_json)["agentic.app.name"]: s
            for s in recording_client.buffer.spans
            if s.name == "adk.runner.run_async"
        }
        assert set(roots) == {"a", "b"}
        assert roots["a"].trace_id != roots["b"].trace_id
        for name, root in roots.items():
            assert root.parent_span_id is None
            tool = find_span(recording_client, f"adk.tool.tool-{name}")
            assert tool.trace_id == root.trace_id
            assert tool.parent_span_id == root.span_id
        # Nothing was published to the shared thread-local context.
        assert get_current_trace() is None
        assert get_current_span() is None

    def test_run_nests_under_a_trace_the_user_opened(self, recording_client):
        from disseqt_agentic_sdk.trace import DisseqtTrace

        instrumentor = MagicMock(client=recording_client)
        wrapper = runner_run_async(instrumentor)
        runner = SimpleNamespace(app_name="app", agent=SimpleNamespace(name="root"))

        async def _drive():
            async for _ in wrapper(lambda **kw: _AsyncGen([1]), runner, (), {}):
                pass

        trace = DisseqtTrace(name="user-trace", client=recording_client)
        try:
            with trace.start_span("user-span", SpanKind.AGENT_EXEC) as user_span:
                _run(_drive())
                # The user's own context is exactly as they left it.
                assert get_current_span() is user_span
                assert get_current_trace() is trace
        finally:
            trace.end()

        root = find_span(recording_client, "adk.runner.run_async")
        assert root.trace_id == trace.trace_id
        assert root.parent_span_id == user_span.span_id
        assert trace.is_ended  # ended by the user, not by the ADK wrapper

    def test_ended_ambient_trace_is_not_reused(self, recording_client):
        from disseqt_agentic_sdk.context import set_current_trace
        from disseqt_agentic_sdk.trace import DisseqtTrace

        stale = DisseqtTrace(name="stale", client=recording_client)
        stale.end()
        set_current_trace(stale)  # what a leaked, finished trace looks like
        try:
            scope = _open_adk_span(recording_client, "adk.agent.fresh", SpanKind.AGENT_EXEC)
            scope.__exit__(None, None, None)
        finally:
            set_current_trace(None)
        assert find_span(recording_client, "adk.agent.fresh").trace_id != stale.trace_id


# ---------------------------------------------------------------------
# Call-site wrappers + de-duplication
# ---------------------------------------------------------------------
class TestCallSites:
    def test_tool_call_site_and_class_wrapper_emit_one_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        site = tool_call_site(instrumentor)
        klass = tool_run_async(instrumentor)

        tool = SimpleNamespace(name="mcp_lookup", description="an MCP tool")
        tool_context = SimpleNamespace(function_call_id="call-9")

        async def run_async(*, args, tool_context):
            return {"rows": 3}

        async def call_tool_async(tool, args, tool_context):
            # ADK: `await tool.run_async(args=args, tool_context=tool_context)`,
            # where run_async is itself class-patched.
            return await klass(run_async, tool, (), {"args": args, "tool_context": tool_context})

        result = _run(
            site(call_tool_async, None, (tool,), {"args": {"q": 1}, "tool_context": tool_context})
        )
        assert result == {"rows": 3}

        spans = [s for s in recording_client.buffer.spans if s.name == "adk.tool.mcp_lookup"]
        assert len(spans) == 1
        attrs = json.loads(spans[0].attributes_json)
        assert attrs[AgenticAttributes.TOOL_CALL_ID] == "call-9"
        assert json.loads(attrs[AgenticAttributes.TOOL_ARGS]) == {"q": 1}
        assert json.loads(attrs[AgenticAttributes.TOOL_RESULT]) == {"rows": 3}

    def test_super_chain_emits_one_span(self, recording_client):
        """A subclass run_async that calls a patched super().run_async."""
        instrumentor = MagicMock(client=recording_client)
        klass = tool_run_async(instrumentor)
        tool = SimpleNamespace(name="google_tool", description=None)

        async def base_run_async(*, args, tool_context):
            return "ok"

        async def sub_run_async(*, args, tool_context):
            return await klass(
                base_run_async, tool, (), {"args": args, "tool_context": tool_context}
            )

        _run(klass(sub_run_async, tool, (), {"args": {}, "tool_context": None}))
        assert (
            len([s for s in recording_client.buffer.spans if s.name == "adk.tool.google_tool"]) == 1
        )

    def test_different_tool_instances_each_get_a_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        klass = tool_run_async(instrumentor)
        outer = SimpleNamespace(name="outer", description=None)
        inner = SimpleNamespace(name="inner", description=None)

        async def inner_run(*, args, tool_context):
            return "inner"

        async def outer_run(*, args, tool_context):
            return await klass(inner_run, inner, (), {"args": {}, "tool_context": None})

        _run(klass(outer_run, outer, (), {"args": {}, "tool_context": None}))
        outer_span = find_span(recording_client, "adk.tool.outer")
        inner_span = find_span(recording_client, "adk.tool.inner")
        assert inner_span.parent_span_id == outer_span.span_id

    def test_llm_call_site_records_model_turn(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_call_site(instrumentor)

        class LiteLlm:  # class name drives provider detection
            model = "openai/gpt-4o"

        invocation_context = SimpleNamespace(
            agent=SimpleNamespace(canonical_model=LiteLlm()),
            run_config=SimpleNamespace(streaming_mode=SimpleNamespace(name="SSE")),
        )
        req = _fake_llm_request(model="openai/gpt-4o")
        final = _fake_llm_response(text="hi", prompt_tokens=4, output_tokens=2)

        def run_and_handle_error(response_generator, invocation_context, llm_request, event):
            return response_generator

        gen = wrapper(
            run_and_handle_error,
            SimpleNamespace(),
            (_AsyncGen([final]), invocation_context, req, SimpleNamespace()),
            {},
        )

        async def _drain():
            return [it async for it in gen]

        assert _run(_drain()) == [final]
        span = find_span(recording_client, "adk.llm.openai/gpt-4o")
        attrs = json.loads(span.attributes_json)
        assert attrs[GenAIAttributes.SYSTEM] == "openai"
        assert attrs[GenAIAttributes.REQUEST_IS_STREAM] is True
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 4

    def test_provider_spans_are_suppressed_during_the_model_call(self, recording_client):
        """
        PR #33 review: with ``instrument_all`` the provider SDK under an
        ADK model (google-genai under ``Gemini``) is instrumented too.
        Its span — and its tool-call notifications — must be suppressed
        while the ADK MODEL_EXEC span is recording the same turn.
        """
        from disseqt_agentic_sdk.instrumentation._tool_result import _notify_planned_tool_calls
        from disseqt_agentic_sdk.instrumentation._utils import provider_spans_suppressed

        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_generate_content_async(instrumentor)
        agg = _ToolCallAggregator()
        flags: list[bool] = []

        async def provider_call(llm_request, stream=False):
            # What GeminiInstrumentor does around api_client.aio.models.generate_content.
            flags.append(provider_spans_suppressed())
            scope = open_llm_span(recording_client, "gemini.generate_content", SpanKind.MODEL_EXEC)
            scope.span.set_model_info("gemini-2.0-flash", "google")
            _notify_planned_tool_calls([{"id": "provider-id", "name": "t", "arguments": "{}"}])
            scope.__exit__(None, None, None)
            yield _fake_llm_response(text="hi", prompt_tokens=5, output_tokens=2)

        async def _drive():
            token = _current_agg.set(agg)
            try:
                async for _ in wrapper(provider_call, None, (_fake_llm_request(),), {}):
                    flags.append(provider_spans_suppressed())
            finally:
                _current_agg.reset(token)

        _run(_drive())

        assert flags == [True, False]  # on during the call, off for the consumer
        model_spans = [s for s in recording_client.buffer.spans if s.kind == "MODEL_EXEC"]
        assert [s.name for s in model_spans] == ["adk.llm.gemini-2.0-flash"]
        assert "provider-id" not in agg._calls

    def test_a_task_started_during_the_model_call_is_not_suppressed(self, recording_client):
        """
        PR #33 review: the suppression used to be a plain flag in a
        ContextVar. A task started while it was set copied it and kept it
        for life, so that task's own model calls were never traced.
        """
        from disseqt_agentic_sdk.instrumentation._utils import provider_spans_suppressed

        instrumentor = MagicMock(client=recording_client)
        wrapper = llm_generate_content_async(instrumentor)
        seen: dict[str, bool] = {}
        background: list[asyncio.Task[None]] = []

        async def background_job(model_call_over: asyncio.Event) -> None:
            # While the ADK model call is still running...
            seen["task_during"] = provider_spans_suppressed()
            scope = open_llm_span(recording_client, "openai.during", SpanKind.MODEL_EXEC)
            scope.__exit__(None, None, None)
            # ...and long after it finished.
            await model_call_over.wait()
            seen["task_after"] = provider_spans_suppressed()
            scope = open_llm_span(recording_client, "openai.after", SpanKind.MODEL_EXEC)
            scope.__exit__(None, None, None)

        async def _drive():
            model_call_over = asyncio.Event()

            async def provider_call(llm_request, stream=False):
                background.append(asyncio.create_task(background_job(model_call_over)))
                await asyncio.sleep(0)  # let the task run inside the window
                seen["model_call"] = provider_spans_suppressed()
                # A blocking request handed to a worker thread is still this model call.
                seen["worker_thread"] = await asyncio.to_thread(provider_spans_suppressed)
                yield _fake_llm_response(text="hi", prompt_tokens=5, output_tokens=2)

            async for _ in wrapper(provider_call, None, (_fake_llm_request(),), {}):
                pass
            model_call_over.set()
            await asyncio.gather(*background)

        _run(_drive())

        assert seen == {
            "task_during": False,
            "model_call": True,
            "worker_thread": True,
            "task_after": False,
        }
        names = [s.name for s in recording_client.buffer.spans if s.kind == "MODEL_EXEC"]
        assert sorted(names) == ["adk.llm.gemini-2.0-flash", "openai.after", "openai.during"]

    def test_a_copied_context_is_not_suppressed_once_the_model_call_is_over(self):
        """A ``copy_context()`` taken during the call outlives it; the suppression must not."""
        import contextvars

        from disseqt_agentic_sdk.instrumentation._utils import (
            provider_spans_suppressed,
            suppress_provider_spans,
        )

        with suppress_provider_spans():
            assert provider_spans_suppressed() is True
            copied = contextvars.copy_context()
            assert copied.run(provider_spans_suppressed) is True  # same call, sync code
        assert provider_spans_suppressed() is False
        assert copied.run(provider_spans_suppressed) is False

    def test_provider_spans_are_not_suppressed_inside_tools(self, recording_client):
        """A tool that calls an LLM SDK itself is a real, separate model call."""
        instrumentor = MagicMock(client=recording_client)
        klass = tool_run_async(instrumentor)
        tool = SimpleNamespace(name="summarize", description=None)

        async def run_async(*, args, tool_context):
            scope = open_llm_span(recording_client, "openai.chat", SpanKind.MODEL_EXEC)
            scope.__exit__(None, None, None)
            return "ok"

        _run(klass(run_async, tool, (), {"args": {}, "tool_context": None}))
        tool_span = find_span(recording_client, "adk.tool.summarize")
        nested = find_span(recording_client, "openai.chat")
        assert nested.parent_span_id == tool_span.span_id
        assert nested.trace_id == tool_span.trace_id

    def test_memory_call_site_and_class_wrapper_emit_one_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        site = memory_call_site(instrumentor)
        klass = memory_search_memory(instrumentor)

        class FirestoreMemoryService:
            pass

        service = FirestoreMemoryService()
        response = SimpleNamespace(memories=[1, 2, 3])
        context = SimpleNamespace(
            _invocation_context=SimpleNamespace(
                memory_service=service, app_name="app", user_id="u-7"
            )
        )

        async def service_search(*, app_name, user_id, query):
            return response

        async def context_search_memory(query):
            ic = context._invocation_context
            return await klass(
                service_search,
                service,
                (),
                {"app_name": ic.app_name, "user_id": ic.user_id, "query": query},
            )

        assert _run(site(context_search_memory, context, ("hiking",), {})) is response
        spans = [s for s in recording_client.buffer.spans if s.name == "adk.memory.search"]
        assert len(spans) == 1
        attrs = json.loads(spans[0].attributes_json)
        assert attrs["agentic.rag.backend"] == "FirestoreMemoryService"
        assert attrs["agentic.rag.query"] == "hiking"
        assert attrs["agentic.session.user_id"] == "u-7"
        assert attrs["agentic.rag.result_count"] == 3

    def test_memory_call_site_without_service_emits_nothing(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        site = memory_call_site(instrumentor)
        context = SimpleNamespace(_invocation_context=SimpleNamespace(memory_service=None))

        async def context_search_memory(query):
            raise ValueError("Memory service is not available.")

        with pytest.raises(ValueError):
            _run(site(context_search_memory, context, ("q",), {}))
        assert recording_client.buffer.spans == []


class TestContentOptOut:
    """TP-2128 round-3 PR #33 review: tool.result + rag.query honor opt-out."""

    def test_tool_result_dropped_when_capture_disabled(self, recording_client):
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import (
            get_capture_content,
            open_llm_span,
            set_capture_content,
        )
        from disseqt_agentic_sdk.instrumentation.adk.patch import _set_tool_result_attrs
        from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes

        scope = open_llm_span(recording_client, "adk.tool.x", SpanKind.TOOL_EXEC)
        prior = get_capture_content()
        set_capture_content(False)
        try:
            _set_tool_result_attrs(scope.span, result={"secret": "s3cret"})
        finally:
            set_capture_content(prior)
            scope.__exit__(None, None, None)
        span = find_span(recording_client, "adk.tool.x")
        attrs = json.loads(span.attributes_json)
        # Neither the AgenticAttributes nor the GenAIAttributes mirror
        # should ship when capture is disabled.
        assert AgenticAttributes.TOOL_RESULT not in attrs
        assert GenAIAttributes.TOOL_RESULT not in attrs

    def test_rag_query_dropped_when_capture_disabled(self, recording_client):
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import (
            get_capture_content,
            open_llm_span,
            set_capture_content,
        )
        from disseqt_agentic_sdk.instrumentation.adk.patch import _set_memory_search_attrs

        scope = open_llm_span(recording_client, "adk.memory.search", SpanKind.RAG_EXEC)
        prior = get_capture_content()
        set_capture_content(False)
        try:
            _set_memory_search_attrs(
                scope.span,
                instance=SimpleNamespace(),
                kwargs={"query": "my SSN is 123-45-6789", "app_name": "a", "user_id": "u"},
            )
        finally:
            set_capture_content(prior)
            scope.__exit__(None, None, None)
        span = find_span(recording_client, "adk.memory.search")
        attrs = json.loads(span.attributes_json)
        assert "agentic.rag.query" not in attrs
        # Non-content keys still land (backend, app, user).
        assert "agentic.rag.backend" in attrs
        assert "agentic.app.name" in attrs

    def test_tool_args_with_non_json_native_type_serializes(self, recording_client):
        """Datetime / UUID / Path on tool args used to crash span serialization."""
        import datetime as dt

        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.instrumentation._utils import open_llm_span
        from disseqt_agentic_sdk.instrumentation.adk.patch import _set_tool_request_attrs
        from disseqt_agentic_sdk.semantics import AgenticAttributes

        scope = open_llm_span(recording_client, "adk.tool.x", SpanKind.TOOL_EXEC)
        _set_tool_request_attrs(
            scope.span,
            instance=SimpleNamespace(name="t", description=None),
            tool_args={"when": dt.datetime(2026, 1, 1)},
            tool_context=None,
        )
        scope.__exit__(None, None, None)
        span = find_span(recording_client, "adk.tool.x")
        attrs = json.loads(span.attributes_json)
        # The attr landed as a string (not a raw datetime), so a later
        # json.dumps of the whole attributes bag won't crash.
        assert isinstance(attrs[AgenticAttributes.TOOL_ARGS], str)
        assert "2026" in attrs[AgenticAttributes.TOOL_ARGS]


class TestProviderDetection:
    """TP-2128 round-3 PR #33 review: LiteLlm models aren't all Gemini."""

    def test_litellm_openai_tagged_as_openai(self):
        from disseqt_agentic_sdk.instrumentation.adk.patch import _detect_provider
        from disseqt_agentic_sdk.semantics import AgenticProvider, GenAISystem

        agentic, gen_ai = _detect_provider(
            instance=SimpleNamespace(),
            llm_request=SimpleNamespace(model="openai/gpt-4o"),
        )
        assert agentic == AgenticProvider.OPENAI
        assert gen_ai == GenAISystem.OPENAI

    def test_litellm_anthropic_tagged_as_anthropic(self):
        from disseqt_agentic_sdk.instrumentation.adk.patch import _detect_provider
        from disseqt_agentic_sdk.semantics import AgenticProvider, GenAISystem

        agentic, gen_ai = _detect_provider(
            instance=SimpleNamespace(),
            llm_request=SimpleNamespace(model="anthropic/claude-3-5-sonnet"),
        )
        assert agentic == AgenticProvider.ANTHROPIC
        assert gen_ai == GenAISystem.ANTHROPIC

    def test_bare_gemini_still_tagged_as_google(self):
        from disseqt_agentic_sdk.instrumentation.adk.patch import _detect_provider
        from disseqt_agentic_sdk.semantics import AgenticProvider, GenAISystem

        agentic, gen_ai = _detect_provider(
            instance=SimpleNamespace(),
            llm_request=SimpleNamespace(model="gemini-2.0-flash"),
        )
        assert agentic == AgenticProvider.GOOGLE
        assert gen_ai == GenAISystem.GEMINI


# ---------------------------------------------------------------------
# BaseMemoryService.search_memory wrapper
# ---------------------------------------------------------------------
class TestMemoryWrapper:
    def test_records_query_and_result_count(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = memory_search_memory(instrumentor)

        response = SimpleNamespace(memories=[SimpleNamespace(id="m1"), SimpleNamespace(id="m2")])

        async def wrapped(*, app_name, user_id, query):
            return response

        # A concrete subclass so type(instance).__name__ reads sensibly.
        class InMemoryMemoryService:
            pass

        result = _run(
            wrapper(
                wrapped,
                InMemoryMemoryService(),
                (),
                {"app_name": "weather-app", "user_id": "u-1", "query": "hiking"},
            )
        )
        assert result is response

        span = find_span(recording_client, "adk.memory.search")
        attrs = json.loads(span.attributes_json)
        assert attrs[AgenticAttributes.OPERATION_NAME] == "search_memory"
        assert attrs["agentic.rag.backend"] == "InMemoryMemoryService"
        assert attrs["agentic.rag.query"] == "hiking"
        assert attrs["agentic.rag.result_count"] == 2
        assert attrs["agentic.app.name"] == "weather-app"
        assert attrs["agentic.session.user_id"] == "u-1"

    def test_exception_closes_span_and_reraises(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = memory_search_memory(instrumentor)

        async def wrapped(*, app_name, user_id, query):
            raise RuntimeError("memory backend down")

        with pytest.raises(RuntimeError, match="memory backend down"):
            _run(
                wrapper(
                    wrapped,
                    SimpleNamespace(),
                    (),
                    {"app_name": "a", "user_id": "u", "query": "q"},
                )
            )
        # Span reached the buffer — scope closed on the error path.
        find_span(recording_client, "adk.memory.search")

    def test_no_memories_field_leaves_result_count_absent(self, recording_client):
        # Old / alternative backend that returns a bare list or a shape
        # without ``.memories`` — instrumentation must not crash and
        # should silently skip the count attribute.
        instrumentor = MagicMock(client=recording_client)
        wrapper = memory_search_memory(instrumentor)

        async def wrapped(*, app_name, user_id, query):
            return SimpleNamespace()  # no .memories

        _run(
            wrapper(
                wrapped,
                SimpleNamespace(),
                (),
                {"app_name": "a", "user_id": "u", "query": "q"},
            )
        )
        span = find_span(recording_client, "adk.memory.search")
        attrs = json.loads(span.attributes_json)
        assert "agentic.rag.result_count" not in attrs


# ---------------------------------------------------------------------
# RemoteA2aAgent._run_async_impl wrapper (A2A CLIENT span)
# ---------------------------------------------------------------------
class TestA2aWrapper:
    def test_records_from_and_to(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = a2a_run_async_impl(instrumentor)

        events = [SimpleNamespace(id="e1"), SimpleNamespace(id="e2")]

        def wrapped(ctx):
            return _AsyncGen(events)

        remote = SimpleNamespace(
            name="weather_agent",
            agent_card="https://peer.example.com/agent.json",
            url=None,
            _timeout=30.0,
        )
        ctx = SimpleNamespace(invocation_id="inv-1")
        gen = wrapper(wrapped, remote, (ctx,), {})

        async def _drain():
            return [ev async for ev in gen]

        drained = _run(_drain())
        assert drained == events

        span = find_span(recording_client, "adk.a2a.send")
        attrs = json.loads(span.attributes_json)
        assert attrs["agentic.a2a.from"] == "weather_agent"
        assert attrs["agentic.a2a.to"] == "https://peer.example.com/agent.json"
        assert attrs["agentic.a2a.timeout_s"] == 30.0
        # Verify span kind is CLIENT — differentiates it from AGENT_EXEC.
        assert span.kind == "CLIENT"

    def test_cancellation_finalizes_span(self, recording_client):
        instrumentor = MagicMock(client=recording_client)
        wrapper = a2a_run_async_impl(instrumentor)

        class _NeverEnds:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(3600)

            async def aclose(self):
                pass

        def wrapped(ctx):
            return _NeverEnds()

        remote = SimpleNamespace(name="peer", agent_card="u", url=None, _timeout=1.0)
        gen = wrapper(wrapped, remote, (SimpleNamespace(),), {})

        async def _drive():
            it = gen.__aiter__()
            task = asyncio.create_task(it.__anext__())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _run(_drive())
        # Span reached the buffer via cancellation path — no deadlock.
        find_span(recording_client, "adk.a2a.send")

    def test_does_not_install_tool_aggregator(self, recording_client):
        # A2A CLIENT span is a transport wrapper; aggregation belongs on
        # the enclosing AGENT_EXEC. Verify install_agg=False actually
        # prevents this wrapper from setting the aggregator contextvar.
        instrumentor = MagicMock(client=recording_client)
        wrapper = a2a_run_async_impl(instrumentor)

        inner = _Probe([SimpleNamespace(id="e")])

        def wrapped(ctx):
            return inner

        remote = SimpleNamespace(name="peer", agent_card="u", url=None, _timeout=1.0)
        gen = wrapper(wrapped, remote, (SimpleNamespace(),), {})

        async def _drive():
            async for _ in gen:
                pass

        _run(_drive())
        # No aggregator is ambient while the A2A stream runs (unlike
        # runner/agent wrappers).
        assert inner.seen_agg and all(agg is None for agg in inner.seen_agg)


class TestPlannedCallsWithoutAnId:
    """
    google-adk 1.x gives a function call its id on a *copy* of the model
    response, so the planned call this SDK sees has none, while the tool
    runs under ADK's ``adk-<uuid>``. The two must still be one entry on
    the AGENT_EXEC span, whichever is recorded first.
    """

    @staticmethod
    def _response(*names: str):
        parts = [
            SimpleNamespace(function_call=SimpleNamespace(name=n, id=None, args={}), text=None)
            for n in names
        ]
        return SimpleNamespace(
            content=SimpleNamespace(parts=parts),
            interaction_id=None,
            model_version=None,
            finish_reason=None,
            error_code=None,
            error_message=None,
            usage_metadata=None,
        )

    def _record(self, agg, name, call_id):
        from disseqt_agentic_sdk.instrumentation.adk.patch import _record_tool_outcome

        tool = SimpleNamespace(name=name)
        context = SimpleNamespace(function_call_id=call_id)
        token = _current_agg.set(agg)
        try:
            _record_tool_outcome(tool, context, {"a": 1}, {"ok": True}, None)
        finally:
            _current_agg.reset(token)

    def _plan(self, agg, *names):
        from disseqt_agentic_sdk.instrumentation.adk.patch import _set_llm_response_attrs

        token = _current_agg.set(agg)
        try:
            _set_llm_response_attrs(MagicMock(), self._response(*names))
        finally:
            _current_agg.reset(token)

    def test_tools_ran_before_the_plan_is_reported(self):
        agg = _ToolCallAggregator()
        self._record(agg, "add", "adk-111")
        self._record(agg, "add", "adk-222")
        self._record(agg, "lookup", "adk-333")
        self._plan(agg, "add", "lookup", "add")

        assert sorted(agg._calls) == ["adk-111", "adk-222", "adk-333"]
        assert all(c["status"] == "success" for c in agg._calls.values())

    def test_plan_is_reported_before_the_tools_run(self):
        agg = _ToolCallAggregator()
        self._plan(agg, "add", "lookup")
        assert len(agg._calls) == 2
        self._record(agg, "lookup", "adk-333")
        self._record(agg, "add", "adk-111")

        assert sorted(agg._calls) == ["adk-111", "adk-333"]
        assert {c["name"]: c["status"] for c in agg._calls.values()} == {
            "add": "success",
            "lookup": "success",
        }

    def test_a_planned_call_that_never_runs_stays_listed(self):
        agg = _ToolCallAggregator()
        self._plan(agg, "add", "lookup")
        self._record(agg, "add", "adk-111")

        assert len(agg._calls) == 2
        assert "status" not in next(c for c in agg._calls.values() if c["name"] == "lookup")
