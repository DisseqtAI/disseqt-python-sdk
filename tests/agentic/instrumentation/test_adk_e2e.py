"""
End-to-end tests for the Google ADK instrumentor against the real
``google-adk`` package: a real ``InMemoryRunner`` driving a real
``LlmAgent`` with real tools, with only the model stubbed out.

``google-adk`` is a dev dependency, so it is imported unconditionally —
if it cannot be imported these tests must fail, not skip. (They used to
sit behind ``pytest.importorskip`` and were silently skipped because the
locked OpenTelemetry version made ``google.adk`` unimportable.)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import patch

from google import genai
from google.adk.agents import LlmAgent
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from disseqt_agentic_sdk.context import get_current_span, get_current_trace
from disseqt_agentic_sdk.enums import SpanStatus
from disseqt_agentic_sdk.instrumentation import instrument, uninstrument
from disseqt_agentic_sdk.instrumentation._tool_result import _current_agg
from disseqt_agentic_sdk.instrumentation.auto import get_instrumented_client
from disseqt_agentic_sdk.semantics import AgenticAttributes


# ---------------------------------------------------------------------
# Real ADK building blocks (only the model is a stub)
# ---------------------------------------------------------------------
class ScriptedLlm(BaseLlm):
    """
    A ``BaseLlm`` subclass that overrides ``generate_content_async``
    without calling ``super()`` — the shape every real model class has.
    First turn: asks for the given tools. After tool results: answers.
    """

    tool_names: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        await asyncio.sleep(0)  # a real model call yields to the loop
        last = llm_request.contents[-1]
        if self.tool_names and not any(p.function_response for p in last.parts):
            calls = [
                types.Part(function_call=types.FunctionCall(name=name, args={"a": 1, "b": 2}))
                for name in self.tool_names
            ]
            yield LlmResponse(content=types.Content(role="model", parts=calls))
            return
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="done")]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=11, candidates_token_count=3, total_token_count=14
            ),
        )


async def add(a: int, b: int) -> dict:
    """Add two numbers."""
    await asyncio.sleep(0)
    return {"sum": a + b}


class NoSuperTool(BaseTool):
    """
    Overrides ``run_async`` without calling ``BaseTool.run_async`` and is
    on no patch list — the same shape as ``McpTool`` / ``RestApiTool`` /
    the LangChain and CrewAI adapters.
    """

    def __init__(self) -> None:
        super().__init__(name="no_super", description="not on any allowlist")

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(type="OBJECT", properties={}),
        )

    async def run_async(self, *, args: dict, tool_context: ToolContext) -> dict:
        return {"ok": True}


class NoSuperMemoryService(BaseMemoryService):
    """Same idea for memory: the shape of ``FirestoreMemoryService``."""

    async def add_session_to_memory(self, session) -> None:  # type: ignore[no-untyped-def]
        return None

    async def search_memory(self, *, app_name: str, user_id: str, query: str):  # type: ignore[no-untyped-def]
        return SearchMemoryResponse(memories=[])


async def recall(a: int, b: int, tool_context: ToolContext) -> dict:
    """Look something up in memory."""
    found = await tool_context.search_memory("trail conditions")
    return {"hits": len(found.memories)}


def _runner(name: str, *, tools: list, tool_names: list[str]) -> InMemoryRunner:
    agent = LlmAgent(
        name=name,
        model=ScriptedLlm(model="scripted-model", tool_names=tool_names),
        tools=tools,
    )
    return InMemoryRunner(agent=agent, app_name=name)


def _default_runner(name: str) -> InMemoryRunner:
    return _runner(name, tools=[FunctionTool(add), NoSuperTool()], tool_names=["add", "no_super"])


async def _events(runner: Runner):  # type: ignore[no-untyped-def]
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    message = types.Content(role="user", parts=[types.Part(text="hi")])
    return runner.run_async(user_id="u", session_id=session.id, new_message=message)


async def _drain(runner: Runner) -> None:
    async for _ in await _events(runner):
        pass


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _spans(client, *, kind: str | None = None, name: str | None = None) -> list:  # type: ignore[no-untyped-def]
    return [
        s
        for s in client.buffer.spans
        if (kind is None or s.kind == kind) and (name is None or s.name == name)
    ]


def _one(client, **kw):  # type: ignore[no-untyped-def]
    found = _spans(client, **kw)
    assert len(found) == 1, f"expected one span for {kw}, got {[s.name for s in found]}"
    return found[0]


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
class TestInMemoryRunner:
    def test_emits_model_and_tool_spans_in_one_tree(self, recording_client):
        assert instrument("adk", recording_client) is True
        _run(_drain(_default_runner("solo")))

        root = _one(recording_client, name="adk.runner.run_async")
        agent = _one(recording_client, name="adk.agent.solo")
        models = _spans(recording_client, kind="MODEL_EXEC")
        tools = _spans(recording_client, kind="TOOL_EXEC")

        # One span per model turn and per tool call — none missing because
        # a class overrides without super(), none doubled by stacked patches.
        assert [s.name for s in models] == ["adk.llm.scripted-model"] * 2
        assert sorted(s.name for s in tools) == ["adk.tool.add", "adk.tool.no_super"]

        assert root.parent_span_id is None
        assert agent.parent_span_id == root.span_id
        for child in models + tools:
            assert child.parent_span_id == agent.span_id
        assert {s.trace_id for s in recording_client.buffer.spans} == {root.trace_id}
        assert all(s.status_code == SpanStatus.OK.value for s in recording_client.buffer.spans)

        # Usage from the final model turn landed on its span.
        usage = [
            json.loads(s.attributes_json).get(AgenticAttributes.USAGE_INPUT_TOKENS) for s in models
        ]
        assert 11 in usage

        # Tool outcomes were fused onto the root AGENT_EXEC span.
        calls = json.loads(root.attributes_json)[AgenticAttributes.TOOL_CALLS]
        assert sorted(c["name"] for c in calls) == ["add", "no_super"]
        assert all(c["status"] == "success" for c in calls)

        # Nothing is left behind once the run is over.
        assert get_current_trace() is None
        assert get_current_span() is None
        assert _current_agg.get() is None

    def test_break_finalizes_on_a_live_loop_and_next_run_is_separate(self, recording_client):
        instrument("adk", recording_client)
        runner = _default_runner("quickstart")

        async def scenario():
            async for event in await _events(runner):
                if event.is_final_response():
                    break  # ADK quickstart

            # Immediately after the break: same loop, no shutdown_asyncgens().
            first = _one(recording_client, name="adk.runner.run_async")
            assert first.status_code == SpanStatus.OK.value
            assert AgenticAttributes.ERROR_MESSAGE not in json.loads(first.attributes_json)
            assert get_current_trace() is None
            assert get_current_span() is None
            assert _current_agg.get() is None

            await _drain(runner)
            return first

        first = _run(scenario())

        roots = _spans(recording_client, name="adk.runner.run_async")
        assert len(roots) == 2
        second = roots[1]
        assert second.trace_id != first.trace_id
        assert second.parent_span_id is None
        # The second run's whole tree is in the second trace.
        second_tree = [s for s in recording_client.buffer.spans if s.trace_id == second.trace_id]
        assert sorted(s.kind for s in second_tree) == sorted(
            ["AGENT_EXEC", "AGENT_EXEC", "MODEL_EXEC", "MODEL_EXEC", "TOOL_EXEC", "TOOL_EXEC"]
        )
        # The abandoned run's spans all closed cleanly — a break is not a failure.
        assert all(s.status_code == SpanStatus.OK.value for s in recording_client.buffer.spans)
        assert len(_spans(recording_client, name="adk.agent.quickstart")) == 2

    def test_gathered_runs_land_in_separate_traces(self, recording_client):
        instrument("adk", recording_client)

        async def scenario():
            await asyncio.gather(_drain(_default_runner("left")), _drain(_default_runner("right")))

        _run(scenario())

        roots = _spans(recording_client, name="adk.runner.run_async")
        assert len(roots) == 2
        assert roots[0].trace_id != roots[1].trace_id
        by_id = {s.span_id: s for s in recording_client.buffer.spans}
        for name in ("left", "right"):
            agent = _one(recording_client, name=f"adk.agent.{name}")
            tree = [s for s in recording_client.buffer.spans if s.trace_id == agent.trace_id]
            # Each trace holds exactly its own run...
            assert len(tree) == 6
            assert by_id[agent.parent_span_id].name == "adk.runner.run_async"
            # ...and every span's parent is in that same trace.
            for span in tree:
                if span.parent_span_id is not None:
                    assert by_id[span.parent_span_id].trace_id == agent.trace_id
                else:
                    assert span.name == "adk.runner.run_async"

    def test_memory_service_without_super_is_traced(self, recording_client):
        instrument("adk", recording_client)
        agent = LlmAgent(
            name="rememberer",
            model=ScriptedLlm(model="scripted-model", tool_names=["recall"]),
            tools=[FunctionTool(recall)],
        )
        runner = Runner(
            agent=agent,
            app_name="rememberer",
            session_service=InMemorySessionService(),
            memory_service=NoSuperMemoryService(),
        )
        _run(_drain(runner))

        rag = _one(recording_client, kind="RAG_EXEC")
        tool = _one(recording_client, name="adk.tool.recall")
        assert rag.parent_span_id == tool.span_id
        attrs = json.loads(rag.attributes_json)
        assert attrs["agentic.rag.backend"] == "NoSuperMemoryService"
        assert attrs["agentic.rag.query"] == "trail conditions"
        assert attrs["agentic.rag.result_count"] == 0


class TestGeminiDoubleCount:
    def test_gemini_turn_emits_one_model_span_with_both_instrumentors(self, recording_client):
        """
        ``instrument_all`` turns on both the ADK and the google-genai
        instrumentors. ``Gemini.generate_content_async`` calls
        ``api_client.aio.models.generate_content``, which google-genai's
        instrumentor wraps — one turn must still be one MODEL_EXEC span.
        """
        instrument("google-genai", recording_client)
        instrument("adk", recording_client)

        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(role="model", parts=[types.Part(text="Paris.")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=7, candidates_token_count=2, total_token_count=9
            ),
            model_version="gemini-2.0-flash-001",
        )

        async def fake_generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
            return response

        agent = LlmAgent(
            name="geo",
            model=Gemini(model="gemini-2.0-flash", client=genai.Client(api_key="fake")),
        )
        with patch("google.genai.models.AsyncModels._generate_content", fake_generate_content):
            _run(_drain(InMemoryRunner(agent=agent, app_name="geo")))

        model = _one(recording_client, kind="MODEL_EXEC")
        assert model.name == "adk.llm.gemini-2.0-flash"
        attrs = json.loads(model.attributes_json)
        assert attrs[AgenticAttributes.USAGE_INPUT_TOKENS] == 7
        assert attrs[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 2

    def test_gemini_called_outside_adk_is_still_traced(self, recording_client):
        """Suppression is scoped to the ADK model call, not global."""
        instrument("google-genai", recording_client)
        instrument("adk", recording_client)

        with patch("google.genai.models.Models._generate_content", return_value=None, create=True):
            genai.Client(api_key="fake").models.generate_content(
                model="gemini-2.0-flash", contents="hi"
            )
        assert _one(recording_client, kind="MODEL_EXEC").name == "gemini.generate_content"


class TestInstall:
    def test_instrument_and_uninstrument_roundtrip(self, recording_client):
        from google.adk.agents import context as adk_context
        from google.adk.flows.llm_flows import _tool_caller, base_llm_flow

        originals = (
            _tool_caller._call_tool_async,
            _tool_caller._call_tool_in_thread_pool,
            base_llm_flow.BaseLlmFlow._run_and_handle_error,
            adk_context.Context.search_memory,
        )

        assert instrument("adk", recording_client) is True
        try:
            # Both alias and canonical resolve to the same active entry.
            assert get_instrumented_client("adk") is recording_client
            assert get_instrumented_client("google-adk") is recording_client
            # The call sites ADK actually goes through are wrapped.
            assert _tool_caller._call_tool_async is not originals[0]
            assert _tool_caller._call_tool_in_thread_pool is not originals[1]
            assert base_llm_flow.BaseLlmFlow._run_and_handle_error is not originals[2]
            assert adk_context.Context.search_memory is not originals[3]
        finally:
            assert uninstrument("adk") is True
        assert get_instrumented_client("google-adk") is None
        assert (
            _tool_caller._call_tool_async,
            _tool_caller._call_tool_in_thread_pool,
            base_llm_flow.BaseLlmFlow._run_and_handle_error,
            adk_context.Context.search_memory,
        ) == originals

        # After uninstrument, a run emits nothing.
        _run(_drain(_default_runner("quiet")))
        assert recording_client.buffer.spans == []
