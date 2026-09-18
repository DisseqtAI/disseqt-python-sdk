"""
Google ADK (Agent Development Kit) instrumentor.

Targets ``google-adk`` >= 1.32.0 (the release that consolidated telemetry
under ``google.adk.telemetry.tracing`` — earlier layouts split the helpers
across ``google.adk.flows.llm_flows.functions`` / ``base_agent.tracer`` and
would require a version-branched patch table).

Patch targets (mirroring openinference-instrumentation-google-adk, the
only production-tested prior art for ADK — Traceloop and Langtrace do
not ship one as of Sep 2026):

  * ``google.adk.runners.Runner.run_async`` — AGENT_EXEC root span,
    wraps the returned ``AsyncGenerator[Event, None]``. ``Runner.run``
    (sync) delegates to ``run_async`` on a background thread, which
    hits this wrapper on that thread; a fresh trace is bootstrapped
    there since ``threading.local`` context doesn't cross threads.
  * ``google.adk.agents.base_agent.BaseAgent.run_async`` — AGENT_EXEC
    per-agent span (nested under the Runner span when present). Every
    concrete agent (``LlmAgent``, ``SequentialAgent``, ``ParallelAgent``,
    ``LoopAgent``, ``LangGraphAgent``, ``RemoteA2aAgent``) inherits
    this method, so one patch covers all agent kinds.
  * ``google.adk.tools.base_tool.BaseTool.run_async`` — TOOL_EXEC span.
    ``FunctionTool``, ``AgentTool``, MCP tools, LangChain / CrewAI
    tool wrappers all override this single choke point.
  * ``google.adk.models.base_llm.BaseLlm.generate_content_async`` —
    MODEL_EXEC span, wraps the async generator that yields ``LlmResponse``
    chunks (both non-streaming and streaming go through this single
    method; ``stream=True`` yields partials + one final aggregated
    ``partial=False`` chunk).
  * ``google.adk.memory.base_memory_service.BaseMemoryService.search_memory``
    — RAG_EXEC span. All memory services (``InMemoryMemoryService``,
    ``VertexAiRagMemoryService``, ``VertexAiMemoryBankService``) override
    this abstract method, so one patch covers the tree.
  * ``google.adk.agents.remote_a2a_agent.RemoteA2aAgent._run_async_impl``
    — CLIENT span nested under the ``BaseAgent.run_async`` AGENT_EXEC
    span. Captures the A2A network hop between processes so multi-agent
    architectures show as a proper service map without needing manual
    ``@disseqt_trace(kind=CLIENT)`` wrappers.

Not patched:

  * ``Runner.run`` (sync) — delegates to ``run_async`` in a background
    thread; patching both would double-emit the top-level AGENT_EXEC.
  * ``Runner.run_live`` / ``BaseAgent.run_live`` — bidi audio/video is
    a distinct control flow and rare in agent workloads today. Can be
    added later if there's demand.

ADK ships its own OTel telemetry via ``google.adk.telemetry.tracing``.
Those spans go to whatever OpenTelemetry exporter the user has configured
(if any) — they do NOT collide with the disseqt spans we emit because we
send to our own client buffer, not through the OTel SDK. openinference
has to suppress ADK's spans specifically because it also uses the OTel
tracer; we do not.
"""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation.adk.patch import (
    a2a_run_async_impl,
    agent_run_async,
    llm_generate_content_async,
    memory_search_memory,
    runner_run_async,
    tool_run_async,
)
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor


class AdkInstrumentor(DisseqtInstrumentor):
    package_name = "google-adk"
    # Floor picked to avoid the pre-1.32 telemetry split — see module
    # docstring. Matches openinference's declared ``_instruments`` floor.
    min_version = "1.32.0"

    def _instrument(self) -> None:
        self._wrap(
            "google.adk.runners",
            "Runner.run_async",
            runner_run_async(self),
        )
        self._wrap(
            "google.adk.agents.base_agent",
            "BaseAgent.run_async",
            agent_run_async(self),
        )
        self._wrap(
            "google.adk.tools.base_tool",
            "BaseTool.run_async",
            tool_run_async(self),
        )
        self._wrap(
            "google.adk.models.base_llm",
            "BaseLlm.generate_content_async",
            llm_generate_content_async(self),
        )
        self._wrap(
            "google.adk.memory.base_memory_service",
            "BaseMemoryService.search_memory",
            memory_search_memory(self),
        )
        self._wrap(
            "google.adk.agents.remote_a2a_agent",
            "RemoteA2aAgent._run_async_impl",
            a2a_run_async_impl(self),
        )
