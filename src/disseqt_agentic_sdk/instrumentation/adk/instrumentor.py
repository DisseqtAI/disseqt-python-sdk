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
  * Tools — TOOL_EXEC span. Primary targets are the two call sites the
    LLM flow uses for *every* tool, whatever its class:
    ``google.adk.flows.llm_flows._tool_caller._call_tool_async`` and
    ``._call_tool_in_thread_pool``. ``run_async`` on ``BaseTool`` and the
    common concrete classes is patched as well, as a backstop for tools
    run directly and for ADK releases that predate ``_tool_caller``. A
    per-class list alone cannot be complete: ``McpTool``, ``RestApiTool``
    and the LangChain / CrewAI adapters override ``run_async`` without
    calling ``super()``. The layers de-duplicate per tool instance, so a
    call produces one span however many wrappers it passes through.
  * Models — MODEL_EXEC span. Primary target is
    ``BaseLlmFlow._run_and_handle_error``, the generator the LLM flow
    wraps around ``llm.generate_content_async(...)`` on every model turn
    (non-streaming and SSE streaming alike; ``stream=True`` yields
    partials + one final aggregated ``partial=False`` chunk). Only when
    that method does not exist is ``generate_content_async`` patched on
    ``BaseLlm`` and the common concrete classes instead — never both,
    since each would open a span for the same turn. While the model call
    is in flight the provider-SDK instrumentors underneath (google-genai,
    litellm, anthropic, openai) are suppressed, so ``instrument_all``
    reports one MODEL_EXEC per turn, not two.
  * Memory — RAG_EXEC span. Primary target is
    ``google.adk.agents.context.Context.search_memory``, the single place
    ADK calls ``memory_service.search_memory(...)``. ``search_memory`` on
    ``BaseMemoryService`` and the shipped services is patched as a
    backstop, de-duplicated per service instance like tools.
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
    llm_call_site,
    llm_generate_content_async,
    memory_call_site,
    memory_search_memory,
    runner_run_async,
    tool_call_site,
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
        # Tools: the flow's call sites first (cover every tool class),
        # then run_async on the classes as a backstop. Wrappers
        # de-duplicate per tool instance — see patch._active_tool.
        tool_site = tool_call_site(self)
        for attr in ("_call_tool_async", "_call_tool_in_thread_pool"):
            self._wrap("google.adk.flows.llm_flows._tool_caller", attr, tool_site)
        tool_wrapper = tool_run_async(self)
        for module, attr in (
            ("google.adk.tools.base_tool", "BaseTool.run_async"),
            ("google.adk.tools.function_tool", "FunctionTool.run_async"),
            ("google.adk.tools.agent_tool", "AgentTool.run_async"),
            ("google.adk.tools.long_running_tool", "LongRunningFunctionTool.run_async"),
        ):
            self._wrap(module, attr, tool_wrapper)

        # Models: the flow's call site, or — only if this ADK has no such
        # method — generate_content_async on the classes. Not both: the
        # class method is called before the call site sees its generator,
        # so the two cannot de-duplicate and would double-report a turn.
        if not self._wrap(
            "google.adk.flows.llm_flows.base_llm_flow",
            "BaseLlmFlow._run_and_handle_error",
            llm_call_site(self),
        ):
            llm_wrapper = llm_generate_content_async(self)
            for module, attr in (
                ("google.adk.models.base_llm", "BaseLlm.generate_content_async"),
                ("google.adk.models.google_llm", "Gemini.generate_content_async"),
                ("google.adk.models.lite_llm", "LiteLlm.generate_content_async"),
                ("google.adk.models.anthropic_llm", "Claude.generate_content_async"),
            ):
                self._wrap(module, attr, llm_wrapper)

        # Memory: the Context call site, then the service classes as a
        # backstop (de-duplicated per service instance).
        self._wrap("google.adk.agents.context", "Context.search_memory", memory_call_site(self))
        memory_wrapper = memory_search_memory(self)
        for module, attr in (
            ("google.adk.memory.base_memory_service", "BaseMemoryService.search_memory"),
            ("google.adk.memory.in_memory_memory_service", "InMemoryMemoryService.search_memory"),
            (
                "google.adk.memory.vertex_ai_rag_memory_service",
                "VertexAiRagMemoryService.search_memory",
            ),
            (
                "google.adk.memory.vertex_ai_memory_bank_service",
                "VertexAiMemoryBankService.search_memory",
            ),
        ):
            self._wrap(module, attr, memory_wrapper)

        self._wrap(
            "google.adk.agents.remote_a2a_agent",
            "RemoteA2aAgent._run_async_impl",
            a2a_run_async_impl(self),
        )
