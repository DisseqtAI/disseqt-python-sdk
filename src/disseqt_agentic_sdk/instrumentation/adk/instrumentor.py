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
  * Tools — TOOL_EXEC span, in two layers that de-duplicate per tool
    instance (one call, one span, however many wrappers it passes):

      1. The call sites the LLM flow uses for *every* tool, whatever its
         class: ``_call_tool_async`` and ``_call_tool_in_thread_pool``.
         These are private and have moved between releases —
         ``flows/llm_flows/functions.py`` (1.x, where the first is
         spelled ``__call_tool_async``), ``flows/llm_flows/_tool_caller.py``
         (2.9), ``flows/llm_flows/tools/_caller.py`` and
         ``tools/_thread_pool.py`` (2.14) — so they are found by *name*
         in whichever ``google.adk.flows.llm_flows`` modules the
         installed ADK has, and every module that holds a binding is
         patched (a ``from ._caller import _call_tool_async`` in a
         sibling module is its own binding).
      2. ``run_async`` on ``BaseTool`` and on every subclass that
         overrides it — the ones already imported, and any defined
         later (``BaseTool.__init_subclass__``). ``McpTool``,
         ``RestApiTool``, the LangChain / CrewAI adapters and user tools
         override ``run_async`` without calling ``super()``, so patching
         the base class alone misses them. This layer depends only on
         ADK's public tool API, so tools stay traced on a release whose
         call sites have moved again; a warning is logged when layer 1
         finds nothing.
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

import contextlib
import importlib
import inspect
import sys
import weakref
from collections.abc import Callable
from typing import Any

import wrapt

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
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger(__name__)

# Where ADK keeps the LLM flow, and the names of the functions in it that
# run a tool. The modules are tried in turn (a missing one is normal: it
# belongs to another ADK release) and are only a way to get the flow
# imported — the call sites themselves are found by name, in every loaded
# module of the package. ``__call_tool_async`` is the 1.x spelling.
_TOOL_FLOW_PACKAGE = "google.adk.flows.llm_flows"
_TOOL_FLOW_MODULES = (
    "functions",
    "_tool_caller",
    "tools._caller",
    "tools._thread_pool",
    "tools._live_caller",
)
_TOOL_CALL_SITE_NAMES = (
    "_call_tool_async",
    "__call_tool_async",
    "_call_tool_in_thread_pool",
)


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
        # Tools: the flow's call sites (cover every tool, and keep the
        # parent when a tool runs in ADK's thread pool), then run_async
        # on the tool classes. Wrappers de-duplicate per tool instance —
        # see patch._active_tool.
        call_sites = self._wrap_tool_call_sites(tool_call_site(self))
        self._wrap_tool_classes(tool_run_async(self))
        if call_sites == 0:
            logger.warning(
                f"google-adk {self.version}: none of the tool call sites "
                f"({', '.join(_TOOL_CALL_SITE_NAMES)}) were found under "
                f"{_TOOL_FLOW_PACKAGE}; tools are traced through "
                "BaseTool.run_async only. Please report this google-adk version."
            )

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

    def _uninstrument(self) -> None:
        self._unwrap_tool_classes()

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def _wrap_tool_call_sites(self, wrapper: Callable[..., Any]) -> int:
        """
        Wrap every binding of a tool call site in the LLM flow package and
        return how many were wrapped.

        Looked up by function name, not module path: the functions are
        private and ADK has moved them twice (see the module docstring).
        A module that re-exports one (``from ._caller import
        _call_tool_async``) holds its own binding, which its callers use,
        so each module is patched separately.
        """
        for suffix in _TOOL_FLOW_MODULES:
            with contextlib.suppress(Exception):
                importlib.import_module(f"{_TOOL_FLOW_PACKAGE}.{suffix}")

        prefix = _TOOL_FLOW_PACKAGE + "."
        wrapped = 0
        for module_name in sorted(sys.modules):
            if not (module_name == _TOOL_FLOW_PACKAGE or module_name.startswith(prefix)):
                continue
            # A failed import can leave None in sys.modules.
            namespace = getattr(sys.modules.get(module_name), "__dict__", None) or {}
            for attr in _TOOL_CALL_SITE_NAMES:
                target = namespace.get(attr)
                # Skip what is not there, and what is already wrapped (by
                # us or by another library).
                if isinstance(target, wrapt.FunctionWrapper) or not inspect.iscoroutinefunction(
                    target
                ):
                    continue
                if self._wrap(module_name, attr, wrapper):
                    wrapped += 1
        return wrapped

    def _wrap_tool_classes(self, wrapper: Callable[..., Any]) -> None:
        """
        Wrap ``run_async`` on ``BaseTool`` and on every subclass that
        defines its own — those imported so far, and those defined later.
        """
        self._tool_class_patches: list[tuple[weakref.ref[type], Any]] = []
        self._tool_base: type | None = None
        self._tool_base_prev_hook: Any = None
        self._tool_base_had_hook = False

        try:
            base = importlib.import_module("google.adk.tools.base_tool").BaseTool
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self.package_name}: skip tool class patches: {e}")
            return
        # The common concrete tools live in their own modules; import them
        # so the walk below sees them. Missing ones are skipped.
        for name in ("function_tool", "agent_tool", "long_running_tool"):
            with contextlib.suppress(Exception):
                importlib.import_module(f"google.adk.tools.{name}")

        seen: set[type] = set()
        stack: list[type] = [base]
        while stack:
            cls = stack.pop()
            if cls in seen:
                continue
            seen.add(cls)
            self._wrap_tool_class(cls, wrapper)
            with contextlib.suppress(Exception):
                stack.extend(cls.__subclasses__())

        # Subclasses defined from now on (a user's own tools, tool modules
        # ADK imports lazily such as the MCP tools).
        instrumentor = self
        prev = base.__dict__.get("__init_subclass__")

        def __init_subclass__(cls: type, **kwargs: Any) -> None:
            if prev is not None:
                prev.__get__(None, cls)(**kwargs)
            else:
                super(base, cls).__init_subclass__(**kwargs)  # type: ignore[misc]
            with contextlib.suppress(Exception):
                instrumentor._wrap_tool_class(cls, wrapper)

        try:
            base.__init_subclass__ = classmethod(__init_subclass__)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self.package_name}: tool subclasses defined later are not patched: {e}")
            return
        self._tool_base = base
        self._tool_base_prev_hook = prev
        self._tool_base_had_hook = prev is not None

    def _wrap_tool_class(self, cls: type, wrapper: Callable[..., Any]) -> None:
        """Wrap ``cls.run_async`` if ``cls`` itself defines it as a coroutine function."""
        own = cls.__dict__.get("run_async")
        if own is None or isinstance(own, wrapt.FunctionWrapper):
            return
        if not inspect.iscoroutinefunction(own):
            return
        installed = wrapt.FunctionWrapper(own, wrapper)
        try:
            setattr(cls, "run_async", installed)  # noqa: B010 — mypy: method assignment
        except Exception as e:  # noqa: BLE001
            logger.debug(f"{self.package_name}: skip patch {cls.__qualname__}.run_async: {e}")
            return
        self._tool_class_patches.append((weakref.ref(cls), installed))

    def _unwrap_tool_classes(self) -> None:
        base = getattr(self, "_tool_base", None)
        if base is not None:
            with contextlib.suppress(Exception):
                if self._tool_base_had_hook:
                    base.__init_subclass__ = self._tool_base_prev_hook
                else:
                    del base.__init_subclass__
            self._tool_base = None
        for ref, installed in getattr(self, "_tool_class_patches", []):
            cls = ref()
            if cls is None:
                continue
            # Only when ours is still the one in place; another library's
            # wrapper on top is left alone.
            if cls.__dict__.get("run_async") is installed:
                with contextlib.suppress(Exception):
                    setattr(cls, "run_async", installed.__wrapped__)  # noqa: B010
        self._tool_class_patches = []
