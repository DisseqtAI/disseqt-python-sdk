"""
AGENT_EXEC tool-call aggregation (Lane B of tool-call instrumentation).

Lane A auto-captures tool_calls on ``MODEL_EXEC`` spans — visible to log
queries, but the four tool validators (tool-failure-rate, tool-call-
accuracy, plan-optimality, plan-coherence) only fire on ``AGENT_EXEC``
spans (per validation_consumer.go:1041 in the backend).

This module bridges the gap:

  * ``agent_span(client, name)`` — context manager that opens an
    ``AGENT_EXEC`` span AND registers a per-context aggregator.
  * ``record_tool_result(...)`` — user calls this after their tool runs
    to record success / failure / error / timeout for a specific call_id.
  * Bubble-up — every nested ``MODEL_EXEC`` span that captures
    ``tool_calls`` (via ``_notify_planned_tool_calls``) is merged into
    the current aggregator, so planned calls surface on the AGENT_EXEC
    span even if the user forgets to call ``record_tool_result``.

On ``agent_span`` exit, the aggregator flushes the fused
``[{id, name, arguments, result, status}, ...]`` list to
``agentic.tool_calls`` and ``gen_ai.tool_calls`` on the AGENT_EXEC span,
which is exactly what the backend validators read.

Async safety: contextvar-based, so concurrent asyncio tasks each see
their own aggregator with no bleed-through.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._utils import safe_set
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes
from disseqt_agentic_sdk.utils.logging import get_logger

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient
    from disseqt_agentic_sdk.span import DisseqtSpan

_logger = get_logger(__name__)

# Canonical statuses the tool-failure-rate validator understands.
_VALID_STATUSES = {"success", "failure", "error", "timeout"}


class _ToolCallAggregator:
    """
    Per-agent-span dict keyed by ``call_id`` fusing planned + executed data.

    Planned data (id, name, arguments) arrives via ``add_planned`` when a
    nested MODEL_EXEC span emits ``tool_calls``. Execution outcome (result,
    status) arrives via ``add_result`` when the user calls
    ``record_tool_result``. Both writers coalesce into the same entry so
    validators see the full lifecycle on a single row.
    """

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}
        # Flipped True by flush_onto so writes reaching this aggregator
        # via a captured reference (after the enclosing agent_span
        # already exited) can log instead of mutating a dict no one
        # will ever read from.
        self.closed: bool = False

    def add_planned(self, tool_calls: list[dict[str, Any]]) -> None:
        """Merge planned tool_calls from a nested MODEL_EXEC span."""
        if self.closed:
            # A nested MODEL_EXEC that finished after agent_span exited
            # (e.g. a fire-and-forget async LLM call). Not a user API
            # misuse, so no warning — just skip.
            return
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            call_id = tc.get("id")
            if not call_id:
                continue
            entry = self._calls.setdefault(call_id, {})
            # setdefault per key so a later record_tool_result that supplied
            # richer args/name is preserved.
            for k, v in tc.items():
                entry.setdefault(k, v)

    def add_result(
        self,
        call_id: str,
        *,
        name: str | None = None,
        arguments: Any = None,
        result: Any = None,
        status: str = "success",
    ) -> None:
        """Record execution outcome for one tool call."""
        if self.closed:
            # Reached via a captured reference after agent_span exited —
            # the outcome will never appear on the AGENT_EXEC span. Warn
            # loudly; this is a real API misuse worth surfacing.
            _logger.warning(
                "record_tool_result(call_id=%r, status=%r) reached a "
                "flushed aggregator (agent_span already exited); this "
                "outcome will NOT appear on the AGENT_EXEC span. Move "
                "the record_tool_result call inside the `with agent_span"
                "(...):` block, or await the tool before block exit.",
                call_id,
                status,
            )
            return
        entry = self._calls.setdefault(call_id, {"id": call_id})
        if name is not None:
            entry["name"] = str(name)
        if arguments is not None:
            entry["arguments"] = arguments if isinstance(arguments, str) else str(arguments)
        if result is not None:
            entry["result"] = result if isinstance(result, str) else str(result)
        entry["status"] = status

    def flush_onto(self, span: DisseqtSpan) -> None:
        """Write the fused tool_calls list onto ``span``. Safe to no-op."""
        self.closed = True
        if not self._calls:
            return
        calls = list(self._calls.values())
        safe_set(span, AgenticAttributes.TOOL_CALLS, calls)
        safe_set(span, GenAIAttributes.TOOL_CALLS, calls)


# Ambient aggregator for the current context. Set by ``agent_span``,
# read by ``_notify_planned_tool_calls`` and ``record_tool_result``.
_current_agg: contextvars.ContextVar[_ToolCallAggregator | None] = contextvars.ContextVar(
    "disseqt_tool_agg", default=None
)


def _notify_planned_tool_calls(tool_calls: list[dict[str, Any]]) -> None:
    """
    Called from provider wrappers when tool_calls are captured on a
    MODEL_EXEC span. Adds them to the enclosing agent_span aggregator
    if any is active; otherwise no-op (Lane A capture still happens on
    the MODEL_EXEC span itself).
    """
    agg = _current_agg.get()
    if agg is not None:
        agg.add_planned(tool_calls)


@contextmanager
def agent_span(
    client: DisseqtAgenticClient,
    name: str,
    *,
    agent_name: str | None = None,
) -> Iterator[DisseqtSpan]:
    """
    Open an AGENT_EXEC span with tool-call aggregation enabled.

    Wrap your agent workflow in this so nested LLM tool calls + user
    ``record_tool_result`` invocations get fused into
    ``agentic.tool_calls`` on the AGENT_EXEC span — which is what the
    backend tool validators read.

    If no trace is currently active one is created (and flushed on exit).
    If a trace is active this span nests under it as a child.
    """
    from disseqt_agentic_sdk.context import get_current_trace
    from disseqt_agentic_sdk.trace import DisseqtTrace

    trace = get_current_trace()
    owns_trace = False
    if trace is None:
        trace = DisseqtTrace(
            name=name,
            project_id=client.project_id,
            service_name=client.service_name,
            service_version=client.service_version,
            environment=client.environment,
            realtime_policy_id=client.realtime_policy_id,
            client=client,
        )
        owns_trace = True

    span = trace.start_span(name, SpanKind.AGENT_EXEC)
    if agent_name:
        # Observability code never crashes the caller. set_agent_info can
        # fail on span-shape mismatches — swallow with the same policy
        # used elsewhere (contextlib.suppress for pure swallows, matching
        # the safe_set / safe_call convention).
        with contextlib.suppress(Exception):
            span.set_agent_info(agent_name)

    agg = _ToolCallAggregator()
    token = _current_agg.set(agg)
    try:
        yield span
    finally:
        # Flush aggregated tool_calls onto the AGENT_EXEC span BEFORE
        # the span ends, so the attributes ship with the span.
        agg.flush_onto(span)
        _current_agg.reset(token)
        span.__exit__(None, None, None)
        if owns_trace:
            trace.end()


def record_tool_result(
    call_id: str,
    *,
    name: str | None = None,
    arguments: Any = None,
    result: Any = None,
    status: str = "success",
) -> None:
    """
    Record the execution outcome of a tool call for the enclosing
    ``agent_span``.

    Call this after your tool code runs. Populates the ``result`` +
    ``status`` fields the ``tool-failure-rate`` validator requires;
    together with the planned tool_calls that Lane A auto-captures, this
    is what the tool-call validators score against.

    ``status`` should be one of: success | failure | error | timeout.
    Anything else is passed through with a warning — the backend may
    treat it as unknown.

    Calling outside an ``agent_span`` block logs a warning and no-ops;
    the tool_calls have nowhere to be aggregated onto.
    """
    agg = _current_agg.get()
    if agg is None:
        _logger.warning(
            "record_tool_result(call_id=%r) called outside agent_span(); "
            "wrap your agent workflow in `with agent_span(client, name=...):` "
            "so tool_calls reach the AGENT_EXEC span the validators read.",
            call_id,
        )
        return
    if status not in _VALID_STATUSES:
        _logger.warning(
            "record_tool_result: status=%r is not one of %s; passing through",
            status,
            sorted(_VALID_STATUSES),
        )
    agg.add_result(call_id, name=name, arguments=arguments, result=result, status=status)
