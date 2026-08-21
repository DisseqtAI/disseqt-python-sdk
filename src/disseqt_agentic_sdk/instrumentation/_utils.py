"""
Shared helpers for provider patches.

`ensure_span_parent()` handles the "there might not be an active trace"
problem: if the user hasn't opened a trace, we transparently create one
so the LLM call still gets recorded. If they did open one, we nest under
it. The wrapper this returns must be `.__exit__()`-ed by the caller so
the auto-created trace (if any) is closed and flushed.
"""

from __future__ import annotations

import contextlib
from types import TracebackType
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.context import get_current_trace
from disseqt_agentic_sdk.enums import SpanKind

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace


class _SpanScope:
    """
    Context manager returned by `open_llm_span()`. On exit, ends the span
    and — if this scope also opened the trace — ends the trace so buffered
    spans flush.
    """

    def __init__(
        self,
        span: DisseqtSpan,
        trace: DisseqtTrace,
        owns_trace: bool,
    ) -> None:
        self.span = span
        self._trace = trace
        self._owns_trace = owns_trace

    def __enter__(self) -> DisseqtSpan:
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Delegate to the span's own __exit__ so the incremental-send path runs.
        self.span.__exit__(exc_type, exc_val, exc_tb)
        if self._owns_trace:
            self._trace.end()


def open_llm_span(
    client: DisseqtAgenticClient,
    name: str,
    kind: SpanKind | str = SpanKind.MODEL_EXEC,
) -> _SpanScope:
    """
    Start (or reuse) a trace and open a span for an LLM call. Callers use
    the returned scope as a context manager and set attributes on
    `scope.span`.
    """
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

    span = trace.start_span(name, kind)
    return _SpanScope(span=span, trace=trace, owns_trace=owns_trace)


def safe_set(span: DisseqtSpan, key: str, value: Any) -> None:
    """Set an attribute if the value is non-empty / non-None. Never raises."""
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    with contextlib.suppress(Exception):
        span.set_attribute(key, value)


def serialize_messages(messages: Any) -> list[dict[str, Any]] | None:
    """
    Normalize a chat-messages input into `[{"role": str, "content": str}, ...]`.
    Accepts OpenAI-style dicts, Anthropic-style dicts, or Pydantic model
    instances (via `.model_dump()`). Returns None if messages is falsy.
    """
    if not messages:
        return None
    out: list[dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            out.append(msg.model_dump())
        elif isinstance(msg, dict):
            out.append(dict(msg))
        else:
            out.append({"role": "unknown", "content": str(msg)})
    return out
