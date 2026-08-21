"""
Shared helpers for provider patches.

`ensure_span_parent()` handles the "there might not be an active trace"
problem: if the user hasn't opened a trace, we transparently create one
so the LLM call still gets recorded. If they did open one, we nest under
it. The wrapper this returns must be `.__exit__()`-ed by the caller so
the auto-created trace (if any) is closed and flushed.

Every span opened via `open_llm_span()` also gets an
``agentic.request.duration_ms`` attribute on scope exit, and a warning
log if the wall-clock duration exceeds the configured slow-call
threshold. Configure the threshold with `set_slow_call_threshold_ms()`
(default 300000 ms = 5 minutes) — long enough to skip most legitimate
slow LLM calls, short enough to flag genuinely hung requests.
"""

from __future__ import annotations

import contextlib
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.context import get_current_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.semantics import AgenticAttributes
from disseqt_agentic_sdk.utils.logging import get_logger

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace

_logger = get_logger(__name__)

# Wall-clock duration above which we log a slow-call warning on span exit.
# 5 minutes is generous — legitimate LLM calls with long streaming
# completions can take a couple of minutes; anything past 5 usually means
# a hung connection.
_DEFAULT_SLOW_THRESHOLD_MS = 5 * 60 * 1000
_slow_threshold_ms: float = _DEFAULT_SLOW_THRESHOLD_MS


def set_slow_call_threshold_ms(threshold_ms: float | None) -> None:
    """
    Override the wall-clock threshold above which slow-call warnings fire.

    Pass ``None`` to disable the warning entirely (duration is still
    recorded on the span; you just won't see a log line for slow ones).
    Value is in milliseconds.
    """
    global _slow_threshold_ms
    _slow_threshold_ms = float("inf") if threshold_ms is None else float(threshold_ms)


def get_slow_call_threshold_ms() -> float:
    """Return the current slow-call warning threshold in milliseconds."""
    return _slow_threshold_ms


class _SpanScope:
    """
    Context manager returned by `open_llm_span()`. On exit, records the
    wall-clock duration, emits a slow-call warning if the duration
    exceeded the configured threshold, ends the span, and — if this scope
    also opened the trace — ends the trace so buffered spans flush.
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
        # perf_counter is monotonic — safe against wall-clock jumps.
        self._start = time.perf_counter()

    def __enter__(self) -> DisseqtSpan:
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        duration_ms = (time.perf_counter() - self._start) * 1000
        safe_set(self.span, AgenticAttributes.REQUEST_DURATION_MS, round(duration_ms, 3))
        if duration_ms > _slow_threshold_ms:
            _logger.warning(
                "slow LLM call: %s took %.0f ms (threshold %.0f ms); "
                "may indicate a hung connection",
                self.span.name,
                duration_ms,
                _slow_threshold_ms,
            )
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


def safe_call(fn: Any, *args: Any, **kwargs: Any) -> None:
    """
    Invoke ``fn(*args, **kwargs)`` and log-and-swallow any exception.

    Every provider wrapper reaches into user-supplied kwargs (messages,
    tools, config objects) and provider-returned responses. A malformed
    input or an unexpected response shape must never break the user's
    LLM call — observability code must degrade gracefully.

    Return value from ``fn`` is discarded; use direct invocation when you
    need the result.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        _logger.warning(
            "disseqt instrumentation error in %s: %s",
            getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
            e,
        )


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
