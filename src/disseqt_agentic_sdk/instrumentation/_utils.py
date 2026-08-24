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
import contextvars
import os
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.context import get_current_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._custom_attrs import _get_ambient_attrs
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes
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
# ContextVar so two concurrent asyncio tasks (or threads that copied the
# caller's context) each see their own threshold — a plain module global
# would race last-write-wins between callers.
_slow_threshold_ms: contextvars.ContextVar[float] = contextvars.ContextVar(
    "disseqt_slow_threshold_ms", default=float(_DEFAULT_SLOW_THRESHOLD_MS)
)


def set_slow_call_threshold_ms(threshold_ms: float | None) -> None:
    """
    Override the wall-clock threshold above which slow-call warnings fire.

    Pass ``None`` to disable the warning entirely (duration is still
    recorded on the span; you just won't see a log line for slow ones).
    Value is in milliseconds.

    Scope: writes into the current ``contextvars`` context. Async tasks
    started before the write keep the prior value; tasks started after
    inherit the new one. For a process-wide default, call this at startup
    before any async work begins.
    """
    value = float("inf") if threshold_ms is None else float(threshold_ms)
    _slow_threshold_ms.set(value)


def get_slow_call_threshold_ms() -> float:
    """Return the current slow-call warning threshold in milliseconds."""
    return _slow_threshold_ms.get()


# ---------------------------------------------------------------------
# Content-capture opt-out (privacy / compliance)
# ---------------------------------------------------------------------
# When disabled, the SDK skips writing message contents, completion text,
# tool-call arguments, tool schemas, and system-instruction attributes
# onto spans. Non-content telemetry (model, token counts, duration, tool
# NAMES, tool-call IDs, finish reasons, response IDs) is still captured.
#
# Rationale: some deployments can't ship prompt text or tool arguments
# for compliance (HIPAA, GDPR) or leak-risk reasons (a `send_email` tool
# with an smtp_password arg). Comparable SDKs gate this behind an
# explicit env var — OpenTelemetry GenAI's
# OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT and OpenLLMetry's
# TRACELOOP_TRACE_CONTENT both work this way.
_CONTENT_ATTR_KEYS: frozenset[str] = frozenset(
    {
        AgenticAttributes.INPUT_MESSAGES,
        AgenticAttributes.OUTPUT_MESSAGES,
        AgenticAttributes.SYSTEM_INSTRUCTIONS,
        AgenticAttributes.TOOL_CALLS,
        AgenticAttributes.TOOL_ARGS,
        AgenticAttributes.REQUEST_TOOLS,
        GenAIAttributes.PROMPT,
        GenAIAttributes.COMPLETION,
        GenAIAttributes.TOOL_CALLS,
        GenAIAttributes.TOOL_ARGS,
        GenAIAttributes.REQUEST_TOOLS,
    }
)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean-ish env var. '0'/'false'/'no'/'off' → False."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# ContextVar (not a bare module global) so concurrent callers of
# set_capture_content(...) don't race a shared bool — a bare global
# has zero isolation between threads/tasks, and this feature exists
# specifically to prevent secrets leaking under normal concurrent
# traffic. Mirrors the _slow_threshold_ms pattern above; TP-2128
# round-2 senior review P0 #0.1 (the round-1 fix promoted the
# threshold to a ContextVar but left this sibling as a bare global).
# B039 is a false positive here — _env_bool returns a plain ``bool``,
# which is immutable; the ruff check exists to catch mutable-default
# aliasing footguns (list/dict/set), which don't apply.
_capture_content: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "disseqt_capture_content",
    default=_env_bool("DISSEQT_SDK_CAPTURE_CONTENT", default=True),  # noqa: B039
)


def set_capture_content(enabled: bool) -> None:
    """
    Toggle whether the SDK captures message contents / tool-call arguments
    on auto-instrumented spans.

    Set to ``False`` in privacy-sensitive deployments (HIPAA, GDPR) or
    when tool calls may include credentials. Non-content telemetry
    (model, tokens, duration, tool names/ids, finish reasons) is always
    captured regardless.

    Scope: writes into the current ``contextvars`` context. Async tasks
    started before the write keep the prior value; tasks started after
    inherit the new one. For a process-wide default, call this at
    startup before any async work begins. ThreadPoolExecutor: bare
    ``.submit(fn, ...)`` does NOT propagate — wrap with
    ``executor.submit(contextvars.copy_context().run, fn, ...)`` or
    prefer ``asyncio.loop.run_in_executor`` (which copies context
    automatically). See ``_custom_attrs.py`` docstring for the same
    threading caveat.
    """
    _capture_content.set(bool(enabled))


def get_capture_content() -> bool:
    """Return whether content capture is currently enabled."""
    return _capture_content.get()


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
        threshold_ms = _slow_threshold_ms.get()
        if duration_ms > threshold_ms:
            _logger.warning(
                "slow LLM call: %s took %.0f ms (threshold %.0f ms); "
                "may indicate a hung connection",
                self.span.name,
                duration_ms,
                threshold_ms,
            )
        # Merge user-supplied ambient attributes LAST — after every auto
        # attribute, before span.__exit__. Any key the user set overrides
        # the auto value. safe_set handles None/empty skips and swallows
        # per-attribute errors so a single bad value can't poison the span.
        for k, v in _get_ambient_attrs().items():
            safe_set(self.span, k, v)
        # Delegate to the span's own __exit__ so the incremental-send path runs.
        self.span.__exit__(exc_type, exc_val, exc_tb)
        if self._owns_trace:
            self._trace.end()


def _get_or_bootstrap_trace(
    client: DisseqtAgenticClient,
    name: str,
) -> tuple[DisseqtTrace, bool]:
    """
    Return the current trace (nesting under it) or bootstrap a new one
    from ``client``'s configured project / service / environment / policy.

    Returns ``(trace, owns_trace)`` — ``owns_trace`` is True when we
    created the trace ourselves and the caller is responsible for
    ending it. Used by both ``open_llm_span`` (MODEL_EXEC spans) and
    ``agent_span`` (AGENT_EXEC spans) so the six-keyword bootstrap
    lives in one place (TP-2128 P4 #4.6).
    """
    from disseqt_agentic_sdk.trace import DisseqtTrace

    trace = get_current_trace()
    if trace is not None:
        return trace, False
    return (
        DisseqtTrace(
            name=name,
            project_id=client.project_id,
            service_name=client.service_name,
            service_version=client.service_version,
            environment=client.environment,
            realtime_policy_id=client.realtime_policy_id,
            client=client,
        ),
        True,
    )


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
    trace, owns_trace = _get_or_bootstrap_trace(client, name)
    span = trace.start_span(name, kind)
    return _SpanScope(span=span, trace=trace, owns_trace=owns_trace)


def read(obj: Any, name: str) -> Any:
    """
    Read a field from a provider response tolerating shape drift.

    Provider SDKs occasionally shuffle between Pydantic models
    (attribute access) and plain dicts (key access) across minor
    releases — sometimes within the same response tree. This helper
    unifies both so instrumentors don't need per-provider branches.
    Returns None on missing keys, missing attributes, or ``obj is
    None``; never raises.

    Canonical implementation for the whole instrumentation package —
    ``_oai_compat``, ``_tool_calls``, ``_batches``, ``_embeddings``,
    and ``anthropic/patch`` all used to carry an identical local copy
    (TP-2128 P4 #4.1). They re-export from here now.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def safe_set(span: DisseqtSpan, key: str, value: Any) -> None:
    """
    Set an attribute if the value is non-empty / non-None. Never raises.

    Also honors the content-capture opt-out: when
    ``set_capture_content(False)`` (or ``DISSEQT_SDK_CAPTURE_CONTENT=0``),
    any write to a content-bearing key (prompts, completions, tool-call
    arguments, tool schema, system instructions) is skipped. Non-content
    keys (model, tokens, duration, tool names/ids, finish reasons) are
    unaffected.
    """
    if not _capture_content.get() and key in _CONTENT_ATTR_KEYS:
        return
    if value is None:
        return
    if isinstance(value, str) and not value:
        return
    with contextlib.suppress(Exception):
        span.set_attribute(key, value)


def set_messages_if_capturing(
    span: DisseqtSpan,
    *,
    input_messages: Any = None,
    output_messages: Any = None,
) -> None:
    """
    Wrap ``span.set_messages(...)`` with the content-capture gate.

    Providers should call this instead of ``span.set_messages(...)``
    directly so a single toggle skips message-body writes across every
    instrumented SDK.
    """
    if not _capture_content.get():
        return
    with contextlib.suppress(Exception):
        span.set_messages(input_messages=input_messages, output_messages=output_messages)


def set_first_tool_call_attrs(span: DisseqtSpan, tool_calls: list[dict[str, Any]]) -> None:
    """
    Populate the single-value ``TOOL_NAME`` / ``TOOL_CALL_ID`` /
    ``TOOL_ARGS`` convenience attributes from ``tool_calls[0]`` on
    both the ``agentic.*`` and ``gen_ai.*`` attribute namespaces.

    The full canonical list already lands on ``TOOL_CALLS``; these
    single-value columns exist because the backend's enriched-table
    query surface has dedicated columns for the *first* tool call, so
    dashboards that don't want to unpack the JSON array can still
    read a value. Every provider wrapper used to open-code the six
    ``safe_set`` calls this helper wraps (TP-2128 P4 #4.2).
    """
    if not tool_calls:
        return
    first = tool_calls[0]
    name = first.get("name")
    call_id = first.get("id")
    args = first.get("arguments")
    safe_set(span, AgenticAttributes.TOOL_NAME, name)
    safe_set(span, GenAIAttributes.TOOL_NAME, name)
    safe_set(span, AgenticAttributes.TOOL_CALL_ID, call_id)
    safe_set(span, GenAIAttributes.TOOL_CALL_ID, call_id)
    safe_set(span, AgenticAttributes.TOOL_ARGS, args)
    safe_set(span, GenAIAttributes.TOOL_ARGS, args)


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
