"""
Context management for active traces and spans.

Uses contextvars.ContextVar (not threading.local) to track the current
trace and span, enabling automatic parent-child relationships.

Why not threading.local(): a single OS thread can interleave many
asyncio Tasks (the event loop switches between them at every `await`).
threading.local() has exactly one slot per THREAD, shared by every Task
running on it -- so two concurrent async flows (e.g. two in-flight
requests in one async web server) would stomp each other's "current
trace"/"current span" pointer. contextvars.ContextVar gives each asyncio
Task its own isolated copy (Tasks fork the Context they were created in
and never see another Task's writes to it), which is exactly the
isolation boundary needed here. A new OS thread still starts with an
empty Context either way, so this is not a behavior change for
plain multi-threaded (non-async) callers.
"""

import contextvars
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace

logger = logging.getLogger(__name__)

_current_trace: "contextvars.ContextVar[DisseqtTrace | None]" = contextvars.ContextVar(
    "disseqt_current_trace", default=None
)
_current_span: "contextvars.ContextVar[DisseqtSpan | None]" = contextvars.ContextVar(
    "disseqt_current_span", default=None
)


def get_current_trace() -> "DisseqtTrace | None":
    """
    Get the current active trace from context.

    Returns:
        DisseqtTrace or None: The current trace, or None if no trace is active
    """
    return _current_trace.get()


def set_current_trace(trace: "DisseqtTrace | None") -> "contextvars.Token":
    """
    Set the current active trace in context.

    Args:
        trace: The trace to set as current, or None to clear

    Returns:
        A Token. Pass it to reset_current_trace() to restore whatever
        trace was current immediately before this call, in this same
        logical context (asyncio Task / thread) -- the caller doesn't
        need to separately remember the previous value.
    """
    return _current_trace.set(trace)


def reset_current_trace(token: "contextvars.Token") -> None:
    """
    Restore the trace that was current before the matching set_current_trace()
    call, using the Token it returned.

    Safe to call even if the matching set_current_trace() happened in a
    different context (e.g. a trace was constructed in one asyncio Task
    and ended from another) -- falls back to clearing this context's
    slot to None rather than raising, since there's no previous value to
    restore.
    """
    try:
        _current_trace.reset(token)
    except ValueError:
        logger.debug(
            "reset_current_trace: token was set in a different context; "
            "clearing this context's current trace instead of restoring"
        )
        _current_trace.set(None)


def get_current_span() -> "DisseqtSpan | None":
    """
    Get the current active span from context.

    Returns:
        DisseqtSpan or None: The current span, or None if no span is active
    """
    return _current_span.get()


def set_current_span(span: "DisseqtSpan | None") -> "contextvars.Token":
    """
    Set the current active span in context.

    Args:
        span: The span to set as current, or None to clear

    Returns:
        A Token. Pass it to reset_current_span() to restore whatever
        span was current immediately before this call, in this same
        logical context (asyncio Task / thread).
    """
    return _current_span.set(span)


def reset_current_span(token: "contextvars.Token") -> None:
    """
    Restore the span that was current before the matching set_current_span()
    call, using the Token it returned.

    Same different-context fallback as reset_current_trace(): clears
    rather than raising if the token can't be reset here.
    """
    try:
        _current_span.reset(token)
    except ValueError:
        logger.debug(
            "reset_current_span: token was set in a different context; "
            "clearing this context's current span instead of restoring"
        )
        _current_span.set(None)


def clear_context() -> None:
    """
    Clear all context (both trace and span) in the current context.

    Useful for cleanup or when starting a new operation. Unlike
    reset_current_trace()/reset_current_span(), this always clears to
    None rather than restoring a previous value -- there's no token to
    restore from.
    """
    _current_trace.set(None)
    _current_span.set(None)
