"""
Context management for active traces and spans.

Two layers:

* **Thread-local storage** — the default. Tracks the current trace and
  span for ordinary synchronous callers, enabling automatic parent-child
  relationships.
* **Isolated context** — an opt-in ``contextvars`` overlay. While one is
  active, every getter and setter in this module reads and writes the
  overlay instead of thread-local storage.

The overlay exists because thread-local storage cannot tell apart two
asyncio tasks on one event loop: with
``asyncio.gather(runner_a.run_async(), runner_b.run_async())`` both tasks
share a thread, so each would see the other's "current" trace and span.
A ``ContextVar`` is per-task (and is copied into child tasks and into
``contextvars.copy_context().run(...)`` worker threads), so code running
inside an isolated context only ever sees its own run.

The ADK instrumentor is the user of this today: it snapshots the parent
at wrapper entry (``instrumentation.adk.patch._capture_parent``), parents
its spans off that snapshot explicitly, and enters an isolated context
around each resumption of the wrapped ADK coroutine / async generator.
Nothing the ADK wrappers do is ever written to thread-local storage.
"""

import contextlib
import contextvars
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # Avoid circular imports
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace

# Thread-local storage for context
_context = threading.local()

# Isolated overlay: ``None`` when inactive, else ``(trace, span)``. The
# tuple is immutable on purpose — a child task gets a copy of the
# context, so its writes must replace the value rather than mutate
# something the parent task also holds.
_isolated: "contextvars.ContextVar[tuple[DisseqtTrace | None, DisseqtSpan | None] | None]" = (
    contextvars.ContextVar("disseqt_isolated_context", default=None)
)


def enter_isolated_context(
    trace: Optional["DisseqtTrace"], span: Optional["DisseqtSpan"]
) -> "contextvars.Token[Any]":
    """
    Activate an isolated context seeded with ``trace`` / ``span``.

    Until the returned token is passed to :func:`exit_isolated_context`,
    the getters and setters in this module operate on the isolated value
    and leave thread-local storage untouched. Enter and exit must happen
    in the same ``contextvars`` context — in practice, within one
    resumption of a coroutine (never across a ``yield`` of an async
    generator, which hands control back to the caller's context).
    """
    return _isolated.set((trace, span))


def exit_isolated_context(token: "contextvars.Token[Any]") -> None:
    """Restore whatever was active before :func:`enter_isolated_context`."""
    # ValueError: token belongs to a different Context. Nothing sensible
    # to do; observability must not raise into the caller.
    with contextlib.suppress(ValueError, RuntimeError):
        _isolated.reset(token)


def get_current_trace() -> Optional["DisseqtTrace"]:
    """
    Get the current active trace.

    Returns:
        DisseqtTrace or None: The current trace, or None if no trace is active
    """
    isolated = _isolated.get()
    if isolated is not None:
        return isolated[0]
    return getattr(_context, "trace", None)


def set_current_trace(trace: Optional["DisseqtTrace"]) -> None:
    """
    Set the current active trace.

    Args:
        trace: The trace to set as current, or None to clear
    """
    isolated = _isolated.get()
    if isolated is not None:
        _isolated.set((trace, isolated[1]))
        return
    if trace is None:
        if hasattr(_context, "trace"):
            delattr(_context, "trace")
    else:
        _context.trace = trace


def get_current_span() -> Optional["DisseqtSpan"]:
    """
    Get the current active span.

    Returns:
        DisseqtSpan or None: The current span, or None if no span is active
    """
    isolated = _isolated.get()
    if isolated is not None:
        return isolated[1]
    return getattr(_context, "span", None)


def set_current_span(span: Optional["DisseqtSpan"]) -> None:
    """
    Set the current active span.

    Args:
        span: The span to set as current, or None to clear
    """
    isolated = _isolated.get()
    if isolated is not None:
        _isolated.set((isolated[0], span))
        return
    if span is None:
        if hasattr(_context, "span"):
            delattr(_context, "span")
    else:
        _context.span = span


def clear_context() -> None:
    """
    Clear all context (both trace and span).

    Useful for cleanup or when starting a new operation.
    """
    if _isolated.get() is not None:
        _isolated.set((None, None))
        return
    if hasattr(_context, "trace"):
        delattr(_context, "trace")
    if hasattr(_context, "span"):
        delattr(_context, "span")
