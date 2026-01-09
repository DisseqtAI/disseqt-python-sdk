"""
Context management for active traces and spans.

Uses thread-local storage to track the current trace and span,
enabling automatic parent-child relationships.
"""

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Avoid circular imports
    from disseqt_agentic_sdk.span import DisseqtSpan
    from disseqt_agentic_sdk.trace import DisseqtTrace

# Thread-local storage for context
_context = threading.local()


def get_current_trace() -> Optional["DisseqtTrace"]:
    """
    Get the current active trace from thread-local storage.

    Returns:
        DisseqtTrace or None: The current trace, or None if no trace is active
    """
    return getattr(_context, "trace", None)


def set_current_trace(trace: Optional["DisseqtTrace"]) -> None:
    """
    Set the current active trace in thread-local storage.

    Args:
        trace: The trace to set as current, or None to clear
    """
    if trace is None:
        if hasattr(_context, "trace"):
            delattr(_context, "trace")
    else:
        _context.trace = trace


def get_current_span() -> Optional["DisseqtSpan"]:
    """
    Get the current active span from thread-local storage.

    Returns:
        DisseqtSpan or None: The current span, or None if no span is active
    """
    return getattr(_context, "span", None)


def set_current_span(span: Optional["DisseqtSpan"]) -> None:
    """
    Set the current active span in thread-local storage.

    Args:
        span: The span to set as current, or None to clear
    """
    if span is None:
        if hasattr(_context, "span"):
            delattr(_context, "span")
    else:
        _context.span = span


def clear_context() -> None:
    """
    Clear all context (both trace and span) from thread-local storage.

    Useful for cleanup or when starting a new operation.
    """
    if hasattr(_context, "trace"):
        delattr(_context, "trace")
    if hasattr(_context, "span"):
        delattr(_context, "span")
