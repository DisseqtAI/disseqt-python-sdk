"""
Context Module

Context management for active traces and spans (thread-local storage).
"""

from .context import (
    clear_context,
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)

__all__ = [
    "get_current_trace",
    "set_current_trace",
    "get_current_span",
    "set_current_span",
    "clear_context",
]
