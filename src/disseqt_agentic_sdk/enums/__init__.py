"""
Enums Module

Enumeration types for span kinds, status codes, and other constants.
"""

from .span_kind import SpanKind
from .status import SpanStatus

__all__ = [
    "SpanKind",
    "SpanStatus",
]
