"""
Utils Module

Utility functions for ID generation, time conversion, validation, logging, etc.
"""

from .ids import generate_span_id, generate_trace_id
from .logging import get_logger, set_log_level
from .time import (
    calculate_duration_ns,
    from_timestamp_ms,
    from_timestamp_ns,
    now_ms,
    now_ns,
    to_timestamp_ms,
    to_timestamp_ns,
)

__all__ = [
    "generate_trace_id",
    "generate_span_id",
    "now_ns",
    "now_ms",
    "to_timestamp_ns",
    "from_timestamp_ns",
    "to_timestamp_ms",
    "from_timestamp_ms",
    "calculate_duration_ns",
    "get_logger",
    "set_log_level",
]
