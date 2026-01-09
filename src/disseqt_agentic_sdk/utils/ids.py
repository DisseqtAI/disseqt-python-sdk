"""
ID generation utilities for traces and spans.

Generates UUIDs matching backend format expectations.
"""

from uuid import uuid4


def generate_trace_id() -> str:
    """
    Generate a unique trace ID.

    Returns a UUID string without dashes (32-char hex string).
    Matches backend format: strings.ReplaceAll(uuid.New().String(), "-", "")

    Returns:
        str: 32-character hexadecimal trace ID
    """
    return uuid4().hex


def generate_span_id() -> str:
    """
    Generate a unique span ID.

    Returns a UUID string without dashes, truncated to 16 chars.
    Matches backend format: strings.ReplaceAll(uuid.New().String(), "-", "")[:16]

    Returns:
        str: 16-character hexadecimal span ID
    """
    return uuid4().hex[:16]
