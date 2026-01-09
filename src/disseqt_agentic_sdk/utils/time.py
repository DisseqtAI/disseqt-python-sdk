"""
Time conversion utilities for nanosecond timestamps.

Handles conversion between datetime objects and nanosecond timestamps
used by OpenTelemetry and the backend.
"""

from datetime import datetime, timezone


def now_ns() -> int:
    """
    Get current time in nanoseconds since Unix epoch.

    Returns:
        int: Current timestamp in nanoseconds
    """
    return int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)


def now_ms() -> int:
    """
    Get current time in milliseconds since Unix epoch.

    Returns:
        int: Current timestamp in milliseconds
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def to_timestamp_ns(dt: datetime) -> int:
    """
    Convert datetime to nanoseconds since Unix epoch.

    Args:
        dt: Datetime object (timezone-aware or naive)

    Returns:
        int: Timestamp in nanoseconds

    Raises:
        ValueError: If datetime is naive and cannot be converted
    """
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def from_timestamp_ns(ns: int) -> datetime:
    """
    Convert nanoseconds since Unix epoch to datetime.

    Args:
        ns: Timestamp in nanoseconds

    Returns:
        datetime: UTC timezone-aware datetime object
    """
    return datetime.fromtimestamp(ns / 1_000_000_000.0, tz=timezone.utc)


def to_timestamp_ms(dt: datetime) -> int:
    """
    Convert datetime to milliseconds since Unix epoch.

    Args:
        dt: Datetime object (timezone-aware or naive)

    Returns:
        int: Timestamp in milliseconds

    Raises:
        ValueError: If datetime is naive and cannot be converted
    """
    if dt.tzinfo is None:
        # Assume UTC if naive
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def from_timestamp_ms(ms: int) -> datetime:
    """
    Convert milliseconds since Unix epoch to datetime.

    Args:
        ms: Timestamp in milliseconds

    Returns:
        datetime: UTC timezone-aware datetime object
    """
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def calculate_duration_ns(start_ns: int, end_ns: int) -> int:
    """
    Calculate duration in nanoseconds between two timestamps.

    Args:
        start_ns: Start timestamp in nanoseconds
        end_ns: End timestamp in nanoseconds

    Returns:
        int: Duration in nanoseconds
    """
    return end_ns - start_ns
