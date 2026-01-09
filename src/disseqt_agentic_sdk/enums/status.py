"""
SpanStatus enum - defines the status of a span.

Matches backend status_code values used in the database schema.
"""

from enum import Enum


class SpanStatus(str, Enum):
    """
    Span status - indicates success or failure of the span operation.

    Values match the backend schema.
    """

    OK = "OK"  # Operation completed successfully
    ERROR = "ERROR"  # Operation failed with an error
