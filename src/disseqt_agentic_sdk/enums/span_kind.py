"""
SpanKind enum - defines the type/category of a span.

Matches backend span_kind values used in the database schema.
"""

from enum import Enum


class SpanKind(str, Enum):
    """
    Span kind - determines how the span is categorized and enriched.

    Values match the backend schema and OTEL conventions.
    """

    # Agentic AI specific kinds
    MODEL_EXEC = "MODEL_EXEC"  # LLM model execution (e.g., GPT-4, Claude)
    TOOL_EXEC = "TOOL_EXEC"  # Tool/function execution (e.g., calculator, API call)
    AGENT_EXEC = "AGENT_EXEC"  # Agent workflow execution
    INTERNAL = "INTERNAL"  # Internal operations

    # Standard OTLP span kinds
    CLIENT = "CLIENT"  # Client span
    SERVER = "SERVER"  # Server span
    PRODUCER = "PRODUCER"  # Producer span (messaging)
    CONSUMER = "CONSUMER"  # Consumer span (messaging)

    # Additional kinds (if needed)
    SPAN_KIND_UNSPECIFIED = "SPAN_KIND_UNSPECIFIED"  # Unspecified kind
