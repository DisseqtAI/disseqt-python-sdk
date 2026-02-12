"""
SpanKind enum - defines the type/category of a span.

Matches backend span_kind values used in the database schema.
"""

from enum import Enum


class SpanKind(str, Enum):
    """
    Span kind - determines how the span is categorized and enriched.

    Values match the backend schema and OTEL conventions.

    Custom span kinds:
        You can use custom span kinds by passing any string value directly to start_span()
        or trace_function(). The SDK accepts both SpanKind enum values and string values.

        Examples:
            # Using enum value
            trace.start_span("operation", SpanKind.MODEL_EXEC)

            # Using custom string
            trace.start_span("custom_operation", "CUSTOM_KIND")
            trace.start_span("data_processing", "DATA_PROCESSING")

            # With decorator
            @trace_function(client, kind="CUSTOM_OPERATION")
            def my_function():
                pass

    Standard span kinds:
        - MODEL_EXEC: LLM model execution (e.g., GPT-4, Claude)
        - TOOL_EXEC: Tool/function execution (e.g., calculator, API call)
        - AGENT_EXEC: Agent workflow execution
        - RAG_EXEC: RAG (Retrieval Augmented Generation) execution - required for RAG validations
        - MCP_EXEC: MCP (Model Context Protocol) execution
        - INTERNAL: Internal operations
        - CLIENT, SERVER, PRODUCER, CONSUMER: Standard OTLP span kinds
    """

    # Agentic AI specific kinds
    MODEL_EXEC = "MODEL_EXEC"  # LLM model execution (e.g., GPT-4, Claude)
    TOOL_EXEC = "TOOL_EXEC"  # Tool/function execution (e.g., calculator, API call)
    AGENT_EXEC = "AGENT_EXEC"  # Agent workflow execution
    RAG_EXEC = "RAG_EXEC"  # RAG (Retrieval Augmented Generation) execution - required for RAG validations
    MCP_EXEC = "MCP_EXEC"  # MCP (Model Context Protocol) execution
    INTERNAL = "INTERNAL"  # Internal operations

    # Standard OTLP span kinds
    CLIENT = "CLIENT"  # Client span
    SERVER = "SERVER"  # Server span
    PRODUCER = "PRODUCER"  # Producer span (messaging)
    CONSUMER = "CONSUMER"  # Consumer span (messaging)
