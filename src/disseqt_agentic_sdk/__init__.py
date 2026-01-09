"""
Disseqt Agentic SDK

A Python SDK for creating traces, spans, and monitoring agentic AI workflows
with the Disseqt LLM Monitoring & Protection backend.

This SDK provides:
- Trace and span creation
- Automatic span context management
- Integration with backend PostgreSQL schema
- Support for agentic AI semantic conventions

Quick Start:
    from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
    from disseqt_agentic_sdk.enums import SpanKind

    # Initialize SDK client
    client = DisseqtAgenticClient(
        api_key="your-key",
        org_id="org_123",
        project_id="proj_456",
        service_name="my-service",
        endpoint="http://localhost:8080/v1/traces"
    )

    # Create traces with spans
    with start_trace(client, "user_request", intent_id="intent_123") as trace:
        with trace.start_span("agent_planning", SpanKind.AGENT_EXEC) as span:
            span.set_agent_info("assistant", "agent_001")

        with trace.start_span("llm_call", SpanKind.MODEL_EXEC) as span:
            span.set_model_info("gpt-4", "openai")
            span.set_token_usage(100, 50)
"""

__version__ = "0.1.0"

# Public API
from disseqt_agentic_sdk.api import (
    start_trace,
    trace_agent_action,
    trace_function,
    trace_llm_call,
    trace_tool_call,
)
from disseqt_agentic_sdk.client import DisseqtAgenticClient

# Enums
from disseqt_agentic_sdk.enums import (
    SpanKind,
    SpanStatus,
    SpanType,
)

# Semantics
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticCacheOperation,
    AgenticFinishReason,
    AgenticOperation,
    AgenticOutputType,
    AgenticProvider,
)
from disseqt_agentic_sdk.span import DisseqtSpan

# Classes (for advanced usage)
from disseqt_agentic_sdk.trace import DisseqtTrace

# Logging utilities
from disseqt_agentic_sdk.utils.logging import get_logger, set_log_level

__all__ = [
    "__version__",
    # Public API
    "start_trace",
    # Helper functions
    "trace_llm_call",
    "trace_agent_action",
    "trace_tool_call",
    "trace_function",
    # Enums
    "SpanKind",
    "SpanStatus",
    "SpanType",
    # Classes
    "DisseqtTrace",
    "DisseqtSpan",
    "DisseqtAgenticClient",
    # Semantics
    "AgenticAttributes",
    "AgenticOperation",
    "AgenticOutputType",
    "AgenticFinishReason",
    "AgenticProvider",
    "AgenticCacheOperation",
    # Logging
    "get_logger",
    "set_log_level",
]
