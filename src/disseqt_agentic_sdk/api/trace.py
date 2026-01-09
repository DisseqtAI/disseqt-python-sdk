"""
start_trace() - Create and start a new trace.

High-level function for creating traces using a DisseqtAgenticClient instance.
"""


from disseqt_agentic_sdk.client import DisseqtAgenticClient
from disseqt_agentic_sdk.trace import DisseqtTrace
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger()


class TraceWrapper:
    """
    Wrapper around DisseqtTrace that automatically sends trace on exit.
    """

    def __init__(self, trace: DisseqtTrace, client: DisseqtAgenticClient):
        self.trace = trace
        self._client = client

    def __getattr__(self, name):
        """Delegate all attribute access to wrapped trace"""
        return getattr(self.trace, name)

    def __enter__(self):
        """Context manager entry"""
        return self.trace.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - send trace and cleanup"""
        # End trace (ends all spans)
        self.trace.__exit__(exc_type, exc_val, exc_tb)

        # Send trace to backend
        if self._client is not None:
            logger.debug(
                "Sending trace to backend",
                extra={
                    "trace_id": self.trace.trace_id,
                    "trace_name": self.trace.name,
                    "span_count": len(self.trace.spans),
                },
            )
            self._client.send_trace(self.trace)
        else:
            logger.warning("No client available to send trace")

        return False


def start_trace(
    client: DisseqtAgenticClient,
    name: str,
    trace_id: str | None = None,
    intent_id: str | None = None,
    workflow_id: str | None = None,
    user_id: str | None = None,
) -> TraceWrapper:
    """
    Start a new trace using a DisseqtAgenticClient instance.

    The trace is automatically sent to the backend when the context manager exits.

    Args:
        client: DisseqtAgenticClient instance (required)
        name: Trace name
        trace_id: Optional trace ID (auto-generated if not provided)
        intent_id: Optional intent ID
        workflow_id: Optional workflow ID
        user_id: Optional user ID (overrides default from client)

    Returns:
        TraceWrapper: Wrapped trace that auto-sends on exit

    Example:
        >>> from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
        >>> from disseqt_agentic_sdk.enums import SpanKind
        >>> client = DisseqtAgenticClient(
        ...     api_key="...", org_id="...", project_id="...", service_name="..."
        ... )
        >>> with start_trace(client, "user_request", intent_id="intent_123") as trace:
        ...     with trace.start_span("agent_planning", SpanKind.AGENT_EXEC) as span:
        ...         span.set_agent_info("assistant", "agent_001")
        # Trace is automatically sent when exiting the 'with' block
    """
    # Use user_id from trace if provided, otherwise use client default
    trace_user_id = user_id if user_id is not None else client.user_id

    trace = DisseqtTrace(
        name=name,
        trace_id=trace_id,
        org_id=client.org_id,
        project_id=client.project_id,
        user_id=trace_user_id,
        service_name=client.service_name,
        service_version=client.service_version,
        environment=client.environment,
        intent_id=intent_id,
        workflow_id=workflow_id,
    )

    return TraceWrapper(trace, client)
