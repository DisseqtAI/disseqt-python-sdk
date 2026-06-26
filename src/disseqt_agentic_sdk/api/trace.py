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

        # Send any remaining spans that weren't sent incrementally
        # (spans are sent incrementally in span.__exit__ if client is available)
        if self._client is not None:
            # Check if incremental sending was enabled (trace has client reference)
            incremental_enabled = self.trace._client is not None

            if incremental_enabled:
                # Incremental sending was enabled - just log completion
                logger.debug(
                    "Trace completed with incremental span sending",
                    extra={
                        "trace_id": self.trace.trace_id,
                        "trace_name": self.trace.name,
                        "span_count": len(self.trace.spans),
                    },
                )
            else:
                # Fallback: send all spans at once (original behavior)
                logger.debug(
                    "Sending trace to backend (batch mode)",
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
    realtime_policy_id: str | None = None,
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
        realtime_policy_id: Per-trace policy override. When set, every
            span in this trace is sent to llm-monitoring with
            ``resource.attributes["policy.id"]`` = this value. Lets two
            agents inside the same application run under different
            policies without re-initializing the client. When omitted,
            falls back to the client's default
            ``realtime_policy_id`` (or no policy at all if the client
            wasn't given one).

    Returns:
        TraceWrapper: Wrapped trace that auto-sends on exit

    Example:
        >>> from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
        >>> from disseqt_agentic_sdk.enums import SpanKind
        >>> # Init the client ONCE for the whole app
        >>> client = DisseqtAgenticClient(
        ...     api_key="...", project_id="...", service_name="my-app"
        ... )
        >>> # Agent A — uses policy A
        >>> with start_trace(client, "agent_a_run",
        ...                  realtime_policy_id="policy-a-uuid") as trace:
        ...     with trace.start_span("plan", SpanKind.AGENT_EXEC) as span:
        ...         span.set_agent_info("agent_a", "a-001")
        >>> # Agent B — same client, different policy
        >>> with start_trace(client, "agent_b_run",
        ...                  realtime_policy_id="policy-b-uuid") as trace:
        ...     with trace.start_span("plan", SpanKind.AGENT_EXEC) as span:
        ...         span.set_agent_info("agent_b", "b-001")
    """
    # Use user_id from trace if provided, otherwise use empty string (client no longer has user_id)
    trace_user_id = user_id if user_id is not None else ""

    # Per-trace policy override beats client default. Empty string here
    # means "fall back to client.realtime_policy_id at transport time".
    effective_policy_id = realtime_policy_id or getattr(client, "realtime_policy_id", None) or ""

    trace = DisseqtTrace(
        name=name,
        trace_id=trace_id,
        org_id="",  # Set by backend middleware for localhost
        project_id=client.project_id,
        user_id=trace_user_id,
        service_name=client.service_name,
        service_version=client.service_version,
        environment=client.environment,
        intent_id=intent_id,
        workflow_id=workflow_id,
        realtime_policy_id=effective_policy_id,
        client=client,  # Pass client for incremental span sending
    )

    return TraceWrapper(trace, client)
