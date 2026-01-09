"""
DisseqtTrace - Trace class for creating and managing traces.

A trace is a collection of spans representing a complete workflow.
"""

from typing import Any

from disseqt_agentic_sdk.context import get_current_trace, set_current_trace
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.span import DisseqtSpan
from disseqt_agentic_sdk.utils import generate_trace_id, now_ns


class DisseqtTrace:
    """
    Trace class for creating and managing traces.

    A trace represents a complete workflow and contains multiple spans.
    Automatically manages span collection and parent-child relationships.
    """

    def __init__(
        self,
        name: str,
        trace_id: str | None = None,
        org_id: str = "",
        project_id: str = "",
        user_id: str = "",
        service_name: str = "",
        service_version: str = "1.0.0",
        environment: str = "production",
        intent_id: str | None = None,
        workflow_id: str | None = None,
    ):
        """
        Initialize a new trace.

        Args:
            name: Trace name
            trace_id: Optional trace ID (auto-generated if not provided)
            org_id: Organization ID
            project_id: Project ID
            user_id: User ID
            service_name: Service name
            service_version: Service version
            environment: Environment (production, staging, dev)
            intent_id: Optional intent ID
            workflow_id: Optional workflow ID
        """
        # Generate trace ID if not provided
        self.trace_id = trace_id or generate_trace_id()

        # Timing
        self.start_time_ns = now_ns()
        self.end_time_ns: int | None = None

        # Store parameters
        self.name = name
        self.org_id = org_id
        self.project_id = project_id
        self.user_id = user_id
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.intent_id = intent_id
        self.workflow_id = workflow_id

        # Span collection
        self.spans: list[DisseqtSpan] = []
        self.is_ended = False

        # Set current trace in context
        set_current_trace(self)

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> DisseqtSpan:
        """
        Start a new span in this trace.

        Args:
            name: Span name
            kind: Span kind (MODEL_EXEC, TOOL_EXEC, AGENT_EXEC, etc.) or string like "MODEL_EXEC"
            span_id: Optional span ID (auto-generated if not provided)
            parent_span_id: Optional parent span ID (auto-detected from context if not provided)

        Returns:
            DisseqtSpan: The created span

        Raises:
            RuntimeError: If trace has already been ended
        """
        if self.is_ended:
            raise RuntimeError("Cannot start a new span on an ended trace.")

        # Convert string to SpanKind if needed
        if isinstance(kind, str):
            kind = SpanKind(kind)

        span = DisseqtSpan(
            trace_id=self.trace_id,
            name=name,
            kind=kind,
            span_id=span_id,
            parent_span_id=parent_span_id,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=self.user_id,
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
        )

        # Add intent_id and workflow_id to span attributes if provided
        if self.intent_id:
            span.set_attribute("agentic.intent.id", self.intent_id)
        if self.workflow_id:
            span.set_attribute("agentic.workflow.id", self.workflow_id)

        self.spans.append(span)
        return span

    def set_intent_id(self, intent_id: str) -> "DisseqtTrace":
        """
        Set intent ID for this trace.

        Args:
            intent_id: Intent ID

        Returns:
            self for method chaining
        """
        self.intent_id = intent_id
        # Update all existing spans
        for span in self.spans:
            span.set_attribute("agentic.intent.id", intent_id)
        return self

    def set_workflow_id(self, workflow_id: str) -> "DisseqtTrace":
        """
        Set workflow ID for this trace.

        Args:
            workflow_id: Workflow ID

        Returns:
            self for method chaining
        """
        self.workflow_id = workflow_id
        # Update all existing spans
        for span in self.spans:
            span.set_attribute("agentic.workflow.id", workflow_id)
        return self

    def end(self) -> "DisseqtTrace":
        """
        End the trace (set end time).

        Returns:
            self for method chaining
        """
        if self.is_ended:
            return self

        self.is_ended = True

        if self.end_time_ns is None:
            self.end_time_ns = now_ns()

        # End all spans that haven't been ended
        for span in self.spans:
            if span.end_time_ns is None:
                span.end()

        # Clear from context
        current_trace = get_current_trace()
        if current_trace is self:
            set_current_trace(None)

        return self

    def get_spans(self) -> list[DisseqtSpan]:
        """
        Get all spans in this trace.

        Returns:
            List of DisseqtSpan objects
        """
        return self.spans.copy()

    def to_enriched_spans(self) -> list[EnrichedSpan]:
        """
        Convert all spans to EnrichedSpan models for sending to backend.

        Returns:
            List of EnrichedSpan instances
        """
        return [span.to_enriched_span() for span in self.spans]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert trace to dictionary format.

        Returns:
            Dictionary representation of the trace
        """
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "intent_id": self.intent_id,
            "workflow_id": self.workflow_id,
            "span_count": len(self.spans),
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically end trace"""
        self.end()
