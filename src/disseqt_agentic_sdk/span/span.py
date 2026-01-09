"""
DisseqtSpan - Span class for creating and managing spans.

Handles span lifecycle, attributes, and automatic parent-child relationships.
"""

import json
from datetime import datetime, timezone
from typing import Any

from disseqt_agentic_sdk.context import get_current_span, set_current_span
from disseqt_agentic_sdk.enums import SpanKind, SpanStatus
from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
)
from disseqt_agentic_sdk.utils import calculate_duration_ns, generate_span_id, now_ns


class DisseqtSpan:
    """
    Span class for creating and managing spans.

    Automatically handles:
    - ID generation
    - Timing (start/end)
    - Parent-child relationships via context
    - Building attributes_json from agentic attributes
    """

    def __init__(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        org_id: str = "",
        project_id: str = "",
        user_id: str = "",
        service_name: str = "",
        service_version: str = "1.0.0",
        environment: str = "production",
    ):
        """
        Initialize a new span.

        Args:
            trace_id: Trace ID this span belongs to
            name: Span name
            kind: Span kind (MODEL_EXEC, TOOL_EXEC, AGENT_EXEC, etc.)
            span_id: Optional span ID (auto-generated if not provided)
            parent_span_id: Optional parent span ID (auto-detected from context if not provided)
            org_id: Organization ID
            project_id: Project ID
            user_id: User ID
            service_name: Service name
            service_version: Service version
            environment: Environment (production, staging, dev)
        """
        # Generate span ID if not provided
        self.span_id = span_id or generate_span_id()

        # Get parent from context if not explicitly provided
        # Only use context if parent_span_id was not passed (None means check context)
        # If parent_span_id was explicitly passed as None, use None (root span)
        if parent_span_id is None:
            current_span = get_current_span()
            if current_span is not None:
                # Compare trace IDs as strings to handle UUID/string types
                # Type narrowing: current_span is not None here
                current_trace_id_str = str(current_span.trace_id)  # type: ignore[has-type]
                trace_id_str = str(trace_id)
                if current_trace_id_str == trace_id_str:
                    parent_span_id = current_span.span_id

        # Determine if root span
        self.root = parent_span_id is None
        self.parent_span_id = parent_span_id

        # Timing
        self.start_time_ns = now_ns()
        self.end_time_ns: int | None = None

        # Store parameters
        self.trace_id = trace_id
        self.name = name
        self.kind = kind.value if isinstance(kind, SpanKind) else kind
        self.org_id = org_id
        self.project_id = project_id
        self.user_id = user_id
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment

        # Status
        self.status = SpanStatus.OK
        self.status_message: str = ""

        # Attributes dictionary (will be serialized to attributes_json)
        self.attributes: dict[str, Any] = {}

        # Set current span in context
        set_current_span(self)

    def set_agent_info(
        self,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_version: str | None = None,
    ) -> "DisseqtSpan":
        """
        Set agent information.

        Args:
            agent_name: Agent name
            agent_id: Agent ID
            agent_version: Agent version

        Returns:
            self for method chaining
        """
        if agent_name:
            self.attributes[AgenticAttributes.AGENT_NAME] = agent_name
        if agent_id:
            self.attributes[AgenticAttributes.AGENT_ID] = agent_id
        if agent_version:
            self.attributes[AgenticAttributes.AGENT_VERSION] = agent_version
        return self

    def set_model_info(
        self,
        model_name: str,
        provider: str,
    ) -> "DisseqtSpan":
        """
        Set model/provider information.

        Args:
            model_name: Model name (e.g., "gpt-4", "claude-3")
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            self for method chaining
        """
        self.attributes[AgenticAttributes.REQUEST_MODEL] = model_name
        self.attributes[AgenticAttributes.PROVIDER_NAME] = provider
        return self

    def set_tool_info(
        self,
        tool_name: str,
        tool_call_id: str | None = None,
    ) -> "DisseqtSpan":
        """
        Set tool information.

        Args:
            tool_name: Tool name
            tool_call_id: Optional tool call ID

        Returns:
            self for method chaining
        """
        self.attributes[AgenticAttributes.TOOL_NAME] = tool_name
        if tool_call_id:
            self.attributes[AgenticAttributes.TOOL_CALL_ID] = tool_call_id
        return self

    def set_operation(self, operation: str) -> "DisseqtSpan":
        """
        Set operation name.

        Args:
            operation: Operation name (create_agent, invoke_agent, execute_tool, etc.)

        Returns:
            self for method chaining
        """
        self.attributes[AgenticAttributes.OPERATION_NAME] = operation
        return self

    def set_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> "DisseqtSpan":
        """
        Set token usage.

        Args:
            input_tokens: Input tokens
            output_tokens: Output tokens

        Returns:
            self for method chaining
        """
        self.attributes[AgenticAttributes.USAGE_INPUT_TOKENS] = input_tokens
        self.attributes[AgenticAttributes.USAGE_OUTPUT_TOKENS] = output_tokens
        self.attributes[AgenticAttributes.USAGE_TOTAL_TOKENS] = input_tokens + output_tokens
        return self

    def set_messages(
        self,
        input_messages: list[dict[str, Any]] | None = None,
        output_messages: list[dict[str, Any]] | None = None,
    ) -> "DisseqtSpan":
        """
        Set input/output messages.

        Args:
            input_messages: List of input messages
            output_messages: List of output messages

        Returns:
            self for method chaining
        """
        if input_messages:
            self.attributes[AgenticAttributes.INPUT_MESSAGES] = input_messages
        if output_messages:
            self.attributes[AgenticAttributes.OUTPUT_MESSAGES] = output_messages
        return self

    def set_status(self, status: SpanStatus | str, message: str = "") -> "DisseqtSpan":
        """
        Set span status.

        Args:
            status: Status (OK, ERROR) or string
            message: Optional status message

        Returns:
            self for method chaining
        """
        if isinstance(status, str):
            self.status = SpanStatus(status)
        else:
            self.status = status
        if message:
            self.status_message = message
        return self

    def set_error(self, error_message: str, error_type: str | None = None) -> "DisseqtSpan":
        """
        Set error information.

        Args:
            error_message: Error message
            error_type: Optional error type

        Returns:
            self for method chaining
        """
        self.status = SpanStatus.ERROR
        self.attributes[AgenticAttributes.ERROR_MESSAGE] = error_message
        if error_type:
            self.attributes[AgenticAttributes.ERROR_TYPE] = error_type
        return self

    def set_attribute(self, key: str, value: Any) -> "DisseqtSpan":
        """
        Set a custom attribute.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            self for method chaining
        """
        self.attributes[key] = value
        return self

    def end(self) -> "DisseqtSpan":
        """
        End the span (set end time).

        Returns:
            self for method chaining
        """
        if self.end_time_ns is None:
            self.end_time_ns = now_ns()

        # Clear from context
        current_span = get_current_span()
        if current_span is self:
            set_current_span(None)

        return self

    def to_enriched_span(self) -> EnrichedSpan:
        """
        Convert to EnrichedSpan model for sending to backend.

        Returns:
            EnrichedSpan instance
        """
        # Calculate duration
        end_time = self.end_time_ns if self.end_time_ns else now_ns()
        duration_ns = calculate_duration_ns(self.start_time_ns, end_time)

        # Serialize attributes to JSON
        attributes_json = json.dumps(self.attributes) if self.attributes else "{}"

        # Create EnrichedSpan
        return EnrichedSpan(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            root=self.root,
            start_time_unix_nano=self.start_time_ns,
            end_time_unix_nano=end_time,
            duration_ns=duration_ns,
            status_code=self.status.value if isinstance(self.status, SpanStatus) else self.status,
            org_id=self.org_id,
            project_id=self.project_id,
            user_id=self.user_id,
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
            dt=datetime.now(timezone.utc),
            attributes_json=attributes_json,
            resource_attributes_json="{}",
            events_json="[]",
            ingestion_time=datetime.now(timezone.utc),
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically end span"""
        if exc_type:
            self.set_error(str(exc_val), error_type=exc_type.__name__)
        self.end()
