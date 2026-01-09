"""
Unit tests for DisseqtSpan.
"""

import pytest

from disseqt_agentic_sdk.enums import SpanKind, SpanStatus
from disseqt_agentic_sdk.semantics import AgenticAttributes
from disseqt_agentic_sdk.span import DisseqtSpan


class TestDisseqtSpan:
    """Tests for DisseqtSpan class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = DisseqtSpan(
            trace_id="test_trace_123",
            name="test_span",
            kind=SpanKind.INTERNAL,
            org_id="org_1",
            project_id="proj_1",
            service_name="test_service",
        )

        assert span.trace_id == "test_trace_123"
        assert span.name == "test_span"
        assert span.kind == "INTERNAL"
        assert span.org_id == "org_1"
        assert span.project_id == "proj_1"
        assert span.service_name == "test_service"
        assert span.status == SpanStatus.OK
        assert span.start_time_ns > 0
        assert span.end_time_ns is None

    def test_span_root_detection(self):
        """Test root span detection."""
        # Clear context first to ensure no parent from context
        from disseqt_agentic_sdk.context import clear_context

        clear_context()

        # Root span (no parent)
        root_span = DisseqtSpan(
            trace_id="test_trace", name="root", kind=SpanKind.INTERNAL, parent_span_id=None
        )
        assert root_span.root is True
        assert root_span.parent_span_id is None

        # Child span (with parent)
        child_span = DisseqtSpan(
            trace_id="test_trace",
            name="child",
            kind=SpanKind.INTERNAL,
            parent_span_id="parent_span_123",
        )
        assert child_span.root is False
        assert child_span.parent_span_id == "parent_span_123"

    def test_span_attributes(self):
        """Test setting span attributes."""
        span = DisseqtSpan(trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL)

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)
        span.set_attribute("key3", {"nested": "data"})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123
        assert span.attributes["key3"] == {"nested": "data"}

    def test_span_agent_info(self):
        """Test setting agent information."""
        span = DisseqtSpan(trace_id="test_trace", name="agent_span", kind=SpanKind.AGENT_EXEC)

        span.set_agent_info(agent_name="assistant", agent_id="agent_001", agent_version="1.0.0")

        assert span.attributes[AgenticAttributes.AGENT_NAME] == "assistant"
        assert span.attributes[AgenticAttributes.AGENT_ID] == "agent_001"
        assert span.attributes[AgenticAttributes.AGENT_VERSION] == "1.0.0"

    def test_span_model_info(self):
        """Test setting model information."""
        span = DisseqtSpan(trace_id="test_trace", name="llm_span", kind=SpanKind.MODEL_EXEC)

        span.set_model_info(model_name="gpt-4", provider="openai")

        assert span.attributes[AgenticAttributes.REQUEST_MODEL] == "gpt-4"
        assert span.attributes[AgenticAttributes.PROVIDER_NAME] == "openai"

    def test_span_tool_info(self):
        """Test setting tool information."""
        span = DisseqtSpan(trace_id="test_trace", name="tool_span", kind=SpanKind.TOOL_EXEC)

        span.set_tool_info(tool_name="calculator", tool_call_id="call_001")

        assert span.attributes[AgenticAttributes.TOOL_NAME] == "calculator"
        assert span.attributes[AgenticAttributes.TOOL_CALL_ID] == "call_001"

    def test_span_token_usage(self):
        """Test setting token usage."""
        span = DisseqtSpan(trace_id="test_trace", name="llm_span", kind=SpanKind.MODEL_EXEC)

        span.set_token_usage(input_tokens=100, output_tokens=50)

        assert span.attributes[AgenticAttributes.USAGE_INPUT_TOKENS] == 100
        assert span.attributes[AgenticAttributes.USAGE_OUTPUT_TOKENS] == 50
        assert span.attributes[AgenticAttributes.USAGE_TOTAL_TOKENS] == 150

    def test_span_messages(self):
        """Test setting messages."""
        span = DisseqtSpan(trace_id="test_trace", name="llm_span", kind=SpanKind.MODEL_EXEC)

        input_msgs = [{"role": "user", "content": "Hello"}]
        output_msgs = [{"role": "assistant", "content": "Hi"}]

        span.set_messages(input_messages=input_msgs, output_messages=output_msgs)

        assert span.attributes[AgenticAttributes.INPUT_MESSAGES] == input_msgs
        assert span.attributes[AgenticAttributes.OUTPUT_MESSAGES] == output_msgs

    def test_span_error(self):
        """Test setting error on span."""
        span = DisseqtSpan(trace_id="test_trace", name="error_span", kind=SpanKind.INTERNAL)

        span.set_error(error_message="Something went wrong", error_type="ValueError")

        assert span.status == SpanStatus.ERROR
        assert span.attributes[AgenticAttributes.ERROR_MESSAGE] == "Something went wrong"
        assert span.attributes[AgenticAttributes.ERROR_TYPE] == "ValueError"

    def test_span_end(self):
        """Test ending a span."""
        span = DisseqtSpan(trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL)

        start_time = span.start_time_ns
        assert span.end_time_ns is None

        span.end()

        assert span.end_time_ns is not None
        assert span.end_time_ns >= start_time

    def test_span_context_manager(self):
        """Test span as context manager."""
        with DisseqtSpan(trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL) as span:
            assert span.start_time_ns > 0

        # Span should be ended after context exit
        assert span.end_time_ns is not None

    def test_span_context_manager_error(self):
        """Test span context manager with error."""
        with pytest.raises(ValueError):
            with DisseqtSpan(
                trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL
            ) as span:
                raise ValueError("Test error")

        # Span should be marked as error
        assert span.status == SpanStatus.ERROR
        assert span.end_time_ns is not None
