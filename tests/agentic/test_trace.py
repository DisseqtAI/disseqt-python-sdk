"""
Unit tests for DisseqtTrace.
"""

import pytest

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.trace import DisseqtTrace


class TestDisseqtTrace:
    """Tests for DisseqtTrace class."""

    def test_trace_creation(self):
        """Test basic trace creation."""
        trace = DisseqtTrace(
            name="test_trace", org_id="org_1", project_id="proj_1", service_name="test_service"
        )

        assert trace.name == "test_trace"
        assert trace.org_id == "org_1"
        assert trace.project_id == "proj_1"
        assert trace.service_name == "test_service"
        assert len(trace.trace_id) == 32  # Hex string
        assert len(trace.spans) == 0
        assert trace.start_time_ns > 0
        assert trace.end_time_ns is None
        assert trace.is_ended is False

    def test_trace_custom_id(self):
        """Test trace with custom ID."""
        custom_id = "custom_trace_123"
        trace = DisseqtTrace(name="test_trace", trace_id=custom_id)

        assert trace.trace_id == custom_id

    def test_trace_start_span(self):
        """Test starting spans in trace."""
        trace = DisseqtTrace(name="test_trace", org_id="org_1", project_id="proj_1")

        span1 = trace.start_span("span1", SpanKind.INTERNAL)
        span2 = trace.start_span("span2", SpanKind.INTERNAL)

        assert len(trace.spans) == 2
        assert span1 in trace.spans
        assert span2 in trace.spans
        assert span1.trace_id == trace.trace_id
        assert span2.trace_id == trace.trace_id

    def test_trace_start_span_parent_child(self):
        """Test parent-child relationships in spans."""
        trace = DisseqtTrace(name="test_trace")

        root_span = trace.start_span("root", SpanKind.INTERNAL)
        assert root_span.root is True
        assert root_span.parent_span_id is None

        # Child span should have root as parent
        with root_span:
            child_span = trace.start_span("child", SpanKind.INTERNAL)
            assert child_span.root is False
            assert child_span.parent_span_id == root_span.span_id

    def test_trace_intent_workflow(self):
        """Test setting intent and workflow IDs."""
        trace = DisseqtTrace(name="test_trace", intent_id="intent_123", workflow_id="workflow_456")

        assert trace.intent_id == "intent_123"
        assert trace.workflow_id == "workflow_456"

        # Create span - should inherit intent/workflow
        span = trace.start_span("test_span", SpanKind.INTERNAL)
        assert span.attributes.get("agentic.intent.id") == "intent_123"
        assert span.attributes.get("agentic.workflow.id") == "workflow_456"

    def test_trace_end(self):
        """Test ending a trace."""
        trace = DisseqtTrace(name="test_trace")

        span1 = trace.start_span("span1", SpanKind.INTERNAL)
        span2 = trace.start_span("span2", SpanKind.INTERNAL)

        # Spans not ended yet
        assert span1.end_time_ns is None
        assert span2.end_time_ns is None

        trace.end()

        # Trace should be ended
        assert trace.is_ended is True
        assert trace.end_time_ns is not None

        # All spans should be ended
        assert span1.end_time_ns is not None
        assert span2.end_time_ns is not None

    def test_trace_context_manager(self):
        """Test trace as context manager."""
        with DisseqtTrace(name="test_trace") as trace:
            span = trace.start_span("test_span", SpanKind.INTERNAL)
            assert len(trace.spans) == 1

        # Trace should be ended after context exit
        assert trace.is_ended is True
        assert span.end_time_ns is not None

    def test_trace_to_enriched_spans(self):
        """Test converting trace to enriched spans."""
        trace = DisseqtTrace(name="test_trace", org_id="org_1", project_id="proj_1")

        span1 = trace.start_span("span1", SpanKind.INTERNAL)
        span2 = trace.start_span("span2", SpanKind.INTERNAL)

        span1.end()
        span2.end()
        trace.end()

        enriched_spans = trace.to_enriched_spans()

        assert len(enriched_spans) == 2
        assert enriched_spans[0].trace_id == trace.trace_id
        assert enriched_spans[1].trace_id == trace.trace_id

    def test_trace_cannot_start_span_after_end(self):
        """Test that spans cannot be started after trace ends."""
        trace = DisseqtTrace(name="test_trace")
        trace.end()

        with pytest.raises(RuntimeError, match="Cannot start a new span on an ended trace"):
            trace.start_span("new_span", SpanKind.INTERNAL)
