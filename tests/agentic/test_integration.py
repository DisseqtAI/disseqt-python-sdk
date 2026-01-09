"""
Integration tests for end-to-end SDK functionality.
"""

from unittest.mock import MagicMock, Mock, patch

from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind


class TestIntegration:
    """Integration tests for SDK workflow."""

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_full_trace_workflow(self, mock_trace_buffer, mock_http_transport):
        """Test complete trace creation and sending workflow."""
        client = DisseqtAgenticClient(
            api_key="test_key",
            org_id="org_123",
            project_id="proj_456",
            service_name="test_service",
            endpoint="http://localhost:8080/v1/traces",
        )

        with start_trace(client, "integration_test", intent_id="intent_123") as trace:
            # Root span
            with trace.start_span("root_span", SpanKind.AGENT_EXEC) as root:
                root.set_agent_info("test_agent", "agent_001")

                # Child span
                with trace.start_span("child_span", SpanKind.MODEL_EXEC) as child:
                    child.set_model_info("gpt-4", "openai")
                    child.set_token_usage(100, 50)

        # Trace should be sent automatically
        assert len(trace.spans) == 2
        assert trace.spans[0].root is True
        assert trace.spans[1].root is False
        assert trace.spans[1].parent_span_id == trace.spans[0].span_id

        client.shutdown()

    @patch("disseqt_agentic_sdk.transport.http.HTTPTransport")
    @patch("disseqt_agentic_sdk.buffer.buffer.TraceBuffer")
    def test_trace_sending(self, mock_trace_buffer_class, mock_transport_class):
        """Test that traces are sent to backend."""
        mock_transport_instance = MagicMock()
        mock_transport_instance.send_spans.return_value = True
        mock_transport_class.return_value = mock_transport_instance

        mock_buffer_instance = MagicMock()
        mock_buffer_instance.start = Mock()
        mock_trace_buffer_class.return_value = mock_buffer_instance

        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        # Replace the buffer's transport with our mock
        client.buffer.transport = mock_transport_instance

        with start_trace(client, "test_trace") as trace:
            span = trace.start_span("test_span", SpanKind.INTERNAL)
            span.end()

        # Flush to ensure spans are sent
        client.flush()

        # Should have attempted to send
        assert mock_transport_instance.send_spans.called

        client.shutdown()

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_context_nesting(self, mock_trace_buffer, mock_http_transport):
        """Test nested span context management."""
        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        with start_trace(client, "nested_test") as trace:
            # Level 1
            with trace.start_span("level1", SpanKind.INTERNAL) as span1:
                assert span1.root is True

                # Level 2
                with trace.start_span("level2", SpanKind.INTERNAL) as span2:
                    assert span2.root is False
                    assert span2.parent_span_id == span1.span_id

                    # Level 3
                    with trace.start_span("level3", SpanKind.INTERNAL) as span3:
                        assert span3.root is False
                        assert span3.parent_span_id == span2.span_id

        assert len(trace.spans) == 3

        client.shutdown()
