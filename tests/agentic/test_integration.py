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

    def test_trace_sending(self):
        """Test that traces are sent to backend."""
        with patch("disseqt_agentic_sdk.transport.http.requests.Session") as mock_session_class:
            # Create mock session and response
            mock_session = MagicMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_session.post.return_value = mock_response
            mock_session_class.return_value = mock_session

            # Create client with real buffer and transport (Session will be mocked)
            client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="proj_456",
                service_name="test_service",
            )

            # Directly add a span to buffer to test the transport
            from disseqt_agentic_sdk.models.span import EnrichedSpan

            test_span = EnrichedSpan(
                trace_id="test_trace",
                span_id="test_span",
                name="test",
                project_id="proj_456",
                service_name="test_service",
            )

            # Add span and flush
            client.buffer.add_span(test_span)
            client.flush()

            # Should have attempted to send via HTTP POST
            assert (
                mock_session.post.called
            ), f"Expected HTTP POST to be called. Call count: {mock_session.post.call_count}"

            client.shutdown()

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_context_nesting(self, mock_trace_buffer, mock_http_transport):
        """Test nested span context management."""
        client = DisseqtAgenticClient(
            api_key="test_key", project_id="proj_456", service_name="test_service"
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
