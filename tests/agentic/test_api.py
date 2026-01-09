"""
Unit tests for public API.
"""

from unittest.mock import MagicMock, patch

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient, start_trace
from disseqt_agentic_sdk.enums import SpanKind


class TestPublicAPI:
    """Tests for public API functions."""

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_client_initialization(self, mock_trace_buffer_class, mock_http_transport_class):
        """Test DisseqtAgenticClient initialization."""
        mock_buffer_instance = MagicMock()
        mock_trace_buffer_class.return_value = mock_buffer_instance

        client = DisseqtAgenticClient(
            api_key="test_key",
            org_id="org_123",
            project_id="proj_456",
            service_name="test_service",
            endpoint="http://localhost:8080/v1/traces",
        )

        assert client is not None
        assert client.org_id == "org_123"
        assert client.project_id == "proj_456"
        assert client.service_name == "test_service"
        mock_http_transport_class.assert_called_once()
        mock_trace_buffer_class.assert_called_once()
        # Verify buffer was created
        assert client.buffer is mock_buffer_instance

    def test_client_missing_api_key(self):
        """Test client fails with missing api_key."""
        with pytest.raises(TypeError):
            DisseqtAgenticClient(
                org_id="org_123", project_id="proj_456", service_name="test_service"
            )

    def test_client_missing_org_id(self):
        """Test client fails with missing org_id."""
        with pytest.raises(TypeError):
            DisseqtAgenticClient(api_key="key", project_id="proj_456", service_name="test_service")

    def test_client_missing_project_id(self):
        """Test client fails with missing project_id."""
        with pytest.raises(TypeError):
            DisseqtAgenticClient(api_key="key", org_id="org_123", service_name="test_service")

    def test_client_missing_service_name(self):
        """Test client fails with missing service_name."""
        with pytest.raises(TypeError):
            DisseqtAgenticClient(api_key="key", org_id="org_123", project_id="proj_456")

    def test_client_empty_api_key(self):
        """Test client fails with empty api_key."""
        with pytest.raises(ValueError, match="api_key is required"):
            DisseqtAgenticClient(
                api_key="", org_id="org_123", project_id="proj_456", service_name="test"
            )

    def test_client_empty_endpoint(self):
        """Test client fails with empty endpoint."""
        with pytest.raises(ValueError, match="endpoint is required"):
            DisseqtAgenticClient(
                api_key="key",
                org_id="org_123",
                project_id="proj_456",
                service_name="test",
                endpoint="",
            )

    def test_client_empty_environment(self):
        """Test client fails with empty environment."""
        with pytest.raises(ValueError, match="environment is required"):
            DisseqtAgenticClient(
                api_key="key",
                org_id="org_123",
                project_id="proj_456",
                service_name="test",
                environment="",
            )

    @patch("disseqt_agentic_sdk.transport.http.HTTPTransport")
    @patch("disseqt_agentic_sdk.buffer.buffer.TraceBuffer")
    def test_start_trace(self, mock_trace_buffer, mock_http_transport):
        """Test starting a trace."""
        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        with start_trace(client, "test_trace", intent_id="intent_123") as trace:
            assert trace.name == "test_trace"
            assert trace.intent_id == "intent_123"
            assert trace.org_id == "org_123"
            assert trace.project_id == "proj_456"

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_start_trace_with_spans(self, mock_trace_buffer, mock_http_transport):
        """Test starting a trace with spans."""
        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        with start_trace(client, "test_trace") as trace:
            span = trace.start_span("test_span", SpanKind.INTERNAL)
            assert span.name == "test_span"
            assert span.trace_id == trace.trace_id
            assert len(trace.spans) == 1

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_client_flush(self, mock_trace_buffer, mock_http_transport):
        """Test flushing spans."""
        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        # Flush should not raise error
        client.flush()

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_client_shutdown(self, mock_trace_buffer, mock_http_transport):
        """Test SDK shutdown."""
        client = DisseqtAgenticClient(
            api_key="test_key", org_id="org_123", project_id="proj_456", service_name="test_service"
        )

        # Shutdown should not raise error
        client.shutdown()
