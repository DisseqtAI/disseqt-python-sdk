"""
Unit tests for API client helper functions.
"""

from unittest.mock import MagicMock, patch

import pytest
from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.api.client import (
    flush,
    get_current_client,
    get_client,
    is_initialized,
    set_client,
    shutdown,
)


class TestAPIClientHelpers:
    """Tests for API client helper functions."""

    def test_get_current_client_when_initialized(self):
        """Test get_current_client returns client when initialized."""
        with patch("disseqt_agentic_sdk.client.client.HTTPTransport"), patch(
            "disseqt_agentic_sdk.client.client.TraceBuffer"
        ):
            client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="test_proj",
                service_name="test_service",
            )
            set_client(client)

            retrieved_client = get_current_client()
            assert retrieved_client is not None
            assert retrieved_client.project_id == "test_proj"

            client.shutdown()
            set_client(None)

    def test_get_current_client_when_not_initialized(self):
        """Test get_current_client raises RuntimeError when not initialized."""
        set_client(None)

        with pytest.raises(RuntimeError, match="SDK not initialized"):
            get_current_client()

    def test_flush_when_initialized(self):
        """Test flush works when client is initialized."""
        with patch("disseqt_agentic_sdk.client.client.HTTPTransport"), patch(
            "disseqt_agentic_sdk.client.client.TraceBuffer"
        ):
            client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="test_proj",
                service_name="test_service",
            )
            set_client(client)

            # Mock flush method
            client.flush = MagicMock()

            flush()
            client.flush.assert_called_once()

            client.shutdown()
            set_client(None)

    def test_flush_when_not_initialized(self):
        """Test flush raises RuntimeError when not initialized."""
        set_client(None)

        with pytest.raises(RuntimeError, match="SDK not initialized"):
            flush()

    def test_shutdown_when_initialized(self):
        """Test shutdown works when client is initialized."""
        with patch("disseqt_agentic_sdk.client.client.HTTPTransport"), patch(
            "disseqt_agentic_sdk.client.client.TraceBuffer"
        ):
            client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="test_proj",
                service_name="test_service",
            )
            set_client(client)

            # Mock shutdown method
            client.shutdown = MagicMock()

            shutdown()
            client.shutdown.assert_called_once()

            # Verify client was cleared
            assert get_client() is None

    def test_shutdown_when_not_initialized(self):
        """Test shutdown raises RuntimeError when not initialized."""
        set_client(None)

        with pytest.raises(RuntimeError, match="SDK not initialized"):
            shutdown()

    def test_is_initialized_true(self):
        """Test is_initialized returns True when client is set."""
        with patch("disseqt_agentic_sdk.client.client.HTTPTransport"), patch(
            "disseqt_agentic_sdk.client.client.TraceBuffer"
        ):
            client = DisseqtAgenticClient(
                api_key="test_key",
                project_id="test_proj",
                service_name="test_service",
            )
            set_client(client)

            assert is_initialized() is True

            client.shutdown()
            set_client(None)

    def test_is_initialized_false(self):
        """Test is_initialized returns False when client is not set."""
        set_client(None)

        assert is_initialized() is False

