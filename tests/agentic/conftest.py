"""
Pytest configuration and fixtures.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient

# Add src directory to Python path for src layout
# This allows tests to import the package without installing it
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_sdk():
    """Reset SDK state before each test."""
    yield
    # Cleanup after test if needed


@pytest.fixture
@patch("disseqt_agentic_sdk.client.HTTPTransport")
@patch("disseqt_agentic_sdk.client.TraceBuffer")
def initialized_client(mock_trace_buffer, mock_http_transport):
    """Fixture providing initialized SDK client."""
    client = DisseqtAgenticClient(
        api_key="test_key",
        org_id="test_org",
        project_id="test_proj",
        service_name="test_service",
        endpoint="http://localhost:8080/v1/traces",
    )
    yield client
    client.shutdown()
