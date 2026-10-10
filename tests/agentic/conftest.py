"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.context import clear_context

# Add src directory to Python path for src layout
# This allows tests to import the package without installing it
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_sdk():
    """
    Reset SDK state before and after each test.

    Several tests across this suite construct a bare DisseqtTrace/
    DisseqtSpan directly (no client, no `with`, no explicit .end()) to
    exercise a specific method in isolation. Since contextvars.ContextVar
    (see disseqt_agentic_sdk/context/context.py) correctly restores the
    PREVIOUS current-trace/current-span on a proper end()/__exit__ -- not
    just "clear to None" -- a test that never ends what it constructed
    leaves it genuinely current for whatever test runs next, and that
    leak now compounds across tests instead of being silently papered
    over. clear_context() here guarantees no test starts with another
    test's leftover trace/span implicitly "current".
    """
    clear_context()
    yield
    clear_context()


@pytest.fixture
@patch("disseqt_agentic_sdk.client.HTTPTransport")
@patch("disseqt_agentic_sdk.client.TraceBuffer")
def initialized_client(mock_trace_buffer, mock_http_transport):
    """Fixture providing initialized SDK client."""
    client = DisseqtAgenticClient(
        api_key="test_key",
        project_id="test_proj",
        service_name="test_service",
        endpoint="http://localhost:8080/v1/traces",
        application_id="test-app-id",
    )
    yield client
    client.shutdown()
