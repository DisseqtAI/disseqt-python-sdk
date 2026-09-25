"""
Tests for header-value validation at the transport layer.

client.client.DisseqtAgenticClient's own validation (see
test_client_header_value_validation.py) only covers callers who go
through the client. HTTPTransport is publicly constructible too, so it
re-validates api_key/application_id in its own __init__ -- a bad value
fails loudly at construction instead of raising an uncaught
UnicodeEncodeError on every send thereafter.
"""

from __future__ import annotations

import pytest

from disseqt_agentic_sdk.transport.http import HTTPTransport


class TestHTTPTransportConstructionValidation:
    """HTTPTransport constructed directly (bypassing DisseqtAgenticClient)."""

    def test_non_latin1_api_key_raises_at_construction(self):
        with pytest.raises(ValueError, match="Latin-1"):
            HTTPTransport(endpoint="http://localhost/v1/traces", api_key="key-\U0001f525-bad")

    def test_non_latin1_application_id_raises_at_construction(self):
        with pytest.raises(ValueError, match="Latin-1"):
            HTTPTransport(
                endpoint="http://localhost/v1/traces",
                api_key="k",
                application_id="app-\U0001f525-bad",
            )

    def test_ordinary_values_construct_fine(self):
        transport = HTTPTransport(
            endpoint="http://localhost/v1/traces", api_key="k", application_id="app-id"
        )
        assert transport.api_key == "k"
