"""
Tests for the header-safety validation on ``api_key``.

It now travels as an HTTP header (X-Api-Key) on every trace POST,
alongside ``application_id`` (X-Application-Id, covered separately in
test_client_application_id.py). Both route through the same
``_validate_header_value`` check (client.py, generalized from the
application_id-only check) so a malformed value in either fails loudly at construction instead of
reaching ``http.client.putheader`` and raising an uncaught
``UnicodeEncodeError`` on every send thereafter.

These tests mirror test_client_application_id.py's structure and use
the same mocked-transport fixture -- they prove the fail-fast-at-
construction behavior specifically. They deliberately do NOT prove
that a background flush thread survives an unvalidated bad value
reaching send time -- that's a different failure mode, covered by the
real (unmocked) thread test in test_buffer_flush_resilience.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient


@pytest.fixture(autouse=True)
def _stub_transport(monkeypatch):
    """Stub transport/buffer so tests don't hit the network."""
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", MagicMock())
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.TraceBuffer", MagicMock())


def _make_client(**overrides):
    kwargs = {
        "api_key": "test_key",
        "service_name": "test_service",
        "endpoint": "http://localhost/v1/traces",
        "application_id": "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
    }
    kwargs.update(overrides)
    return DisseqtAgenticClient(**kwargs)


class TestApiKeyHeaderSafety:
    def test_newline_in_api_key_raises(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(api_key="key-with-\nnewline")

    def test_carriage_return_in_api_key_raises(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(api_key="key\r\nX-Injected: 1")

    def test_non_latin1_character_in_api_key_raises(self):
        with pytest.raises(ValueError, match="Latin-1"):
            _make_client(api_key="key-\U0001f525-secret")

    def test_ordinary_api_key_accepted(self):
        client = _make_client(api_key="secret-key-42")
        assert client.api_key == "secret-key-42"


class TestErrorMessagesNameTheField:
    """
    Two fields now share one validator -- confirm the error message
    still names the actual offending field, not a generic/wrong one,
    so a customer debugging a construction failure isn't misdirected.
    """

    def test_api_key_error_names_api_key(self):
        with pytest.raises(ValueError, match="api_key contains"):
            _make_client(api_key="bad-\U0001f525-key")

    def test_application_id_error_still_names_application_id(self):
        with pytest.raises(ValueError, match="application_id contains"):
            _make_client(application_id="bad-\U0001f525-app")
