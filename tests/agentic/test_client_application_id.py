"""
Tests for the required-``application_id`` contract on
``DisseqtAgenticClient``.

``application_id`` is a **required, keyword-only** constructor argument.
Every trace POST needs the ``X-Application-Id`` header for Kong's
traces-auth check; a client that can't deliver spans is worse than one
that refuses to construct. Missing / empty / whitespace-only raises
``ValueError`` at construction; malformed values (embedded line breaks,
non-Latin-1 characters) raise a different, header-encoding-specific
``ValueError`` — the same character-class checks the round-3 review
tightened to match what ``requests`` / ``http.client`` actually reject.
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
        "project_id": "test_proj",
        "service_name": "test_service",
        "endpoint": "http://localhost/v1/traces",
        "application_id": "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
    }
    kwargs.update(overrides)
    return DisseqtAgenticClient(**kwargs)


class TestApplicationIdRequired:
    def test_missing_application_id_raises_type_error(self):
        """Not passing application_id at all is a Python-level TypeError."""
        with pytest.raises(TypeError, match="application_id"):
            DisseqtAgenticClient(
                api_key="k",
                project_id="p",
                service_name="s",
                endpoint="http://localhost/v1/traces",
            )

    def test_none_application_id_raises_value_error(self):
        with pytest.raises(ValueError, match="application_id is required"):
            _make_client(application_id=None)

    def test_empty_application_id_raises_value_error(self):
        with pytest.raises(ValueError, match="application_id is required"):
            _make_client(application_id="")

    def test_whitespace_only_application_id_raises_value_error(self):
        with pytest.raises(ValueError, match="application_id is required"):
            _make_client(application_id="   ")

    def test_application_id_must_be_keyword_only(self):
        """
        ``application_id`` sits after a ``*`` in the signature so a
        caller can't accidentally position-pass some other value into
        it. This test locks that in.
        """
        with pytest.raises(TypeError, match="positional"):
            # Passing 11 positional args tries to fill application_id
            # positionally — the kwonly barrier should refuse.
            DisseqtAgenticClient(
                "k",  # api_key
                "p",  # project_id
                "s",  # service_name
                "http://x/v1",  # endpoint
                "1.0.0",  # service_version
                "prod",  # environment
                100,  # max_batch_size
                1.0,  # flush_interval
                3,  # max_retries
                None,  # realtime_policy_id
                "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",  # would land on kwonly application_id
            )

    def test_valid_application_id_constructs(self):
        client = _make_client(application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558")
        assert client.application_id == "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558"

    def test_application_id_is_trimmed(self):
        client = _make_client(application_id="  app-id  ")
        assert client.application_id == "app-id"


class TestApplicationIdValidation:
    """
    TP-2128 round-2 P2 #2.3: a malformed application_id (embedded line
    breaks, non-Latin-1 characters) breaks sending it as an HTTP header,
    so every send fails and never hits the CRITICAL auth-failure path.

    Round-3 senior review P1 #1.1: the original check was a C0-control
    character blacklist that didn't match what ``requests`` /
    ``http.client`` actually reject — rejected harmless tab while
    letting through non-Latin-1 (real crash risk). Tightened to check
    the two real failure modes: embedded ``\\r``/``\\n`` (requests'
    ``InvalidHeader``) and non-Latin-1 characters
    (``http.client.putheader``'s ``.encode("latin-1")``).
    """

    def test_newline_raises(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(application_id="app-id-with-\nnewline")

    def test_carriage_return_raises(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(application_id="app\r\nX-Injected: 1")

    def test_non_latin1_character_raises(self):
        with pytest.raises(ValueError, match="Latin-1"):
            _make_client(application_id="app-\U0001f525-id")

    def test_tab_accepted(self):
        # `requests` does NOT reject tab over the wire — the old blacklist
        # was wrong to fail here.
        _make_client(application_id="app\tid")

    def test_valid_uuid_accepted(self):
        _make_client(application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558")
