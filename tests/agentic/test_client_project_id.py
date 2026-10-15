"""
Tests for the optional ``project_id`` contract on ``DisseqtAgenticClient``.

Pre-#31 ``project_id`` was a required constructor argument sent on every
trace POST. Current Kong (traces-auth plugin 2.2.1+) resolves the owning
project from ``api_key`` server-side, so the kwarg is now optional:
callers running against new Kong can omit it; callers still running
against older plugin versions pass it and it flows through to the
``project.id`` resource attribute + ``X-Project-Id`` header exactly as
before.

Locks in three behaviours of the "optional" path:
  1. a non-empty ``project_id`` reaches ``HTTPTransport`` so the
     transport-side stamping path has something to stamp;
  2. blank / whitespace-only values normalise to ``None`` — "nothing
     sent" is a single state regardless of how the caller spelled it;
  3. a value containing characters that would break HTTP header
     encoding at send time (embedded CRLF, non-Latin-1) raises
     ``ValueError`` at construction, not silently at every flush
     forever.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient


@pytest.fixture
def _captured_transport(monkeypatch):
    """Stub HTTPTransport + TraceBuffer, return the HTTPTransport mock so
    tests can assert on the ctor kwargs the client passed through."""
    transport_mock = MagicMock()
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", transport_mock)
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.TraceBuffer", MagicMock())
    return transport_mock


def _make_client(**overrides):
    kwargs = {
        "api_key": "test_key",
        "service_name": "test_service",
        "endpoint": "http://localhost/v1/traces",
        "application_id": "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
    }
    kwargs.update(overrides)
    return DisseqtAgenticClient(**kwargs)


class TestProjectIdPassThrough:
    def test_project_id_kwarg_reaches_transport(self, _captured_transport):
        """
        A caller-supplied ``project_id`` must land on ``HTTPTransport``
        — that's the layer that actually stamps it into the OTLP body
        and the ``X-Project-Id`` header. If it stops here, the whole
        "optional + still works against old Kong" guarantee silently
        regresses to "accepted and ignored".
        """
        client = _make_client(project_id="proj-abc")

        assert client.project_id == "proj-abc"
        # HTTPTransport is called once at construction; project_id is
        # one of its kwargs.
        assert _captured_transport.call_args.kwargs["project_id"] == "proj-abc"

    def test_project_id_omitted_defaults_to_none_and_transport_sees_none(self, _captured_transport):
        """
        Omitting the kwarg entirely is the common new-Kong path. The
        stored attribute must be ``None`` (not ``""``) and the transport
        must receive ``None`` so its own "if self.project_id" guards
        skip both the body stamp and the header.
        """
        client = _make_client()

        assert client.project_id is None
        assert _captured_transport.call_args.kwargs["project_id"] is None


class TestProjectIdBlankNormalisation:
    @pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\t\t "])
    def test_blank_or_whitespace_project_id_normalises_to_none(self, blank, _captured_transport):
        """
        A caller who passed ``project_id=""`` or ``project_id="  "``
        meant "no project" every time. Treating these as valid and
        sending them would produce an empty ``X-Project-Id`` header
        and an empty-string ``project.id`` resource attribute — both
        land in Kong / backend logs as a distinct bogus identity.
        Normalise them to ``None`` so "nothing sent" is a single
        state.
        """
        client = _make_client(project_id=blank)

        assert client.project_id is None
        assert _captured_transport.call_args.kwargs["project_id"] is None


class TestProjectIdHeaderValueValidation:
    def test_newline_raises_at_construction(self):
        """
        A ``project_id`` containing an embedded ``\\n`` would raise
        ``requests.InvalidHeader`` at every send — infinitely, since
        the retain-on-failure buffer re-queues the batch. Fail loudly
        at construction instead so the operator sees it the moment
        they build the client.
        """
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(project_id="proj-\nabc")

    def test_carriage_return_raises_at_construction(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(project_id="proj-\rabc")

    def test_non_latin1_raises_at_construction(self):
        """
        ``http.client.putheader`` encodes header values as latin-1 and
        raises on anything outside that range. Match the character
        class the actual sender rejects, not an arbitrary ASCII-only
        rule.
        """
        with pytest.raises(ValueError, match="latin-1|non-ASCII|header"):
            _make_client(project_id="proj-☃")  # snowman: non-latin-1
