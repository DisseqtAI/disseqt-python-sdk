"""
Tests for the header-first identity contract on ``HTTPTransport``.

Historically the SDK stamped ``api.key``, ``project.id``, and
``policy.id`` only inside the OTLP body's ``resource.attributes``. That
forced Kong's ``traces-auth`` plugin to buffer + JSON-parse the entire
payload just to decide auth — the read path that landed next to the
8 KB spool-to-disk bug in TP-2314.

The transport now stamps the same three values as HTTP headers on every
outgoing POST (``X-Api-Key``, ``X-Project-Id``, ``X-Realtime-Policy-Id``)
so a header-aware Kong plugin can authenticate before touching any body.
``resource.attributes`` are still populated so this SDK version keeps
working against Kong plugin versions that only look in the body — the
migration is safe old-server / new-client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.transport.http import HTTPTransport


def _make_span(project_id: str, realtime_policy_id: str = "") -> EnrichedSpan:
    return EnrichedSpan(
        trace_id=str(uuid4()),
        span_id=str(uuid4()),
        name="probe",
        kind="MODEL_EXEC",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        duration_ns=1_000_000_000,
        status_code="OK",
        project_id=project_id,
        service_name="probe-service",
        realtime_policy_id=realtime_policy_id,
    )


def _post_call(transport: HTTPTransport, response_status: int = 200):
    """Fire one send, return the captured ``session.post`` call args."""
    fake_response = MagicMock(status_code=response_status)
    fake_response.raise_for_status = MagicMock()
    with patch.object(transport.session, "post", return_value=fake_response) as post:
        transport.send_spans([_make_span("proj-123")])
        assert post.called, "transport did not POST to the endpoint"
        return post.call_args


class TestHeaderFirstIdentity:
    def test_api_key_and_project_id_are_sent_as_headers(self):
        """
        A header-aware Kong plugin authenticates off headers before ever
        reading the body. Both fields must therefore be on the request
        line, not only in the payload — that's the whole point of the
        migration path documented in the transport module.
        """
        transport = HTTPTransport(
            endpoint="https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
            api_key="secret-key-42",
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
        )
        _, kwargs = _post_call(transport)

        headers = kwargs["headers"]
        assert headers["X-Api-Key"] == "secret-key-42"
        assert headers["X-Project-Id"] == "proj-123"
        # X-Application-Id must survive the header migration — every trace
        # POST already relied on it for policy-management verification and
        # a regression here would silently 401 every SDK.
        assert headers["X-Application-Id"] == "7ce57144-9df6-4fa4-8aad-8cbc1ffdb558"

    def test_realtime_policy_id_is_sent_when_the_client_default_is_set(self):
        """
        Client-level ``realtime_policy_id`` becomes the fallback for every
        span that didn't override it. The header must reflect the same
        value the body will carry — a header-aware plugin routes on it.
        """
        transport = HTTPTransport(
            endpoint="https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
            api_key="secret-key-42",
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
            realtime_policy_id="pol-default",
        )
        _, kwargs = _post_call(transport)

        assert kwargs["headers"]["X-Realtime-Policy-Id"] == "pol-default"

    def test_realtime_policy_id_is_not_sent_when_unset(self):
        """
        Non-policy callers get bit-for-bit the same request shape as
        before — the header is only emitted when there is a policy to
        stamp. This mirrors the resource.attributes["policy.id"] contract
        added earlier and keeps the ``if not header, do the old thing``
        branch on Kong reachable.
        """
        transport = HTTPTransport(
            endpoint="https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
            api_key="secret-key-42",
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
        )
        _, kwargs = _post_call(transport)

        assert "X-Realtime-Policy-Id" not in kwargs["headers"]

    def test_resource_attributes_still_carry_identity_for_backward_compat(self):
        """
        Backward compatibility with plugin versions that only read the
        body — a new SDK must keep working against an old Kong. This
        assertion locks in that we haven't accidentally *replaced* the
        body-side identity, only *added* the header-side copy.
        """
        transport = HTTPTransport(
            endpoint="https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
            api_key="secret-key-42",
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
            realtime_policy_id="pol-default",
        )
        _, kwargs = _post_call(transport)

        attrs = kwargs["json"]["resource"]["attributes"]
        assert attrs["api.key"] == "secret-key-42"
        assert attrs["project.id"] == "proj-123"
        assert attrs["policy.id"] == "pol-default"

    def test_api_key_header_omitted_when_client_has_no_key(self):
        """
        Empty api_key must not turn into ``X-Api-Key: `` on the wire —
        Kong header handling and any downstream rate-limit-by-key
        keying would treat an empty string as a distinct identity.
        """
        transport = HTTPTransport(
            endpoint="https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
            api_key=None,
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
        )
        _, kwargs = _post_call(transport)

        assert "X-Api-Key" not in kwargs["headers"]
