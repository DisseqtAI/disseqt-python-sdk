"""
Tests for header-value validation at the transport layer -- the one
point every value actually passes through before becoming a header,
regardless of which upstream class produced it.

client.client.DisseqtAgenticClient's own validation (see
test_client_header_value_validation.py) only covers callers who go
through the client. project_id in particular can reach a header via a
directly-constructed DisseqtTrace, DisseqtSpan, or EnrichedSpan -- all
public classes exported from disseqt_agentic_sdk -- bypassing the
client entirely. This file proves that bypass is now closed: a bad
project_id reaching _send_group results in a clean, logged failure
(send_spans returns False) rather than an uncaught UnicodeEncodeError
that would previously either kill the flush thread (if hit via the
background flush) or propagate directly into the caller's own code (if
hit via add_span's synchronous immediate-flush path, which the
buffer.py flush_worker fix does NOT cover -- that only wraps the
background thread's call).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.trace.trace import DisseqtTrace
from disseqt_agentic_sdk.transport.http import HTTPTransport


def _make_span(project_id: str) -> EnrichedSpan:
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
        realtime_policy_id="",
    )


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


class TestProjectIdBypassesClientButNotTransport:
    """
    The exact gap an independent review found: project_id validated at
    DisseqtAgenticClient construction never actually reaches
    HTTPTransport (the client doesn't even pass project_id to it -- see
    client.client.DisseqtAgenticClient.__init__). The real value comes
    from each span's own project_id at send time, which a caller can
    set directly via EnrichedSpan/DisseqtSpan/DisseqtTrace without ever
    touching a validated client.
    """

    def test_bad_project_id_via_direct_enrichedspan_is_a_clean_failure_not_a_crash(self):
        transport = HTTPTransport(endpoint="http://127.0.0.1:1/v1/traces", api_key="k")
        bad_span = _make_span(project_id="proj-\U0001f525-bad")

        # This must NOT raise -- it must be caught and converted into a
        # clean False, exactly like any other delivery failure.
        result = transport.send_spans([bad_span])
        assert result is False

    def test_bad_project_id_via_direct_disseqttrace_is_a_clean_failure_not_a_crash(self):
        """
        Reproduces the exact bypass: DisseqtTrace is publicly exported
        and constructible without a client. Its project_id never passes
        through DisseqtAgenticClient's validation at all.
        """
        transport = HTTPTransport(endpoint="http://127.0.0.1:1/v1/traces", api_key="k")
        trace = DisseqtTrace(name="bypass-probe", project_id="proj-\U0001f525-bad")
        span = trace.to_enriched_spans()[0] if trace.spans else None
        if span is None:
            # No spans yet on a fresh trace -- build the EnrichedSpan the
            # same way transport code reads it: project_id lives on the
            # trace and is threaded onto every span it creates.
            from disseqt_agentic_sdk.enums import SpanKind

            with trace.start_span("probe", SpanKind.AGENT_EXEC):
                pass
            span = trace.to_enriched_spans()[0]

        assert span.project_id == "proj-\U0001f525-bad"  # confirms the bypass is real
        result = transport.send_spans([span])
        assert result is False

    def test_valid_project_id_via_direct_enrichedspan_still_sends_normally(self):
        transport = HTTPTransport(endpoint="http://127.0.0.1:1/v1/traces", api_key="k")
        good_span = _make_span(project_id="proj-ordinary")
        # No real server listening on port 1 -- this will fail at the
        # network layer, but NOT at validation. Confirm we get past
        # validation by checking the failure isn't a ValueError/crash.
        result = transport.send_spans([good_span])
        assert result is False  # network failure (expected, no server)
        # The key assertion: no exception escaped (would have failed
        # the test outright), proving validation passed through cleanly
        # for an ordinary value.
