"""
Additional tests to achieve 100% coverage.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.api.client import set_client
from disseqt_agentic_sdk.api.trace import TraceWrapper
from disseqt_agentic_sdk.buffer import TraceBuffer
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.trace import DisseqtTrace
from disseqt_agentic_sdk.transport import HTTPTransport
from disseqt_agentic_sdk.utils.time import (
    from_timestamp_ms,
    from_timestamp_ns,
    to_timestamp_ms,
    to_timestamp_ns,
)


class TestTraceCoverage:
    """Tests to cover missing trace.py lines."""

    def test_trace_set_intent_id(self):
        """Test set_intent_id method."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        span = trace.start_span("span1", SpanKind.INTERNAL)

        trace.set_intent_id("intent_123")
        assert trace.intent_id == "intent_123"
        assert span.attributes["agentic.intent.id"] == "intent_123"

    def test_trace_set_workflow_id(self):
        """Test set_workflow_id method."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        span = trace.start_span("span1", SpanKind.INTERNAL)

        trace.set_workflow_id("workflow_456")
        assert trace.workflow_id == "workflow_456"
        assert span.attributes["agentic.workflow.id"] == "workflow_456"

    def test_trace_end_already_ended(self):
        """Test end() when trace is already ended."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        trace.end()
        end_time_1 = trace.end_time_ns

        # Call end() again - should return early
        trace.end()
        assert trace.end_time_ns == end_time_1  # Should not change

    def test_trace_get_spans(self):
        """Test get_spans() method."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        span1 = trace.start_span("span1", SpanKind.INTERNAL)
        span2 = trace.start_span("span2", SpanKind.INTERNAL)

        spans = trace.get_spans()
        assert len(spans) == 2
        assert spans[0] is span1
        assert spans[1] is span2
        # Should be a copy, not the same list
        assert spans is not trace.spans

    def test_trace_exit(self):
        """Test trace __exit__ method."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        span = trace.start_span("span1", SpanKind.INTERNAL)

        # Simulate context manager exit
        trace.__exit__(None, None, None)
        assert trace.is_ended is True
        assert span.end_time_ns is not None

    def test_trace_to_dict(self):
        """Test to_dict method."""
        trace = DisseqtTrace(
            name="test_trace",
            trace_id="custom_id",
            org_id="o",
            project_id="p",
            service_name="s",
            service_version="1.0",
            environment="dev",
            intent_id="i1",
            workflow_id="w1",
        )
        trace.start_span("span1", SpanKind.INTERNAL)
        trace.end()

        trace_dict = trace.to_dict()
        assert trace_dict["trace_id"] == "custom_id"
        assert trace_dict["name"] == "test_trace"
        assert trace_dict["org_id"] == "o"
        assert trace_dict["project_id"] == "p"
        assert trace_dict["service_name"] == "s"
        assert trace_dict["service_version"] == "1.0"
        assert trace_dict["environment"] == "dev"
        assert trace_dict["intent_id"] == "i1"
        assert trace_dict["workflow_id"] == "w1"
        assert trace_dict["span_count"] == 1
        assert trace_dict["start_time_ns"] > 0
        assert trace_dict["end_time_ns"] is not None


class TestTraceWrapperCoverage:
    """Tests to cover missing trace.py wrapper lines."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup fixture with mocked dependencies."""
        with (
            patch("disseqt_agentic_sdk.client.client.HTTPTransport"),
            patch("disseqt_agentic_sdk.client.client.TraceBuffer"),
        ):
            self.client = DisseqtAgenticClient(api_key="k", project_id="p", service_name="s")
            set_client(self.client)
            yield
            try:
                self.client.shutdown()
            except RuntimeError:
                pass  # Client already cleared

    def test_trace_wrapper_getattr(self):
        """Test TraceWrapper __getattr__ delegation."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        wrapper = TraceWrapper(trace, self.client)

        # Access trace attributes through wrapper using __getattr__
        assert wrapper.name == "test"
        assert wrapper.trace_id is not None
        assert wrapper.org_id == "o"
        assert wrapper.project_id == "p"

    def test_trace_wrapper_enter(self):
        """Test TraceWrapper __enter__ method."""
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        wrapper = TraceWrapper(trace, self.client)

        # Test __enter__ returns the trace itself
        result = wrapper.__enter__()
        assert result is trace
        wrapper.__exit__(None, None, None)

    def test_trace_wrapper_no_client_exit(self):
        """Test TraceWrapper when client is None in __exit__."""
        # Create wrapper with valid client first
        trace = DisseqtTrace(name="test", org_id="o", project_id="p", service_name="s")
        wrapper = TraceWrapper(trace, self.client)

        # Manually set client to None to test the warning path
        wrapper._client = None

        # Should not raise error, just log warning
        wrapper.__exit__(None, None, None)


class TestBufferCoverage:
    """Tests to cover missing buffer.py lines."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.transport = Mock(spec=HTTPTransport)
        # Default the retention-critical method to a stable "no failures"
        # return so the background flush thread doesn't trip the
        # `Mock + list` addition inside _flush_locked between tests.
        self.transport.send_spans_with_failures.return_value = []
        self.buffer = TraceBuffer(self.transport, max_batch_size=3, flush_interval=0.1)
        yield
        # Stop the flush thread so no daemon flush fires after the
        # transport mock is torn down.
        self.buffer._stop_flush_thread = True
        if self.buffer._flush_thread:
            self.buffer._flush_thread.join(timeout=0.5)

    def test_buffer_add_span(self):
        """Test add_span method."""
        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )

        self.buffer.add_span(span)
        assert len(self.buffer.buffer) == 1

    def test_buffer_add_span_flush_on_batch_size(self):
        """Test add_span triggers flush when batch size reached."""
        # Buffer now calls send_spans_with_failures (P1 #1.2). Empty
        # list = full success.
        self.transport.send_spans_with_failures.return_value = []

        # Add spans up to batch size
        for i in range(3):
            span = EnrichedSpan(
                trace_id=f"t{i}",
                span_id=f"s{i}",
                name="test",
                org_id="o",
                project_id="p",
                service_name="s",
            )
            self.buffer.add_span(span)

        # Should have flushed
        self.transport.send_spans_with_failures.assert_called_once()
        assert len(self.buffer.buffer) == 0

    def test_buffer_add_spans_flush_on_batch_size(self):
        """Test add_spans triggers flush when batch size reached."""
        self.transport.send_spans_with_failures.return_value = []

        spans = [
            EnrichedSpan(
                trace_id=f"t{i}",
                span_id=f"s{i}",
                name="test",
                org_id="o",
                project_id="p",
                service_name="s",
            )
            for i in range(3)
        ]

        self.buffer.add_spans(spans)
        self.transport.send_spans_with_failures.assert_called_once()

    def test_buffer_should_flush(self):
        """Test should_flush method."""
        # Stop the background flush thread to test manually
        self.buffer._stop_flush_thread = True
        if self.buffer._flush_thread:
            self.buffer._flush_thread.join(timeout=0.5)

        # Empty buffer should return False
        assert self.buffer.should_flush() is False

        # Add span
        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )
        # Reset last_flush_time to ensure proper timing
        import time

        self.buffer.last_flush_time = time.time()
        self.buffer.add_span(span)

        # Should return True after flush interval
        time.sleep(0.15)  # Wait longer than flush_interval (0.1s)
        assert self.buffer.should_flush() is True


class TestBufferFailurePolicy:
    """
    TP-2128 P2 #2.4: buffer used to clear before send_spans and discard the
    return value — a 401/403 or 5xx dropped every batch silently. Now the
    buffer only clears on success, retries on failure up to
    max_retained_spans, and drops the oldest first once the cap is hit.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.transport = Mock(spec=HTTPTransport)
        # Small cap keeps the overflow test fast.
        self.buffer = TraceBuffer(
            self.transport,
            max_batch_size=3,
            flush_interval=0.1,
            max_retained_spans=5,
        )
        self.buffer._stop_flush_thread = True
        if self.buffer._flush_thread:
            self.buffer._flush_thread.join(timeout=0.5)
        yield

    def _mk(self, i):
        return EnrichedSpan(
            trace_id=f"t{i}",
            span_id=f"s{i}",
            name="test",
            org_id="o",
            project_id="p",
            service_name="s",
        )

    def test_failed_send_retains_spans_for_retry(self):
        # All spans fail → all retained.
        spans = [self._mk(0), self._mk(1)]
        self.transport.send_spans_with_failures.return_value = spans
        self.buffer.buffer.extend(spans)
        self.buffer.flush()

        assert (
            len(self.buffer.buffer) == 2
        ), f"failed send must retain spans, got {len(self.buffer.buffer)}"
        self.transport.send_spans_with_failures.assert_called_once()

    def test_successful_send_clears_only_sent_spans(self):
        # Empty failed list → buffer emptied.
        self.transport.send_spans_with_failures.return_value = []
        self.buffer.buffer.extend([self._mk(0), self._mk(1)])
        self.buffer.flush()
        assert self.buffer.buffer == []

    def test_partial_multi_group_failure_retains_only_failed_group(self):
        """
        TP-2128 round-2 P1 #1.2: two groups in one batch — one
        succeeds, one fails — used to retain the whole batch and
        re-POST the succeeded group on the next flush. Now the buffer
        retains only the failed spans, so the succeeded group is
        never re-sent.
        """
        succeeded = self._mk(0)  # group A, will succeed
        failed = self._mk(1)  # group B, will fail
        self.buffer.buffer.extend([succeeded, failed])
        # Transport reports only the failed one.
        self.transport.send_spans_with_failures.return_value = [failed]
        self.buffer.flush()

        # Only the failed span survives — the succeeded one is dropped
        # from the buffer so the next flush doesn't re-POST it.
        assert [s.span_id for s in self.buffer.buffer] == [
            "s1"
        ], f"expected only failed s1 retained; got {[s.span_id for s in self.buffer.buffer]}"

        # Simulate a next flush where the retry succeeds — the failed
        # span must go out again and clear.
        self.transport.send_spans_with_failures.reset_mock()
        self.transport.send_spans_with_failures.return_value = []
        self.buffer.flush()
        # Second call sent exactly the retained span, not both.
        call_args, _ = self.transport.send_spans_with_failures.call_args
        sent_on_retry = call_args[0]
        assert [s.span_id for s in sent_on_retry] == [
            "s1"
        ], "succeeded span must NOT be re-POSTed on retry flush"
        assert self.buffer.buffer == []

    def test_buffer_caps_at_max_retained_on_repeated_failure(self):
        # Every span in every flush fails.
        def _echo(spans):
            return list(spans)

        self.transport.send_spans_with_failures.side_effect = _echo
        # Feed 8 spans across two failing flushes; cap is 5.
        for i in range(4):
            self.buffer.buffer.append(self._mk(i))
        self.buffer.flush()
        assert len(self.buffer.buffer) == 4  # under cap, all retained

        for i in range(4, 8):
            self.buffer.buffer.append(self._mk(i))
        self.buffer.flush()
        assert (
            len(self.buffer.buffer) == 5
        ), f"buffer must cap at max_retained_spans=5, got {len(self.buffer.buffer)}"
        # Oldest dropped first: span_ids should be the newest 5 (s3..s7).
        remaining_ids = [s.span_id for s in self.buffer.buffer]
        assert remaining_ids == ["s3", "s4", "s5", "s6", "s7"]


class TestTransportAuthFailures:
    """TP-2128 P2 #2.4: 401/403 must log at CRITICAL and set status_code
    in extras so ops can route on it."""

    def test_401_logs_critical(self, caplog):
        import logging

        import requests as _requests

        transport = HTTPTransport("http://localhost:8080/v1/traces", api_key="k")
        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )
        with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
            fake = Mock()
            fake.status_code = 401
            fake.raise_for_status.side_effect = _requests.exceptions.HTTPError(
                "401 Unauthorized", response=fake
            )
            mock_post.return_value = fake

            with caplog.at_level(logging.CRITICAL, logger="disseqt_agentic_sdk"):
                # Bump the transport logger too — it uses the same tree
                # but may be silenced by disseqt_logging by default.
                from disseqt_agentic_sdk.transport import http as http_mod

                prior_level = http_mod.logger.level
                http_mod.logger.setLevel(logging.CRITICAL)
                try:
                    ok = transport.send_spans([span])
                finally:
                    http_mod.logger.setLevel(prior_level)

        assert ok is False
        critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert critical, "401 must log at CRITICAL"
        assert any("auth rejected" in r.getMessage() for r in critical)
        assert any(getattr(r, "status_code", None) == 401 for r in critical)

    def test_401_writes_stderr_without_logging_configured(self, capsys):
        """
        TP-2128 round-2 P1 #1.3: the disseqt_logging silent-by-default
        gate suppresses the CRITICAL log line in fresh / unconfigured
        processes, so operators saw NOTHING when their credentials
        broke. Auth failures now write directly to stderr as well,
        bypassing the logging stack.

        Test does NOT bump any logger level — matches the shape of a
        real fresh install.
        """
        import requests as _requests

        transport = HTTPTransport("http://prod.example/v1/traces", api_key="k")
        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )
        with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
            fake = Mock()
            fake.status_code = 403
            fake.raise_for_status.side_effect = _requests.exceptions.HTTPError(
                "403 Forbidden", response=fake
            )
            mock_post.return_value = fake
            ok = transport.send_spans([span])

        assert ok is False
        captured = capsys.readouterr()
        assert (
            "CRITICAL: ingest auth rejected (403)" in captured.err
        ), f"stderr must carry the auth-failure banner; got: {captured.err!r}"
        assert "prod.example" in captured.err
        assert "DISSEQT_API_KEY" in captured.err

    def test_401_stderr_opt_out(self, capsys, monkeypatch):
        """DISSEQT_SDK_SILENCE_AUTH_STDERR=1 disables the fallback."""
        import importlib

        import requests as _requests

        from disseqt_agentic_sdk.transport import http as http_mod

        monkeypatch.setenv("DISSEQT_SDK_SILENCE_AUTH_STDERR", "1")
        importlib.reload(http_mod)
        try:
            transport = http_mod.HTTPTransport("http://localhost:8080/v1/traces", api_key="k")
            span = EnrichedSpan(
                trace_id="t1",
                span_id="s1",
                name="test",
                org_id="o",
                project_id="p",
                service_name="s",
            )
            with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
                fake = Mock()
                fake.status_code = 401
                fake.raise_for_status.side_effect = _requests.exceptions.HTTPError(
                    "401 Unauthorized", response=fake
                )
                mock_post.return_value = fake
                transport.send_spans([span])
            captured = capsys.readouterr()
            assert (
                "CRITICAL" not in captured.err
            ), f"opt-out env var must silence stderr fallback; got: {captured.err!r}"
        finally:
            importlib.reload(http_mod)


class TestTransportCoverage:
    """Tests to cover missing transport.py lines."""

    def test_transport_empty_spans(self):
        """Test send_spans with empty list."""
        transport = HTTPTransport("http://localhost:8080/v1/traces")
        result = transport.send_spans([])
        assert result is True

    def test_transport_gen_ai_attributes(self):
        """Test handling gen_ai attributes."""
        transport = HTTPTransport("http://localhost:8080/v1/traces", api_key="test-api-key")

        span = EnrichedSpan(
            trace_id="t1",
            span_id="s1",
            name="test",
            org_id="o",
            project_id="p",
            service_name="s",
            attributes_json='{"gen_ai.test": "value", "agentic.test": "value2"}',
        )

        with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            transport.send_spans([span])

            # Check that attributes are included in payload (no genAi/agentic separation)
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            spans = payload["traces"][0]["spans"]
            assert "attributes" in spans[0]
            assert spans[0]["attributes"]["gen_ai.test"] == "value"
            assert spans[0]["attributes"]["agentic.test"] == "value2"

            # Check that API key is in resource attributes
            assert "api.key" in payload["resource"]["attributes"]
            assert payload["resource"]["attributes"]["api.key"] == "test-api-key"

    def test_transport_error_handling(self):
        """Test error handling in send_spans."""
        transport = HTTPTransport("http://localhost:8080/v1/traces")

        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )

        with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
            from requests.exceptions import RequestException

            mock_post.side_effect = RequestException("Network error")

            result = transport.send_spans([span])
            assert result is False

    def test_transport_send_trace(self):
        """Test send_trace alias method."""
        transport = HTTPTransport("http://localhost:8080/v1/traces")

        span = EnrichedSpan(
            trace_id="t1", span_id="s1", name="test", org_id="o", project_id="p", service_name="s"
        )

        with patch("disseqt_agentic_sdk.transport.http.requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            result = transport.send_trace([span])
            assert result is True
            mock_post.assert_called_once()


class TestUtilsCoverage:
    """Tests to cover missing utils/time.py lines."""

    def test_to_timestamp_ns_naive_datetime(self):
        """Test to_timestamp_ns with naive datetime."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        ns = to_timestamp_ns(naive_dt)
        assert isinstance(ns, int)
        assert ns > 0

    def test_from_timestamp_ns_naive_datetime(self):
        """Test from_timestamp_ns returns timezone-aware datetime."""
        ns = 1704110400000000000  # 2024-01-01 12:00:00 UTC
        dt = from_timestamp_ns(ns)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_to_timestamp_ms_naive_datetime(self):
        """Test to_timestamp_ms with naive datetime."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        ms = to_timestamp_ms(naive_dt)
        assert isinstance(ms, int)
        assert ms > 0

    def test_from_timestamp_ms_naive_datetime(self):
        """Test from_timestamp_ms returns timezone-aware datetime."""
        ms = 1704110400000  # 2024-01-01 12:00:00 UTC
        dt = from_timestamp_ms(ms)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc


class TestModelsCoverage:
    """Tests to cover missing models/span.py lines."""

    def test_enriched_span_from_dict_datetime_strings(self):
        """Test from_dict with datetime strings."""
        data = {
            "trace_id": "550e8400-e29b-41d4-a716-446655440000",
            "span_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "name": "test",
            "kind": "INTERNAL",
            "org_id": "o",
            "project_id": "p",
            "service_name": "s",
            "dt": "2024-01-01T12:00:00Z",
            "ingestion_time": "2024-01-01T12:00:01Z",
        }

        span = EnrichedSpan.from_dict(data)
        assert isinstance(span.dt, datetime)
        assert isinstance(span.ingestion_time, datetime)

    def test_enriched_span_from_dict_invalid_uuid(self):
        """Test from_dict with invalid UUID strings."""
        data = {
            "trace_id": "not-a-valid-uuid",
            "span_id": "also-not-valid",
            "parent_span_id": "invalid-too",
            "name": "test",
            "kind": "INTERNAL",
            "org_id": "o",
            "project_id": "p",
            "service_name": "s",
        }

        span = EnrichedSpan.from_dict(data)
        assert span.trace_id == "not-a-valid-uuid"
        assert span.span_id == "also-not-valid"
        assert span.parent_span_id == "invalid-too"

    def test_enriched_span_from_dict_empty_parent_span_id(self):
        """Test from_dict with empty parent_span_id."""
        data = {
            "trace_id": "t1",
            "span_id": "s1",
            "parent_span_id": "",
            "name": "test",
            "kind": "INTERNAL",
            "org_id": "o",
            "project_id": "p",
            "service_name": "s",
        }

        span = EnrichedSpan.from_dict(data)
        assert span.parent_span_id is None
