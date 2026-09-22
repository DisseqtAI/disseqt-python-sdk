"""
Real (unmocked) background-thread reproduction of the flush-thread-death
bug.

A ``ValueError`` assertion at construction time (see
test_client_header_value_validation.py) does NOT prove the background
``TraceBufferFlushThread`` survives an unexpected exception -- it only
proves a bad value never gets that far in the one path validation
covers. This test bypasses client-level validation entirely (it never
constructs a ``DisseqtAgenticClient``) and drives the REAL
``TraceBuffer`` + its REAL background thread directly, forcing
``transport.send_spans_with_failures`` -- the exact call
``buffer.py:_flush_locked`` makes -- to raise on its first invocation,
exactly the shape an uncaught ``UnicodeEncodeError`` from
``http.client.putheader`` would take (not a
``requests.exceptions.RequestException``, so the transport's own
except-block wouldn't have caught it either).

Before the fix: ``flush_worker`` (buffer.py) called ``self.flush()``
with no try/except. The raised exception propagates out of the
thread's target function, and Python kills the thread silently (a
traceback to stderr via the default thread excepthook, nothing else --
no CRITICAL log, no retry, no restart). All FUTURE automatic flushes
stop forever, for the life of the process.

After the fix: flush_worker catches the exception, logs it, and the
while-loop's next iteration tries again -- the thread survives and a
later flush attempt (once the transport recovers) succeeds normally.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock
from uuid import uuid4

from disseqt_agentic_sdk.buffer.buffer import TraceBuffer
from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.transport.http import HTTPTransport

_FLUSH_INTERVAL = 0.05
_POLL_TIMEOUT = 3.0


def _make_span() -> EnrichedSpan:
    return EnrichedSpan(
        trace_id=str(uuid4()),
        span_id=str(uuid4()),
        name="probe",
        kind="MODEL_EXEC",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        duration_ns=1_000_000_000,
        status_code="OK",
        project_id="proj-resilience-test",
        service_name="probe-service",
        realtime_policy_id="",
    )


def _wait_until(predicate, timeout=_POLL_TIMEOUT, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_flush_thread_survives_an_unexpected_exception_from_send():
    transport = HTTPTransport(endpoint="http://127.0.0.1:1/v1/traces", api_key="k")

    call_count = {"n": 0}

    def flaky_send(spans):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # The exact shape of the bug: a raw exception that is NOT a
            # requests.exceptions.RequestException, exactly what
            # http.client.putheader raises for a non-Latin-1 header
            # value, and exactly what _send_group's
            # `except requests.exceptions.RequestException` would not
            # catch.
            raise UnicodeEncodeError("latin-1", "bad-value", 0, 1, "ordinal not in range(256)")
        return []  # no failures -- second call "succeeds"

    transport.send_spans_with_failures = MagicMock(side_effect=flaky_send)

    buffer = TraceBuffer(transport=transport, max_batch_size=1000, flush_interval=_FLUSH_INTERVAL)
    try:
        buffer.add_span(_make_span())

        # Wait for the background thread to attempt (and fail) its first
        # flush.
        assert _wait_until(
            lambda: call_count["n"] >= 1
        ), "background thread never attempted a flush"

        # This is the actual bug reproduction: pre-fix, the thread is
        # already dead here.
        assert buffer._flush_thread.is_alive(), (
            "TraceBufferFlushThread died after an unexpected exception from "
            "send_spans_with_failures -- this is the exact bug: an uncaught "
            "exception (not a requests.exceptions.RequestException) escapes "
            "flush_worker's bare self.flush() call and kills the daemon "
            "thread for the life of the process."
        )

        # Prove genuine recovery, not just "didn't crash yet": a second
        # span should still get picked up and actually delivered on a
        # later cycle.
        buffer.add_span(_make_span())
        assert _wait_until(lambda: call_count["n"] >= 2), (
            "flush thread did not attempt a second flush -- it stopped "
            "iterating even though it's technically still alive"
        )
        assert buffer._flush_thread.is_alive()
    finally:
        buffer.stop()
