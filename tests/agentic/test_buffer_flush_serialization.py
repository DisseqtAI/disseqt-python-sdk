"""
Regression test for a concurrency gap introduced by the P2 fix itself
(found in independent review, not part of the original brief).

Before P2, TraceBuffer.flush() held `self.lock` for its ENTIRE body,
including the network send -- bad for add_span callers (that's what P2
fixed), but it had the side effect of serializing any two concurrent
flush() calls on the same buffer: one call's send always fully finished
before another's began.

After P2 released the lock during the network call, that serialization
guarantee silently disappeared. In practice this bites in
TraceBuffer.stop(): it joins the background flush thread with a 2s
timeout, then unconditionally calls self.flush() itself regardless of
whether the background thread is still mid-send. If a size-triggered
flush's send is still in flight past that 2s window (slow network, large
batch) and new spans arrived in the meantime, stop()'s own flush() call
now races the background one -- two concurrent outbound POSTs to the
same endpoint with no ordering guarantee between them. Each still gets a
correct, disjoint batch (no duplication or loss), but out-of-order
delivery of two batches to an ingest endpoint is still a real defect for
an SDK that documents flush() as give-me-a-clean-send-boundary semantics.

The fix: a dedicated `_send_lock` that flush() holds for its whole body
(extraction through the network call to failure-merge), serializing
concurrent sends back to the pre-P2 guarantee -- while add_span/add_spans
continue to never touch it, so they still never block on I/O.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock
from uuid import uuid4

from disseqt_agentic_sdk.buffer.buffer import TraceBuffer
from disseqt_agentic_sdk.models.span import EnrichedSpan

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
        project_id="proj-serialization-test",
        service_name="probe-service",
        realtime_policy_id="",
    )


def _wait_until(predicate, timeout=_POLL_TIMEOUT, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestFlushSerializesConcurrentSends:
    def test_two_concurrent_flush_calls_do_not_send_at_the_same_time(self):
        events: list[tuple[str, int, float]] = []
        events_lock = threading.Lock()
        call_count = {"n": 0}
        count_lock = threading.Lock()
        release_a = threading.Event()
        arrived_a = threading.Event()

        transport = MagicMock()

        def slow_send(spans):
            with count_lock:
                call_count["n"] += 1
                n = call_count["n"]
            with events_lock:
                events.append(("start", n, time.monotonic()))
            if n == 1:
                arrived_a.set()
                release_a.wait(timeout=_POLL_TIMEOUT)
            with events_lock:
                events.append(("end", n, time.monotonic()))
            return []

        transport.send_spans_with_failures.side_effect = slow_send

        buffer = TraceBuffer(transport=transport, max_batch_size=1000, flush_interval=60.0)
        try:
            with buffer.lock:
                buffer.buffer.append(_make_span())
            thread_a = threading.Thread(target=buffer.flush)
            thread_a.start()
            assert _wait_until(arrived_a.is_set), "first flush() never reached the transport"

            # A second batch arrives (e.g. stop()'s final flush racing the
            # background thread's still-in-flight size-triggered one)
            # while the first send is still blocked on release_a.
            with buffer.lock:
                buffer.buffer.append(_make_span())
            thread_b = threading.Thread(target=buffer.flush)
            thread_b.start()

            # Give thread_b every chance to race ahead if the bug is present.
            time.sleep(0.2)
            with count_lock:
                calls_before_release = call_count["n"]
            assert calls_before_release == 1, (
                "a second flush() call reached the transport while the "
                "first send was still in flight -- concurrent sends have "
                "no ordering guarantee"
            )

            release_a.set()
            thread_a.join(timeout=_POLL_TIMEOUT)
            thread_b.join(timeout=_POLL_TIMEOUT)

            with events_lock:
                starts = {n: t for kind, n, t in events if kind == "start"}
                ends = {n: t for kind, n, t in events if kind == "end"}
            assert 1 in ends and 2 in starts, f"expected both calls to complete: {events}"
            assert ends[1] <= starts[2], "second send started before the first one finished"
        finally:
            release_a.set()
            buffer.stop()

    def test_add_span_still_never_blocks_while_a_send_is_in_flight(self):
        """The _send_lock fix must not leak into add_span's path."""
        release = threading.Event()
        arrived = threading.Event()
        transport = MagicMock()

        def slow_send(spans):
            arrived.set()
            release.wait(timeout=_POLL_TIMEOUT)
            return []

        transport.send_spans_with_failures.side_effect = slow_send

        buffer = TraceBuffer(transport=transport, max_batch_size=1, flush_interval=60.0)
        try:
            buffer.add_span(_make_span())  # crosses max_batch_size, wakes flush thread
            assert _wait_until(arrived.is_set), "background flush never reached the transport"

            start = time.monotonic()
            buffer.add_span(_make_span())  # must return immediately, not wait on _send_lock
            elapsed = time.monotonic() - start
            assert elapsed < 0.2, f"add_span blocked for {elapsed:.3f}s behind an in-flight send"
        finally:
            release.set()
            buffer.stop()
