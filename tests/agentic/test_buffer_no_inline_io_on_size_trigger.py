"""
Regression tests for P2: the size-triggered batch flush must not do
network I/O in the caller's own thread, and must never hold
TraceBuffer.lock while that I/O is in flight.

Root cause (before this fix): add_span/add_spans called
self._flush_locked() directly, from inside `with self.lock:`.
_flush_locked() calls transport.send_spans_with_failures(...) -- the
actual HTTP POST -- while still holding the lock, and it runs on
whatever thread called add_span (the application's own thread, e.g. the
one making an LLM call). So the Nth add_span() that happens to cross
max_batch_size blocks the CALLER on a network round trip, and every
other thread calling add_span/add_spans/flush() on the same buffer
blocks too, waiting for that same lock.

flush() itself is untouched and stays synchronous by design -- a caller
who explicitly calls flush() (e.g. on shutdown) is expected to block
until it's done.
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
        project_id="proj-p2-test",
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


def _make_slow_transport(release: threading.Event, arrived: threading.Event):
    """A stub transport whose send call blocks until the test releases it."""
    transport = MagicMock()

    def slow_send(spans):
        arrived.set()  # tell the test the send has actually started
        release.wait(timeout=_POLL_TIMEOUT)  # held open until the test says go
        return []  # no failures

    transport.send_spans_with_failures.side_effect = slow_send
    return transport


class TestSizeTriggerDoesNotBlockCaller:
    def test_add_span_returns_immediately_even_while_the_send_is_still_in_flight(self):
        release = threading.Event()
        arrived = threading.Event()
        transport = _make_slow_transport(release, arrived)
        buffer = TraceBuffer(transport=transport, max_batch_size=2, flush_interval=60.0)
        try:
            buffer.add_span(_make_span())  # 1 of 2 -- no flush yet

            start = time.monotonic()
            buffer.add_span(_make_span())  # 2 of 2 -- crosses max_batch_size
            elapsed = time.monotonic() - start

            # The send is proven to have actually started (on some other
            # thread) and is being held open by `release` -- if add_span
            # had done the send inline, this call could not have returned
            # until `release.set()`, which we haven't called yet.
            assert _wait_until(arrived.is_set), "transport.send was never invoked"
            assert elapsed < 0.2, f"add_span blocked for {elapsed:.3f}s on the in-flight send"
        finally:
            release.set()
            buffer.stop()

    def test_lock_is_free_while_the_size_triggered_send_is_in_flight(self):
        # Must run add_span on its own thread: if it inline-blocks (the
        # bug), calling it synchronously here would mean the call has
        # already returned -- lock released -- by the time we get to
        # check it, which would pass even on the buggy code for the
        # wrong reason. A separate thread lets the main thread probe lock
        # state while that call is genuinely still stuck in the network
        # wait.
        release = threading.Event()
        arrived = threading.Event()
        transport = _make_slow_transport(release, arrived)
        buffer = TraceBuffer(transport=transport, max_batch_size=1, flush_interval=60.0)
        adder = threading.Thread(target=lambda: buffer.add_span(_make_span()))
        try:
            adder.start()
            assert _wait_until(arrived.is_set), "transport.send was never invoked"

            # The send is in flight right now (blocked on `release`). The
            # buffer's own lock must NOT be held during that window, or
            # every other add_span/flush caller would queue up behind a
            # network call.
            got_lock = buffer.lock.acquire(blocking=False)
            try:
                assert got_lock, "buffer.lock was held during in-flight network I/O"
            finally:
                if got_lock:
                    buffer.lock.release()
        finally:
            release.set()
            adder.join(timeout=_POLL_TIMEOUT)
            buffer.stop()

    def test_no_span_is_lost_the_size_triggered_batch_is_still_delivered(self):
        release = threading.Event()
        release.set()  # let it send immediately, no need to hold it open here
        arrived = threading.Event()
        transport = _make_slow_transport(release, arrived)
        buffer = TraceBuffer(transport=transport, max_batch_size=2, flush_interval=60.0)
        try:
            buffer.add_span(_make_span())
            buffer.add_span(_make_span())

            assert _wait_until(lambda: transport.send_spans_with_failures.called)
            assert _wait_until(lambda: len(buffer.buffer) == 0)
            (sent_spans,) = transport.send_spans_with_failures.call_args.args
            assert len(sent_spans) == 2
        finally:
            buffer.stop()


class TestFlushStaysSynchronous:
    def test_explicit_flush_still_blocks_the_caller_until_the_send_completes(self):
        release = threading.Event()
        arrived = threading.Event()
        transport = _make_slow_transport(release, arrived)
        buffer = TraceBuffer(transport=transport, max_batch_size=1000, flush_interval=60.0)

        def release_shortly_after_send_starts():
            arrived.wait(timeout=_POLL_TIMEOUT)
            time.sleep(0.1)
            release.set()

        releaser = threading.Thread(target=release_shortly_after_send_starts)
        releaser.start()
        try:
            with buffer.lock:
                buffer.buffer.append(_make_span())

            start = time.monotonic()
            buffer.flush()  # explicit call -- must not return before the send finishes
            elapsed = time.monotonic() - start

            assert elapsed >= 0.1, "flush() returned before the in-flight send completed"
            transport.send_spans_with_failures.assert_called_once()
        finally:
            releaser.join(timeout=_POLL_TIMEOUT)
            buffer.stop()
