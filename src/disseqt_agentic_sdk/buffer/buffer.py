"""
Buffer for batching spans before sending to backend.
"""

import time
from threading import Event, Lock, Thread

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.transport import HTTPTransport
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger()


class TraceBuffer:
    """
    Buffer for batching spans before sending to backend.

    Supports:
    - Size-based flushing (max batch size)
    - Time-based flushing (interval)
    - Thread-safe operations
    """

    def __init__(
        self,
        transport: HTTPTransport,
        max_batch_size: int = 100,
        flush_interval: float = 1.0,
        max_retained_spans: int | None = None,
    ):
        """
        Initialize buffer.

        Args:
            transport: HTTPTransport instance for sending
            max_batch_size: Maximum number of spans per batch (triggers immediate flush)
            flush_interval: Flush interval in seconds (time-based flushing)
            max_retained_spans: Hard cap on retained spans across failed
                sends. Prevents unbounded growth when the backend is down
                or auth is misconfigured. Oldest spans are dropped first
                with a WARNING log. Defaults to ``max_batch_size * 10``.
        """
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval
        self.max_retained_spans = (
            max_retained_spans if max_retained_spans is not None else max_batch_size * 10
        )

        self.buffer: list[EnrichedSpan] = []
        self.last_flush_time = time.time()
        self.lock = Lock()
        # Held for flush()'s entire body (extraction through the network
        # call to failure-merge), so two flush() calls on different
        # threads (e.g. stop()'s final flush racing the background
        # thread's still-in-flight size-triggered one) never send
        # concurrently -- restores the ordering guarantee that holding
        # `self.lock` across the network call used to provide before that
        # was split out so add_span/add_spans would stop blocking on I/O.
        # add_span/add_spans never acquire this -- they must still never
        # block behind an in-flight send.
        self._send_lock = Lock()
        self._stop_flush_thread = False
        self._flush_thread: Thread | None = None
        # Set by add_span/add_spans when max_batch_size is crossed, to wake
        # the background flush thread immediately instead of doing the
        # network send inline on the caller's own thread. See
        # _start_flush_thread's flush_worker.
        self._flush_requested = Event()

        # Start background thread for time-based flushing
        self._start_flush_thread()

    def add_span(self, span: EnrichedSpan) -> None:
        """
        Add a span to the buffer.

        If this crosses max_batch_size, wakes the background flush thread
        to send the batch -- this call itself never does network I/O and
        never blocks on one, regardless of batch size.

        Args:
            span: EnrichedSpan to add
        """
        with self.lock:
            self.buffer.append(span)
            should_flush_now = len(self.buffer) >= self.max_batch_size

        if should_flush_now:
            self._flush_requested.set()

    def add_spans(self, spans: list[EnrichedSpan]) -> None:
        """
        Add multiple spans to the buffer.

        If this crosses max_batch_size, wakes the background flush thread
        to send the batch -- this call itself never does network I/O and
        never blocks on one, regardless of batch size.

        Args:
            spans: List of EnrichedSpan objects
        """
        with self.lock:
            self.buffer.extend(spans)
            should_flush_now = len(self.buffer) >= self.max_batch_size

        if should_flush_now:
            self._flush_requested.set()

    def flush(self) -> None:
        """
        Flush all buffered spans to backend.

        Synchronous: does not return until the send (and any
        retry/retention bookkeeping) completes. Does NOT hold self.lock
        during the network call itself -- only while extracting the batch
        beforehand and merging back whatever failed afterwards -- so
        add_span/add_spans callers on other threads are never blocked
        behind this call's I/O, whether this runs on the caller's own
        thread (an explicit flush()) or the background flush thread (a
        size- or time-triggered one).

        Holds self._send_lock for the whole call, so two flush() calls
        from different threads (e.g. stop()'s final flush racing the
        background thread's still-in-flight size-triggered one) always
        send one at a time, in the order they reached this method --
        never two concurrent outbound POSTs with no ordering guarantee
        between them. add_span/add_spans never touch self._send_lock.
        """
        with self._send_lock:
            with self.lock:
                if not self.buffer:
                    return
                spans_to_send = self.buffer.copy()
                span_count = len(spans_to_send)
                self.last_flush_time = time.time()
                # Optimistically drop the batch we're about to send now,
                # while still holding the lock, so add_span callers during
                # the unlocked network call below land after these spans
                # in self.buffer, not interleaved with them. (Always []:
                # spans_to_send is a full copy of self.buffer taken under
                # this same lock, and self._send_lock rules out another
                # flush() call having appended to self.buffer meanwhile.)
                self.buffer = []

            logger.debug(
                "Flushing spans from buffer",
                extra={
                    "span_count": span_count,
                    "buffer_size_before": span_count,
                },
            )

            # Retain only the spans that actually failed to send. Using
            # send_spans_with_failures (not the bool-returning send_spans)
            # gives per-group granularity: a multi-policy_id batch where
            # one group succeeds and another fails now retains only the
            # failing group's spans, so the succeeded group isn't
            # re-POSTed on the next flush (round-2 P1 #1.2 — the earlier
            # round-1 fix retained the whole batch, silently double-
            # delivering the succeeded group).
            # self.lock not held here -- this is the actual network call
            # (P2) -- but self._send_lock still is, so no other flush()
            # call can start sending until this one finishes.
            failed_spans = self.transport.send_spans_with_failures(spans_to_send)
            if not failed_spans:
                return

            with self.lock:
                # Put the failed spans back at the front (they're older,
                # keep relative retry order), ahead of anything add_span
                # appended to self.buffer while the send above was in
                # flight.
                self.buffer = failed_spans + self.buffer

                logger.warning(
                    "Buffer flush partially/fully failed — retaining failed spans for retry",
                    extra={
                        "attempted": span_count,
                        "failed": len(failed_spans),
                        "succeeded": span_count - len(failed_spans),
                        "buffer_size_after": len(self.buffer),
                        "max_retained_spans": self.max_retained_spans,
                    },
                )
                # Guard against runaway growth if the backend stays down
                # or auth is permanently misconfigured. Drop the oldest
                # first; the newer spans are more useful for live
                # debugging.
                if len(self.buffer) > self.max_retained_spans:
                    overflow = len(self.buffer) - self.max_retained_spans
                    dropped = self.buffer[:overflow]
                    self.buffer = self.buffer[overflow:]
                    logger.error(
                        "Buffer exceeded max_retained_spans — dropping oldest",
                        extra={
                            "dropped_count": len(dropped),
                            "retained_count": len(self.buffer),
                            "max_retained_spans": self.max_retained_spans,
                        },
                    )

    def should_flush(self) -> bool:
        """
        Check if buffer should be flushed based on time interval.

        Returns:
            bool: True if flush interval has elapsed
        """
        with self.lock:
            return (
                len(self.buffer) > 0 and (time.time() - self.last_flush_time) >= self.flush_interval
            )

    def _start_flush_thread(self) -> None:
        """Start background thread for time-based AND size-triggered flushing"""

        def flush_worker():
            while not self._stop_flush_thread:
                # Waits for either: max_batch_size crossed (add_span/
                # add_spans sets _flush_requested -- wakes immediately,
                # doesn't wait out the rest of the interval) or a plain
                # timeout (the existing time-based check below).
                woken_by_size_trigger = self._flush_requested.wait(timeout=self.flush_interval)
                if self._stop_flush_thread:
                    break
                if woken_by_size_trigger:
                    self._flush_requested.clear()
                    logger.debug("Size-triggered flush running on background thread")
                    try:
                        self.flush()
                    except Exception:
                        logger.exception(
                            "Size-triggered flush failed unexpectedly — flush thread continuing"
                        )
                elif self.should_flush():
                    logger.debug("Time-based flush triggered")
                    try:
                        self.flush()
                    except Exception:
                        # An uncaught exception here would otherwise
                        # propagate out of flush_worker and kill this
                        # daemon thread silently for the life of the
                        # process -- no more automatic flushes, ever,
                        # with nothing louder than a raw stderr traceback
                        # from Python's default thread excepthook.
                        # Validation at client construction (see
                        # client.py's _validate_header_value) closes the
                        # known trigger -- a header value that breaks
                        # HTTP encoding -- before it can ever reach this
                        # call; this catches whatever that validation
                        # didn't anticipate, and keeps the thread alive
                        # to try again next interval.
                        logger.exception(
                            "Time-based flush failed unexpectedly — flush thread continuing"
                        )

        self._flush_thread = Thread(target=flush_worker, daemon=True, name="TraceBufferFlushThread")
        self._flush_thread.start()
        logger.debug(f"Started time-based flush thread (interval: {self.flush_interval}s)")

    def stop(self) -> None:
        """
        Stop the buffer and flush all remaining spans.

        Should be called during shutdown to ensure all spans are sent.
        """
        self._stop_flush_thread = True
        self._flush_requested.set()  # wake flush_worker immediately, don't wait out flush_interval
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)
        # Final flush of any remaining spans
        self.flush()
        logger.debug("Buffer stopped and flushed")
