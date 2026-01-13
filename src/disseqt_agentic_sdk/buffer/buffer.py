"""
Buffer for batching spans before sending to backend.
"""

import time
from threading import Lock, Thread

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
    ):
        """
        Initialize buffer.

        Args:
            transport: HTTPTransport instance for sending
            max_batch_size: Maximum number of spans per batch (triggers immediate flush)
            flush_interval: Flush interval in seconds (time-based flushing)
        """
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.flush_interval = flush_interval

        self.buffer: list[EnrichedSpan] = []
        self.last_flush_time = time.time()
        self.lock = Lock()
        self._stop_flush_thread = False
        self._flush_thread: Thread | None = None

        # Start background thread for time-based flushing
        self._start_flush_thread()

    def add_span(self, span: EnrichedSpan) -> None:
        """
        Add a span to the buffer.

        Automatically flushes if batch size is reached.

        Args:
            span: EnrichedSpan to add
        """
        with self.lock:
            self.buffer.append(span)

            # Flush if batch size reached
            if len(self.buffer) >= self.max_batch_size:
                self._flush_locked()

    def add_spans(self, spans: list[EnrichedSpan]) -> None:
        """
        Add multiple spans to the buffer.

        Args:
            spans: List of EnrichedSpan objects
        """
        with self.lock:
            self.buffer.extend(spans)

            # Flush if batch size reached
            if len(self.buffer) >= self.max_batch_size:
                self._flush_locked()

    def flush(self) -> None:
        """
        Flush all buffered spans to backend.
        """
        with self.lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush method (assumes lock is held)"""
        if not self.buffer:
            return

        spans_to_send = self.buffer.copy()
        span_count = len(spans_to_send)
        self.buffer.clear()
        self.last_flush_time = time.time()

        logger.debug(
            "Flushing spans from buffer",
            extra={
                "span_count": span_count,
                "buffer_size_before": span_count,
            },
        )

        # Send spans (release lock before network call)
        self.transport.send_spans(spans_to_send)

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
        """Start background thread for time-based flushing"""

        def flush_worker():
            while not self._stop_flush_thread:
                time.sleep(self.flush_interval)
                if self.should_flush():
                    logger.debug("Time-based flush triggered")
                    self.flush()

        self._flush_thread = Thread(target=flush_worker, daemon=True, name="TraceBufferFlushThread")
        self._flush_thread.start()
        logger.debug(f"Started time-based flush thread (interval: {self.flush_interval}s)")

    def stop(self) -> None:
        """
        Stop the buffer and flush all remaining spans.

        Should be called during shutdown to ensure all spans are sent.
        """
        self._stop_flush_thread = True
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=2.0)
        # Final flush of any remaining spans
        self.flush()
        logger.debug("Buffer stopped and flushed")
