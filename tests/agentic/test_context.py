"""
Unit tests for context management.
"""

import threading

from disseqt_agentic_sdk.context import (
    clear_context,
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.span import DisseqtSpan
from disseqt_agentic_sdk.trace import DisseqtTrace


class TestContext:
    """Tests for context management."""

    def test_trace_context(self):
        """Test trace context management."""
        assert get_current_trace() is None

        trace = DisseqtTrace(name="test_trace")
        set_current_trace(trace)

        assert get_current_trace() == trace

        set_current_trace(None)
        assert get_current_trace() is None

    def test_span_context(self):
        """Test span context management."""
        assert get_current_span() is None

        span = DisseqtSpan(trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL)
        set_current_span(span)

        assert get_current_span() == span

        set_current_span(None)
        assert get_current_span() is None

    def test_clear_context(self):
        """Test clearing all context."""
        trace = DisseqtTrace(name="test_trace")
        span = DisseqtSpan(trace_id="test_trace", name="test_span", kind=SpanKind.INTERNAL)

        set_current_trace(trace)
        set_current_span(span)

        assert get_current_trace() == trace
        assert get_current_span() == span

        clear_context()

        assert get_current_trace() is None
        assert get_current_span() is None

    def test_thread_local_context(self):
        """Test that context is thread-local."""
        trace1 = DisseqtTrace(name="trace1")
        trace2 = DisseqtTrace(name="trace2")

        set_current_trace(trace1)

        def set_trace_in_thread():
            set_current_trace(trace2)
            assert get_current_trace() == trace2

        thread = threading.Thread(target=set_trace_in_thread)
        thread.start()
        thread.join()

        # Original thread should still have trace1
        assert get_current_trace() == trace1
