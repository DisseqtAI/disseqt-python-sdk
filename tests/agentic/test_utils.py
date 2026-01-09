"""
Unit tests for utility functions.
"""

from datetime import datetime, timezone

from disseqt_agentic_sdk.utils.ids import generate_span_id, generate_trace_id
from disseqt_agentic_sdk.utils.time import (
    calculate_duration_ns,
    from_timestamp_ms,
    from_timestamp_ns,
    now_ms,
    now_ns,
    to_timestamp_ms,
    to_timestamp_ns,
)


class TestIDGeneration:
    """Tests for ID generation utilities."""

    def test_generate_trace_id(self):
        """Test trace ID generation."""
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32  # 32 hex characters
        assert all(c in "0123456789abcdef" for c in trace_id)

    def test_generate_trace_id_unique(self):
        """Test trace IDs are unique."""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_span_id(self):
        """Test span ID generation."""
        span_id = generate_span_id()
        assert isinstance(span_id, str)
        assert len(span_id) == 16  # 16 hex characters
        assert all(c in "0123456789abcdef" for c in span_id)

    def test_generate_span_id_unique(self):
        """Test span IDs are unique."""
        ids = {generate_span_id() for _ in range(100)}
        assert len(ids) == 100


class TestTimeUtilities:
    """Tests for time conversion utilities."""

    def test_now_ns(self):
        """Test nanoseconds timestamp generation."""
        ts = now_ns()
        assert isinstance(ts, int)
        assert ts > 0
        # Should be roughly current time (within 1 second)
        expected = datetime.now(timezone.utc).timestamp() * 1_000_000_000
        assert abs(ts - expected) < 1_000_000_000

    def test_now_ms(self):
        """Test milliseconds timestamp generation."""
        ts = now_ms()
        assert isinstance(ts, int)
        assert ts > 0
        # Should be roughly current time (within 1 second)
        expected = datetime.now(timezone.utc).timestamp() * 1_000
        assert abs(ts - expected) < 1_000

    def test_to_timestamp_ns(self):
        """Test datetime to nanoseconds conversion."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = to_timestamp_ns(dt)
        assert isinstance(ts, int)
        assert ts == 1704110400_000_000_000

    def test_from_timestamp_ns(self):
        """Test nanoseconds to datetime conversion."""
        ts = 1704110400_000_000_000
        dt = from_timestamp_ns(ts)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1

    def test_to_timestamp_ms(self):
        """Test datetime to milliseconds conversion."""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts = to_timestamp_ms(dt)
        assert isinstance(ts, int)
        assert ts == 1704110400_000

    def test_from_timestamp_ms(self):
        """Test milliseconds to datetime conversion."""
        ts = 1704110400_000
        dt = from_timestamp_ms(ts)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1

    def test_calculate_duration_ns(self):
        """Test duration calculation."""
        start = 1000_000_000_000
        end = 2000_000_000_000
        duration = calculate_duration_ns(start, end)
        assert duration == 1000_000_000_000

    def test_time_roundtrip(self):
        """Test roundtrip conversion."""
        dt = datetime.now(timezone.utc)
        ts_ns = to_timestamp_ns(dt)
        dt2 = from_timestamp_ns(ts_ns)
        # Should be within 1 microsecond
        assert abs((dt - dt2).total_seconds()) < 0.000001
