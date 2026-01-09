"""
Unit tests for data models.
"""

from uuid import UUID

from disseqt_agentic_sdk.models.span import EnrichedSpan


class TestEnrichedSpan:
    """Tests for EnrichedSpan model."""

    def test_enriched_span_creation(self):
        """Test creating EnrichedSpan."""
        span = EnrichedSpan(
            trace_id="test_trace_123",
            span_id="test_span_456",
            name="test_span",
            kind="INTERNAL",
            org_id="org_1",
            project_id="proj_1",
            service_name="test_service",
        )

        assert span.trace_id == "test_trace_123"
        assert span.span_id == "test_span_456"
        assert span.name == "test_span"
        assert span.kind == "INTERNAL"
        assert span.org_id == "org_1"
        assert span.project_id == "proj_1"
        assert span.service_name == "test_service"

    def test_enriched_span_uuid_conversion(self):
        """Test UUID to string conversion."""
        trace_uuid = UUID("550e8400-e29b-41d4-a716-446655440000")
        span_uuid = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

        span = EnrichedSpan(
            trace_id=trace_uuid,
            span_id=span_uuid,
            name="test",
            kind="INTERNAL",
            org_id="org_1",
            project_id="proj_1",
            service_name="test",
        )

        # Should accept UUID objects
        assert isinstance(span.trace_id, (UUID, str))
        assert isinstance(span.span_id, (UUID, str))

        # to_dict should convert to strings
        span_dict = span.to_dict()
        assert isinstance(span_dict["trace_id"], str)
        assert isinstance(span_dict["span_id"], str)

    def test_enriched_span_to_dict(self):
        """Test serialization to dict."""
        span = EnrichedSpan(
            trace_id="test_trace",
            span_id="test_span",
            name="test",
            kind="INTERNAL",
            org_id="org_1",
            project_id="proj_1",
            service_name="test",
        )

        span_dict = span.to_dict()

        assert isinstance(span_dict, dict)
        assert span_dict["trace_id"] == "test_trace"
        assert span_dict["span_id"] == "test_span"
        assert span_dict["name"] == "test"
        assert span_dict["kind"] == "INTERNAL"

    def test_enriched_span_json_serialization(self):
        """Test JSON serialization."""
        span = EnrichedSpan(
            trace_id="test_trace",
            span_id="test_span",
            name="test",
            kind="INTERNAL",
            org_id="org_1",
            project_id="proj_1",
            service_name="test",
        )

        json_str = span.to_json()
        assert isinstance(json_str, str)
        assert "test_trace" in json_str
        assert "test_span" in json_str

    def test_enriched_span_from_dict(self):
        """Test deserialization from dict."""
        span_dict = {
            "trace_id": "test_trace",
            "span_id": "test_span",
            "name": "test",
            "kind": "INTERNAL",
            "org_id": "org_1",
            "project_id": "proj_1",
            "service_name": "test",
            "status_code": "OK",
            "attributes_json": '{"key": "value"}',
            "resource_attributes_json": "{}",
            "events_json": "[]",
        }

        span = EnrichedSpan.from_dict(span_dict)

        assert span.trace_id == "test_trace"
        assert span.span_id == "test_span"
        assert span.name == "test"
        assert span.kind == "INTERNAL"
