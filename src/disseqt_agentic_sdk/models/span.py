"""
Span models matching the backend EnrichedSpan structure.

This module contains the EnrichedSpan model that matches the backend's
flat_span.go EnrichedSpan struct exactly.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID


@dataclass
class EnrichedSpan:
    """
    EnrichedSpan - Minimal SDK span schema.

    Core fields only - all enrichment data goes in attributes_json.
    Backend will extract and enrich common fields into dedicated columns.
    """

    # Core span identity
    trace_id: UUID | str  # UUID type, serialized as string
    span_id: UUID | str  # UUID type, serialized as string
    parent_span_id: UUID | str | None = None  # Optional UUID
    name: str = ""
    kind: str = ""  # span_kind (MODEL_EXEC, TOOL_EXEC, AGENT_EXEC, etc.)
    root: bool = False

    # Timing (nanoseconds since epoch)
    start_time_unix_nano: int = 0
    end_time_unix_nano: int = 0
    duration_ns: int = 0

    # Status
    status_code: str = "OK"  # OK, ERROR
    status_message: str = ""

    # Multi-tenancy (required for routing)
    org_id: str = ""
    project_id: str = ""
    user_id: str = ""

    # Service information (required)
    service_name: str = ""
    service_version: str = "1.0.0"
    environment: str = "production"

    # Timestamp for partitioning (server time)
    dt: datetime | None = None  # Backend will set if not provided

    # JSON fields - all enrichment data goes here (OTEL attributes)
    attributes_json: str = "{}"  # Contains gen_ai.*, agent.*, tool.* attributes
    resource_attributes_json: str = "{}"
    events_json: str = "[]"

    # Optional scope information
    scope_name: str = ""
    scope_version: str = ""

    # Ingestion timestamp
    ingestion_time: datetime | None = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.
        Minimal schema - enrichment data in attributes_json.
        """
        # Convert UUIDs to strings for JSON serialization
        trace_id_str = str(self.trace_id) if isinstance(self.trace_id, UUID) else self.trace_id
        span_id_str = str(self.span_id) if isinstance(self.span_id, UUID) else self.span_id
        parent_span_id_str = (
            str(self.parent_span_id)
            if isinstance(self.parent_span_id, UUID)
            else (self.parent_span_id or "")
        )

        return {
            "trace_id": trace_id_str,
            "span_id": span_id_str,
            "parent_span_id": parent_span_id_str,
            "name": self.name,
            "kind": self.kind,
            "root": self.root,
            "start_time_unix_nano": self.start_time_unix_nano,
            "end_time_unix_nano": self.end_time_unix_nano,
            "duration_ns": self.duration_ns,
            "status_code": self.status_code,
            "status_message": self.status_message,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "dt": self.dt.isoformat() if self.dt else None,
            "attributes_json": self.attributes_json,
            "resource_attributes_json": self.resource_attributes_json,
            "events_json": self.events_json,
            "scope_name": self.scope_name,
            "scope_version": self.scope_version,
            "ingestion_time": self.ingestion_time.isoformat() if self.ingestion_time else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "EnrichedSpan":
        """Create EnrichedSpan from dictionary"""
        # Handle datetime fields
        dt = data.get("dt")
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        elif dt is None:
            dt = datetime.now(timezone.utc)

        ingestion_time = data.get("ingestion_time")
        if isinstance(ingestion_time, str):
            ingestion_time = datetime.fromisoformat(ingestion_time.replace("Z", "+00:00"))

        # Handle UUID fields - convert strings to UUID objects
        trace_id = data.get("trace_id", "")
        if isinstance(trace_id, str) and trace_id:
            try:
                trace_id = UUID(trace_id)
            except (ValueError, AttributeError):
                pass  # Keep as string if invalid UUID format

        span_id = data.get("span_id", "")
        if isinstance(span_id, str) and span_id:
            try:
                span_id = UUID(span_id)
            except (ValueError, AttributeError):
                pass  # Keep as string if invalid UUID format

        parent_span_id = data.get("parent_span_id")
        if isinstance(parent_span_id, str) and parent_span_id:
            try:
                parent_span_id = UUID(parent_span_id)
            except (ValueError, AttributeError):
                pass  # Keep as string if invalid UUID format
        elif not parent_span_id:
            parent_span_id = None

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=data.get("name", ""),
            kind=data.get("kind", ""),
            root=data.get("root", False),
            start_time_unix_nano=data.get("start_time_unix_nano", 0),
            end_time_unix_nano=data.get("end_time_unix_nano", 0),
            duration_ns=data.get("duration_ns", 0),
            status_code=data.get("status_code", "OK"),
            status_message=data.get("status_message", ""),
            org_id=data.get("org_id", ""),
            project_id=data.get("project_id", ""),
            user_id=data.get("user_id", ""),
            service_name=data.get("service_name", ""),
            service_version=data.get("service_version", "1.0.0"),
            environment=data.get("environment", "production"),
            dt=dt,
            attributes_json=data.get("attributes_json", "{}"),
            resource_attributes_json=data.get("resource_attributes_json", "{}"),
            events_json=data.get("events_json", "[]"),
            scope_name=data.get("scope_name", ""),
            scope_version=data.get("scope_version", ""),
            ingestion_time=ingestion_time,
        )
