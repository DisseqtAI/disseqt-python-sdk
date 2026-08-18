"""
Fixtures for instrumentation tests.

We build a Client whose buffer captures every EnrichedSpan added to it
so tests can assert on emitted attributes without touching the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.instrumentation import uninstrument_all


class RecordingBuffer:
    """Drop-in TraceBuffer replacement — stores spans instead of sending them."""

    def __init__(self) -> None:
        self.spans: list = []

    def add_span(self, span) -> None:  # type: ignore[no-untyped-def]
        self.spans.append(span)

    def add_spans(self, spans) -> None:  # type: ignore[no-untyped-def]
        self.spans.extend(spans)

    def flush(self) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.fixture
def recording_client(monkeypatch):
    """
    Client wired to an in-memory RecordingBuffer. `client.buffer.spans`
    holds every EnrichedSpan the instrumentation emitted.
    """
    # Stub out both transport and buffer before construction.
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", MagicMock())
    monkeypatch.setattr(
        "disseqt_agentic_sdk.client.client.TraceBuffer", lambda **kw: RecordingBuffer()
    )
    client = DisseqtAgenticClient(
        api_key="test_key",
        project_id="test_proj",
        service_name="test_service",
        endpoint="http://localhost/v1/traces",
    )
    yield client
    uninstrument_all()
    client.shutdown()


def find_span(client: DisseqtAgenticClient, name: str):
    """Locate the last recorded span matching `name`. Raises if not found."""
    for span in reversed(client.buffer.spans):  # type: ignore[attr-defined]
        if span.name == name:
            return span
    names = [s.name for s in client.buffer.spans]  # type: ignore[attr-defined]
    raise AssertionError(f"no span named {name!r}; got {names}")
