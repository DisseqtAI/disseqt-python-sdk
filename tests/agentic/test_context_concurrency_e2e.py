"""
End-to-end regression tests for the context.py contextvars migration.

IMPORTANT (found the hard way, see git history of this file): the
concurrency bug this migration fixes does NOT reproduce through the
documented `with start_trace(...) as trace: with trace.start_span(...):`
pattern, because a caller using that pattern already threads the
specific `trace`/`parent span` object explicitly through their own call
stack -- `trace.start_span()` always attaches to `self.trace_id`, never
to whatever `get_current_trace()` happens to return. An earlier version
of this file tested exactly that pattern and all four scenarios below
passed even against the OLD threading.local() code -- proving those
tests never exercised the vulnerable path at all.

The actual vulnerable path is the IMPLICIT bootstrap: `agent_span()`
(and anything built on `_get_or_bootstrap_trace` -- the `@disseqt_trace`
decorator, auto-instrumentation) calls `get_current_trace()` with no
explicit trace object in hand, to decide "is there already a trace
running I should nest under, or do I need to create one?". Two
concurrent flows that both rely on this ARE vulnerable to seeing each
other's "current trace" under threading.local() (same OS thread, shared
slot) -- one flow's span can end up silently nested inside the other
flow's trace instead of getting its own. contextvars.ContextVar isolates
this correctly per asyncio Task. These tests exercise agent_span()
specifically for that reason.

Unlike test_tool_result.py's TestLaneB::test_concurrent_agent_spans_are_isolated
(which asserts against an in-memory RecordingBuffer), these tests run a
real local HTTP stub server and assert on the actual JSON body it
received -- trace id, parent span id, attributes, and span count -- the
same shape llm-monitoring's real ingest endpoint would see on the wire.
Fake API keys only; no call ever reaches a real Disseqt server.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient, agent_span

_POLL_TIMEOUT = 5.0


class _RecordingHandler(BaseHTTPRequestHandler):
    """Records every POST body this stub server receives."""

    received: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming convention
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length) if length else b""
        self.received.append(json.loads(body_raw) if body_raw else {})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence default stderr access logging


@pytest.fixture
def stub_server():
    """
    A real local HTTP server (127.0.0.1, OS-assigned ephemeral port via
    bind-to-0 -- structurally collision-free, not just "probably free").
    """
    _RecordingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, _RecordingHandler.received
    finally:
        server.shutdown()
        thread.join(timeout=_POLL_TIMEOUT)


def _make_client(server: HTTPServer, **overrides: Any) -> DisseqtAgenticClient:
    port = server.server_port
    kwargs: dict[str, Any] = {
        "api_key": "fake-api-key",
        "project_id": "fake-project-id",
        "service_name": "context-concurrency-e2e",
        "endpoint": f"http://127.0.0.1:{port}/v1/traces",
        "application_id": "fake-application-id",
        "flush_interval": 60.0,  # only explicit client.flush() calls should deliver
    }
    kwargs.update(overrides)
    return DisseqtAgenticClient(**kwargs)


def _all_spans(received: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every span across every trace across every POST received."""
    spans = []
    for payload in received:
        for trace in payload.get("traces", []):
            for span in trace.get("spans", []):
                spans.append({**span, "_traceId": trace["traceId"]})
    return spans


def _find_span(spans: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [s for s in spans if s["name"] == name]
    assert len(matches) == 1, f"expected exactly one span named {name!r}, got {matches}"
    return matches[0]


class TestConcurrentAsyncioTasks:
    def test_two_concurrent_agent_spans_get_separate_traces(self, stub_server):
        # Two flows, each relying on agent_span()'s IMPLICIT
        # get_current_trace() lookup (no trace object threaded by the
        # caller) to decide whether to bootstrap a new trace. Under
        # threading.local(), flow B's lookup could see flow A's trace
        # (same OS thread, shared slot) and wrongly nest under it
        # instead of getting its own -- that's the bug this proves fixed.
        server, received = stub_server
        client = _make_client(server)
        try:

            async def one_flow(agent_name: str, delay_ms: float) -> None:
                with agent_span(client, agent_name) as span:
                    span.set_attribute("flow", agent_name)
                    await asyncio.sleep(delay_ms / 1000)

            async def main() -> None:
                # Staggered delays so the two flows' awaits genuinely
                # interleave on the event loop, not run back-to-back.
                await asyncio.gather(
                    one_flow("alpha_agent", 15),
                    one_flow("beta_agent", 5),
                )

            asyncio.run(main())
            client.flush()

            spans = _all_spans(received)
            alpha = _find_span(spans, "alpha_agent")
            beta = _find_span(spans, "beta_agent")

            # The bug: beta's span silently becomes a CHILD of alpha's
            # trace (or vice versa) instead of getting its own trace.
            assert alpha["_traceId"] != beta["_traceId"], (
                "the two concurrent agent_span() flows were merged onto "
                "the same trace -- current-trace context leaked across "
                "the asyncio Tasks"
            )
            assert alpha["attributes"]["flow"] == "alpha_agent"
            assert beta["attributes"]["flow"] == "beta_agent"
            # Each is the ROOT of its own trace, not nested under the
            # other flow's span.
            assert alpha.get("parentSpanId", "") == ""
            assert beta.get("parentSpanId", "") == ""
            assert len(spans) == 2
        finally:
            client.shutdown()


class TestTwoThreads:
    def test_two_threads_get_separate_traces(self, stub_server):
        # Same implicit-bootstrap mechanism, but across two real OS
        # threads instead of two asyncio Tasks. threading.local() already
        # isolates per-OS-thread (that was never the bug -- the bug is
        # specifically asyncio Tasks sharing ONE thread) -- this proves
        # the contextvars migration didn't regress the case that already
        # worked.
        server, received = stub_server
        client = _make_client(server)
        try:
            barrier = threading.Barrier(2, timeout=_POLL_TIMEOUT)

            def one_flow(agent_name: str) -> None:
                with agent_span(client, agent_name) as span:
                    span.set_attribute("flow", agent_name)
                    # Force genuine interleaving: both threads must be
                    # inside their own agent_span before either exits.
                    barrier.wait()

            t1 = threading.Thread(target=one_flow, args=("gamma_agent",))
            t2 = threading.Thread(target=one_flow, args=("delta_agent",))
            t1.start()
            t2.start()
            t1.join(timeout=_POLL_TIMEOUT)
            t2.join(timeout=_POLL_TIMEOUT)
            client.flush()

            spans = _all_spans(received)
            gamma = _find_span(spans, "gamma_agent")
            delta = _find_span(spans, "delta_agent")

            assert gamma["_traceId"] != delta["_traceId"]
            assert gamma["attributes"]["flow"] == "gamma_agent"
            assert delta["attributes"]["flow"] == "delta_agent"
            assert gamma.get("parentSpanId", "") == ""
            assert delta.get("parentSpanId", "") == ""
            assert len(spans) == 2
        finally:
            client.shutdown()


class TestNestedSpans:
    def test_nested_agent_spans_get_correct_parent_and_context_restores_after_exit(
        self, stub_server
    ):
        server, received = stub_server
        client = _make_client(server)
        try:
            with agent_span(client, "outer_agent"):
                with agent_span(client, "inner_agent"):
                    pass  # inner is now current-span; outer's trace stays open (owns_trace=False for inner)
                # inner has exited -- current-span context must have
                # restored to "outer" (via the token captured when
                # inner's span was constructed), not stayed on inner.
                with agent_span(client, "sibling_agent"):
                    pass
            client.flush()

            spans = _all_spans(received)
            outer = _find_span(spans, "outer_agent")
            inner = _find_span(spans, "inner_agent")
            sibling = _find_span(spans, "sibling_agent")

            # All three share ONE trace -- inner/sibling correctly nested
            # under the still-open outer trace (_get_or_bootstrap_trace
            # found it "current" and did not start a second one).
            assert inner["_traceId"] == outer["_traceId"]
            assert sibling["_traceId"] == outer["_traceId"]
            assert outer.get("parentSpanId", "") == ""
            assert inner["parentSpanId"] == outer["spanId"]
            # The real assertion: sibling's parent is "outer", proving
            # context correctly popped back to outer after inner's
            # exit -- not left pointing at inner, and not cleared to
            # root/None either.
            assert sibling["parentSpanId"] == outer["spanId"]
            assert len(spans) == 3
        finally:
            client.shutdown()


class TestExceptionInsideSpan:
    def test_exception_inside_agent_span_propagates_and_context_still_restores(self, stub_server):
        # Note: agent_span()'s cleanup calls span.__exit__(None, None,
        # None) unconditionally in its `finally` (see
        # instrumentation/_tool_result.py) -- it does not forward the
        # real exception into the wrapped span's own error-marking, so
        # this span will NOT come through with status="ERROR". That's a
        # separate, pre-existing behavior of agent_span() and out of
        # scope here. What this test actually proves: the exception
        # still propagates out to the caller (cleanup doesn't swallow
        # it), and current-trace/current-span context still correctly
        # restores afterward -- a SUBSEQUENT agent_span() call gets a
        # clean, separate trace, not corrupted or stuck on the failed one.
        server, received = stub_server
        client = _make_client(server)
        try:
            with pytest.raises(ValueError, match="boom"):
                with agent_span(client, "failing_agent"):
                    raise ValueError("boom")

            with agent_span(client, "clean_agent"):
                pass
            client.flush()

            spans = _all_spans(received)
            failing = _find_span(spans, "failing_agent")
            clean = _find_span(spans, "clean_agent")

            assert failing.get("parentSpanId", "") == ""
            assert clean.get("parentSpanId", "") == ""
            assert clean["_traceId"] != failing["_traceId"], (
                "the post-exception agent_span() call incorrectly nested "
                "under (or reused) the failed flow's trace -- context "
                "was not properly restored after the exception"
            )
            assert len(spans) == 2
        finally:
            client.shutdown()
