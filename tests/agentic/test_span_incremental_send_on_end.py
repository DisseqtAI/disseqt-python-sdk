"""
Regression tests for P1: DisseqtSpan.end() must deliver the span exactly
once, not just __exit__().

Root cause: the incremental-send push to `client.buffer.add_span(...)` used
to live ONLY inside DisseqtSpan.__exit__, never inside end() itself. Every
helper in api/helpers.py (trace_llm_call, trace_agent_action,
trace_tool_call) calls `trace.start_span(...)` directly and returns the bare
span -- it is never used as a `with span:` context manager, so __exit__
never ran. DisseqtTrace.end()'s cleanup sweep calls `span.end()` (not
`span.__exit__()`) on any span the caller didn't explicitly end, so even
that safety net never delivered anything. Net effect: the documented
`trace_llm_call` usage inside `with start_trace(...) as trace:` sent zero
spans, silently.
"""

from unittest.mock import MagicMock

from disseqt_agentic_sdk.api.helpers import trace_llm_call
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.span import DisseqtSpan
from disseqt_agentic_sdk.trace import DisseqtTrace


def _make_client_stub():
    client = MagicMock()
    client.buffer = MagicMock()
    return client


class TestSpanEndDelivers:
    def test_end_without_a_with_block_still_pushes_to_the_buffer(self):
        """A span ended by calling .end() directly (no `with`) must still be sent."""
        client = _make_client_stub()
        span = DisseqtSpan(
            trace_id="t1", name="manual_end_span", kind=SpanKind.INTERNAL, client=client
        )

        span.end()

        client.buffer.add_span.assert_called_once()
        (enriched_span,) = client.buffer.add_span.call_args.args
        assert enriched_span.name == "manual_end_span"

    def test_end_is_idempotent_and_never_double_sends(self):
        client = _make_client_stub()
        span = DisseqtSpan(trace_id="t1", name="s", kind=SpanKind.INTERNAL, client=client)

        span.end()
        span.end()  # calling twice must not push twice

        client.buffer.add_span.assert_called_once()

    def test_with_block_still_sends_exactly_once_not_twice(self):
        """__exit__ delegates to end() now -- must not double-push."""
        client = _make_client_stub()
        with DisseqtSpan(trace_id="t1", name="s", kind=SpanKind.INTERNAL, client=client) as span:
            pass

        assert span.end_time_ns is not None
        client.buffer.add_span.assert_called_once()


class TestTraceLlmCallDeliversViaTraceEndSweep:
    def test_documented_usage_delivers_exactly_one_span(self):
        """
        The documented pattern:
            with start_trace(client, "my_trace") as trace:
                trace_llm_call(trace, name=..., model_name=..., provider=...)
        must deliver exactly one span, with no extra .end()/.send() step by
        the caller. This reproduces the exact bug report: "ran the
        documented usage... zero POSTs" (buffer.add_span is the point right
        before the network boundary the SDK's own buffer flush would then
        POST from).
        """
        client = _make_client_stub()
        trace = DisseqtTrace(
            name="my_trace",
            project_id="proj",
            service_name="svc",
            client=client,  # what start_trace() wires up internally
        )

        span = trace_llm_call(
            trace,
            name="chat_completion",
            model_name="gpt-4",
            provider="openai",
        )
        assert span.end_time_ns is None  # confirms trace_llm_call itself never ends it

        trace.end()  # what TraceWrapper.__exit__ does on `with start_trace(...) as trace:` exit

        client.buffer.add_span.assert_called_once()
        (enriched_span,) = client.buffer.add_span.call_args.args
        assert enriched_span.name == "chat_completion"

    def test_caller_can_still_use_the_returned_span(self):
        """Fix must not break callers who read/mutate the returned span."""
        client = _make_client_stub()
        trace = DisseqtTrace(name="t", project_id="p", service_name="s", client=client)

        span = trace_llm_call(trace, name="call", model_name="gpt-4", provider="openai")
        span.set_attribute("custom.tag", "value")  # still a live, usable DisseqtSpan
        assert span.attributes["custom.tag"] == "value"

        trace.end()
        client.buffer.add_span.assert_called_once()
