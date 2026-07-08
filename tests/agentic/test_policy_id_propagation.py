"""Tests for realtime_policy_id propagation: Client → Transport → resource attributes.

The contract (see llm-monitoring/internal/consumer/span_raw_consumer.go):
when a span carries policy.id as a resource attribute, the validation
consumer routes it through policy-driven evaluation instead of the
default feature_settings path. The agentic SDK is the producer side of
that contract — set realtime_policy_id on DisseqtAgenticClient and every span
batch the client emits carries policy.id in the OTel-style resource
attributes block.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.trace import DisseqtTrace
from disseqt_agentic_sdk.transport import HTTPTransport


class TestClientPolicyIdPassthrough:
    """Client(realtime_policy_id=...) hands it to the transport."""

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_realtime_policy_id_forwarded_to_transport(self, _buf, mock_transport):
        DisseqtAgenticClient(
            api_key="k",
            project_id="p",
            service_name="svc",
            endpoint="http://localhost/traces",
            realtime_policy_id="pol-123",
        )
        mock_transport.assert_called_once()
        _, kwargs = mock_transport.call_args
        assert kwargs["realtime_policy_id"] == "pol-123"

    @patch("disseqt_agentic_sdk.client.client.HTTPTransport")
    @patch("disseqt_agentic_sdk.client.client.TraceBuffer")
    def test_no_realtime_policy_id_passes_none(self, _buf, mock_transport):
        DisseqtAgenticClient(
            api_key="k",
            project_id="p",
            service_name="svc",
            endpoint="http://localhost/traces",
        )
        _, kwargs = mock_transport.call_args
        assert kwargs["realtime_policy_id"] is None


class TestTransportEmitsPolicyId:
    """HTTPTransport.send_spans includes policy.id in resource.attributes."""

    def _fake_span(self, span_id="s1") -> MagicMock:
        # Build a span object whose .trace_id / .to_dict() match what
        # send_spans expects, without depending on the real EnrichedSpan.
        span = MagicMock()
        span.trace_id = "trace-abc"
        # The transport reads .realtime_policy_id off the span object via
        # getattr(...); MagicMock would otherwise return a MagicMock here
        # and pollute the resource attrs. Default to empty string —
        # individual tests can override.
        span.realtime_policy_id = ""
        span.to_dict.return_value = {
            "trace_id": "trace-abc",
            "span_id": span_id,
            "parent_span_id": "",
            "name": "llm.call",
            "kind": "MODEL_EXEC",
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 2_000_000_000,
            "status_code": "OK",
            "attributes_json": "{}",
            "service_name": "svc",
            "service_version": "1.0",
            "environment": "test",
            "project_id": "p",
        }
        return span

    def test_realtime_policy_id_present_when_set(self):
        transport = HTTPTransport(
            endpoint="http://localhost/traces",
            api_key="k",
            realtime_policy_id="pol-456",
        )
        captured = {}

        def fake_post(url, json=None, headers=None, **kwargs):
            captured["payload"] = json
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]
        ok = transport.send_spans([self._fake_span()])
        assert ok is True

        attrs = captured["payload"]["resource"]["attributes"]
        assert attrs["policy.id"] == "pol-456"

    def test_realtime_policy_id_absent_when_not_set(self):
        transport = HTTPTransport(endpoint="http://localhost/traces", api_key="k")
        captured = {}

        def fake_post(url, json=None, headers=None, **kwargs):
            captured["payload"] = json
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]
        transport.send_spans([self._fake_span()])

        attrs = captured["payload"]["resource"]["attributes"]
        assert "policy.id" not in attrs


class TestPerTracePolicyOverride:
    """When traces in a single batch carry different per-trace policy
    ids, the transport sends one POST per distinct policy, each with
    the right resource.attributes['policy.id'].

    This is the core mechanism that lets two agents in the same app
    run under different policies without re-initialising the client.
    """

    def _fake_span(self, span_id: str, trace_id: str, policy_id: str = "") -> MagicMock:
        span = MagicMock()
        span.trace_id = trace_id
        span.realtime_policy_id = policy_id
        span.to_dict.return_value = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": "",
            "name": "agent.exec",
            "kind": "AGENT_EXEC",
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 2_000_000_000,
            "status_code": "OK",
            "attributes_json": "{}",
            "service_name": "my-app",
            "service_version": "1.0",
            "environment": "test",
            "project_id": "p",
        }
        return span

    def test_two_traces_with_different_policies_produce_two_posts(self):
        # Agent A's trace (policy A) + Agent B's trace (policy B), same
        # batch. The transport should issue two POSTs — one per policy.
        transport = HTTPTransport(endpoint="http://localhost/traces", api_key="k")
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]

        ok = transport.send_spans(
            [
                self._fake_span("s1", "trace-a", policy_id="policy-a"),
                self._fake_span("s2", "trace-b", policy_id="policy-b"),
            ]
        )

        assert ok is True
        assert len(posts) == 2

        policy_ids = {p["resource"]["attributes"].get("policy.id") for p in posts}
        assert policy_ids == {"policy-a", "policy-b"}

    def test_single_policy_batch_still_one_post(self):
        # Both traces use the same policy → one POST, both traces inside.
        transport = HTTPTransport(endpoint="http://localhost/traces", api_key="k")
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]

        transport.send_spans(
            [
                self._fake_span("s1", "trace-a", policy_id="policy-a"),
                self._fake_span("s2", "trace-b", policy_id="policy-a"),
            ]
        )

        assert len(posts) == 1
        assert posts[0]["resource"]["attributes"]["policy.id"] == "policy-a"
        # Both traces in one payload.
        assert len(posts[0]["traces"]) == 2

    def test_per_trace_policy_id_beats_client_default(self):
        # Client has a default policy, but the span's per-trace override
        # wins. (Same span has policy_id="trace-policy".)
        transport = HTTPTransport(
            endpoint="http://localhost/traces",
            api_key="k",
            realtime_policy_id="client-default-policy",
        )
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]

        transport.send_spans([self._fake_span("s1", "trace-a", policy_id="trace-policy")])

        assert posts[0]["resource"]["attributes"]["policy.id"] == "trace-policy"

    def test_empty_per_trace_policy_falls_back_to_client_default(self):
        transport = HTTPTransport(
            endpoint="http://localhost/traces",
            api_key="k",
            realtime_policy_id="client-default-policy",
        )
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]

        # No per-trace override on the span.
        transport.send_spans([self._fake_span("s1", "trace-a", policy_id="")])

        assert posts[0]["resource"]["attributes"]["policy.id"] == "client-default-policy"


class TestNoPolicyAnywhere:
    """The agentic SDK works when neither the client nor any trace sets
    a realtime_policy_id — spans go out without policy.id and
    llm-monitoring routes them through its feature_settings path."""

    def _fake_span(self, span_id: str, trace_id: str) -> MagicMock:
        span = MagicMock()
        span.trace_id = trace_id
        span.realtime_policy_id = ""  # no per-trace override
        span.to_dict.return_value = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": "",
            "name": "agent.exec",
            "kind": "AGENT_EXEC",
            "start_time_unix_nano": 1_000_000_000,
            "end_time_unix_nano": 2_000_000_000,
            "status_code": "OK",
            "attributes_json": "{}",
            "service_name": "my-app",
            "service_version": "1.0",
            "environment": "test",
            "project_id": "p",
        }
        return span

    def test_no_policy_id_anywhere_means_no_policy_attr_on_wire(self):
        # Neither the transport (client default) nor the span (per-trace
        # override) carries a policy_id. Resource block should NOT include
        # policy.id — llm-monitoring then falls back to feature_settings.
        transport = HTTPTransport(endpoint="http://localhost/traces", api_key="k")
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]

        ok = transport.send_spans([self._fake_span("s1", "trace-a")])

        assert ok is True
        assert len(posts) == 1
        attrs = posts[0]["resource"]["attributes"]
        # The other resource attrs (service.name etc.) still flow.
        assert attrs["service.name"] == "my-app"
        assert attrs["api.key"] == "k"
        # But there's no policy.id — explicit non-presence.
        assert "policy.id" not in attrs


class TestPerSpanPolicyOverride:
    """Per-span realtime_policy_id override.

    Precedence: span override → trace override → client default. A single
    trace can carry multiple policies by setting realtime_policy_id on
    individual start_span() calls; the transport buckets by effective
    policy_id and emits one POST per distinct policy.
    """

    def test_start_span_override_beats_trace_default(self):
        # Trace default is TRACE-P; the span asks for SPAN-P explicitly.
        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace.start_span("s1", SpanKind.INTERNAL, realtime_policy_id="SPAN-P")
        assert span.realtime_policy_id == "SPAN-P"

    def test_start_span_without_override_inherits_trace_default(self):
        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace.start_span("s1", SpanKind.INTERNAL)
        assert span.realtime_policy_id == "TRACE-P"

    def test_start_span_override_empty_string_is_no_policy(self):
        # Explicit empty string means "no policy for this span", even
        # when the trace has a default. Lets callers opt a specific
        # span out of policy evaluation.
        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace.start_span("s1", SpanKind.INTERNAL, realtime_policy_id="")
        assert span.realtime_policy_id == ""

    def test_start_span_override_with_no_trace_default(self):
        trace = DisseqtTrace(name="t")
        span = trace.start_span("s1", SpanKind.INTERNAL, realtime_policy_id="SPAN-P")
        assert span.realtime_policy_id == "SPAN-P"

    def test_mixed_span_policies_in_one_trace_produce_separate_posts(self):
        # A trace with default TRACE-P, one span overriding to SPAN-P,
        # two spans inheriting. Transport must send 2 POSTs — one per
        # distinct policy_id — each with the right resource.policy.id.
        trace = DisseqtTrace(
            name="t",
            org_id="",
            project_id="p",
            service_name="svc",
            environment="test",
            realtime_policy_id="TRACE-P",
        )
        trace.start_span("plan", SpanKind.AGENT_EXEC).end()
        trace.start_span("critical", SpanKind.MODEL_EXEC, realtime_policy_id="SPAN-P").end()
        trace.start_span("tool", SpanKind.TOOL_EXEC).end()
        trace.end()

        enriched = trace.to_enriched_spans()

        transport = HTTPTransport(endpoint="http://localhost/traces", api_key="k")
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]
        ok = transport.send_spans(enriched)

        assert ok is True
        assert len(posts) == 2

        by_policy = {p["resource"]["attributes"]["policy.id"]: p for p in posts}
        assert set(by_policy) == {"TRACE-P", "SPAN-P"}
        # 2 spans landed in the TRACE-P bucket, 1 in SPAN-P.
        assert sum(len(t["spans"]) for t in by_policy["TRACE-P"]["traces"]) == 2
        assert sum(len(t["spans"]) for t in by_policy["SPAN-P"]["traces"]) == 1

    def test_span_override_beats_client_default_via_transport(self):
        # No per-trace override, client default is CLIENT-P, span
        # overrides to SPAN-P. Transport should stamp SPAN-P.
        trace = DisseqtTrace(name="t", project_id="p", service_name="svc")
        trace.start_span("s1", SpanKind.INTERNAL, realtime_policy_id="SPAN-P").end()
        trace.end()

        transport = HTTPTransport(
            endpoint="http://localhost/traces",
            api_key="k",
            realtime_policy_id="CLIENT-P",
        )
        posts: list[dict] = []

        def fake_post(url, json=None, headers=None, **kwargs):
            posts.append(json)
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            return resp

        transport.session.post = fake_post  # type: ignore[assignment]
        transport.send_spans(trace.to_enriched_spans())

        assert len(posts) == 1
        assert posts[0]["resource"]["attributes"]["policy.id"] == "SPAN-P"


class TestHelpersForwardSpanPolicy:
    """The trace_llm_call / trace_agent_action / trace_tool_call helpers
    accept realtime_policy_id and thread it into the created span."""

    def test_trace_llm_call_forwards_policy(self):
        from disseqt_agentic_sdk.api.helpers import trace_llm_call

        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace_llm_call(
            trace,
            name="chat",
            model_name="gpt-4",
            provider="openai",
            realtime_policy_id="SPAN-P",
        )
        assert span.realtime_policy_id == "SPAN-P"

    def test_trace_agent_action_forwards_policy(self):
        from disseqt_agentic_sdk.api.helpers import trace_agent_action

        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace_agent_action(
            trace,
            name="plan",
            agent_name="a",
            realtime_policy_id="SPAN-P",
        )
        assert span.realtime_policy_id == "SPAN-P"

    def test_trace_tool_call_forwards_policy(self):
        from disseqt_agentic_sdk.api.helpers import trace_tool_call

        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        span = trace_tool_call(
            trace,
            name="weather",
            tool_name="get_weather",
            realtime_policy_id="SPAN-P",
        )
        assert span.realtime_policy_id == "SPAN-P"

    def test_helpers_without_policy_inherit_trace_default(self):
        from disseqt_agentic_sdk.api.helpers import (
            trace_agent_action,
            trace_llm_call,
            trace_tool_call,
        )

        trace = DisseqtTrace(name="t", realtime_policy_id="TRACE-P")
        s1 = trace_llm_call(trace, name="chat", model_name="gpt-4", provider="openai")
        s2 = trace_agent_action(trace, name="plan", agent_name="a")
        s3 = trace_tool_call(trace, name="weather", tool_name="get_weather")
        assert s1.realtime_policy_id == "TRACE-P"
        assert s2.realtime_policy_id == "TRACE-P"
        assert s3.realtime_policy_id == "TRACE-P"
