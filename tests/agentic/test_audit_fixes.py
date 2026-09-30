"""Regression tests for the SDK audit fixes (M02/F1/F3, M05, M07, M08, M09, M15-M20)."""

from __future__ import annotations

import contextvars
import http.server
import io
import json
import sys
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from urllib3.util.retry import Retry

import disseqt_logging
from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk.buffer.buffer import TraceBuffer
from disseqt_agentic_sdk.instrumentation import _utils as inst_utils
from disseqt_agentic_sdk.instrumentation.base import (
    _naive_version_lt,
    _version_lt,
)
from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.span import DisseqtSpan
from disseqt_agentic_sdk.transport import http as http_mod
from disseqt_agentic_sdk.transport.http import HTTPTransport
from disseqt_logging.redaction import SENSITIVE_VALUE_TOKEN, redact_field


def _span(span_id="s1", trace_id="t1", **kw) -> EnrichedSpan:
    base = {
        "trace_id": trace_id,
        "span_id": span_id,
        "name": "n",
        "kind": "MODEL_EXEC",
        "project_id": "p",
        "service_name": "svc",
        "service_version": "1.0",
        "environment": "test",
    }
    base.update(kw)
    return EnrichedSpan(**base)


# ---------------------------------------------------------------------------
# M05: span attribute serialisation
# ---------------------------------------------------------------------------
class TestAttributesJson:
    def _attrs(self, **attrs):
        span = DisseqtSpan(trace_id="t", name="n", kind="MODEL_EXEC")
        span.attributes.update(attrs)
        return json.loads(span.to_enriched_span().attributes_json)

    def test_datetime_set_bytes_do_not_raise(self):
        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        out = self._attrs(when=now, tags={"a"}, blob=b"xy", ok=1)
        assert out["ok"] == 1
        assert "2026" in out["when"]
        assert out["tags"] == "{'a'}"
        assert out["blob"] == "b'xy'"

    def test_bad_attribute_dropped_not_span(self):
        circular: list = []
        circular.append(circular)

        class Boom:
            def __str__(self):
                raise RuntimeError("no str")

            __repr__ = __str__

        out = self._attrs(good="yes", circ=circular, boom=Boom())
        assert out["good"] == "yes"
        assert isinstance(out.get("circ", ""), str)  # coerced to str (or dropped)
        assert out["boom"] == "<unserializable Boom>"

    def test_non_string_keys_coerced(self):
        out = self._attrs(nested={("a", 1): "v"})
        assert isinstance(out["nested"], (str, dict))


# ---------------------------------------------------------------------------
# M02 / F1 / F3: retry config, permanent vs retryable
# ---------------------------------------------------------------------------
class _Handler(http.server.BaseHTTPRequestHandler):
    statuses: list[int] = []
    hits = 0

    def do_POST(self):  # noqa: N802
        type(self).hits += 1
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        seq = type(self).statuses
        status = seq[min(type(self).hits - 1, len(seq) - 1)]
        self.send_response(status)
        if status == 503:
            self.send_header("Retry-After", "0")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    def make(statuses):
        handler = type("H", (_Handler,), {"statuses": statuses, "hits": 0})
        srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return srv, handler

    servers: list = []
    yield make
    for s in servers:
        s.shutdown()
        s.server_close()


def _fast(transport: HTTPTransport) -> HTTPTransport:
    transport.session.get_adapter("http://").max_retries.backoff_factor = 0
    return transport


class TestRetryPolicy:
    def test_retry_allows_post_and_respects_retry_after(self):
        retry = (
            HTTPTransport("http://x/v1/traces", api_key="k")
            .session.get_adapter("https://x")
            .max_retries
        )
        assert isinstance(retry, Retry)
        assert retry.is_retry("POST", 503)
        assert retry.is_retry("POST", 429)
        assert "POST" in retry.allowed_methods
        assert "GET" in retry.allowed_methods
        assert retry.respect_retry_after_header is True

    def test_real_server_503_then_200_is_retried(self, server):
        srv, handler = server([503, 503, 200])
        t = _fast(HTTPTransport(f"http://127.0.0.1:{srv.server_port}/v1/traces", api_key="k"))
        assert t.send_spans([_span()]) is True
        assert handler.hits == 3

    def test_real_server_400_dropped_not_retryable(self, server):
        srv, handler = server([400])
        t = _fast(HTTPTransport(f"http://127.0.0.1:{srv.server_port}/v1/traces", api_key="k"))
        spans = [_span()]
        assert t.send_spans_with_failures(spans) == []
        assert handler.hits == 1  # no re-POST of a permanent failure
        result = t.send_spans_classified(spans)
        assert result.retryable == [] and result.permanent == spans
        assert t.send_spans(spans) is False

    def _post_raising(self, status):
        resp = Mock()
        resp.status_code = status
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            str(status), response=resp
        )
        return resp

    @pytest.mark.parametrize(
        "status,permanent",
        [(400, True), (404, True), (422, True), (408, False), (429, False), (500, False)],
    )
    def test_classification(self, status, permanent):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        t.session.post = lambda *a, **k: self._post_raising(status)
        r = t.send_spans_classified([_span()])
        assert bool(r.permanent) is permanent
        assert bool(r.retryable) is (not permanent)

    def test_buffer_does_not_retain_permanently_rejected(self):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        t.session.post = lambda *a, **k: self._post_raising(400)
        buf = TraceBuffer(t, max_batch_size=100, flush_interval=60)
        try:
            buf.add_span(_span())
            buf.flush()
            assert buf.buffer == []
        finally:
            buf.stop()

    def test_buffer_retains_retryable(self):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        t.session.post = lambda *a, **k: self._post_raising(503)
        buf = TraceBuffer(t, max_batch_size=100, flush_interval=60)
        try:
            buf.add_span(_span())
            buf.flush()
            assert len(buf.buffer) == 1
        finally:
            t.session.post = lambda *a, **k: self._ok()
            buf.stop()

    @staticmethod
    def _ok():
        r = Mock()
        r.status_code = 200
        r.raise_for_status.return_value = None
        return r


# ---------------------------------------------------------------------------
# M07: group by full resource identity
# ---------------------------------------------------------------------------
class TestResourceGrouping:
    def _run(self, spans):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        posts = []

        def fake_post(url, json=None, headers=None, **kw):
            posts.append((json, headers))
            r = Mock()
            r.status_code = 200
            r.raise_for_status.return_value = None
            return r

        t.session.post = fake_post
        assert t.send_spans(spans) is True
        return posts

    def test_same_identity_single_post(self):
        assert len(self._run([_span("a"), _span("b", trace_id="t2")])) == 1

    @pytest.mark.parametrize(
        "field,value",
        [
            ("project_id", "p2"),
            ("service_name", "other"),
            ("service_version", "2.0"),
            ("environment", "prod"),
            ("realtime_policy_id", "pol"),
        ],
    )
    def test_differing_identity_splits_posts(self, field, value):
        posts = self._run([_span("a"), _span("b", **{field: value})])
        assert len(posts) == 2

    def test_each_post_carries_its_own_resource(self):
        posts = self._run(
            [_span("a", project_id="p1"), _span("b", project_id="p2", service_name="z")]
        )
        seen = {
            (p["resource"]["attributes"]["project.id"], p["resource"]["attributes"]["service.name"])
            for p, _ in posts
        }
        assert seen == {("p1", "svc"), ("p2", "z")}
        assert {h["X-Project-Id"] for _, h in posts} == {"p1", "p2"}


# ---------------------------------------------------------------------------
# M08: one-way process-wide capture_content
# ---------------------------------------------------------------------------
class TestCaptureContentOneWay:
    def test_true_does_not_lift_process_wide_off(self, monkeypatch):
        monkeypatch.setattr(inst_utils, "_capture_content_default", True)
        seen: dict[str, bool] = {}

        def body():
            inst_utils.set_capture_content(False)
            inst_utils.set_capture_content(True)
            seen["own_context"] = inst_utils.get_capture_content()
            t = threading.Thread(target=lambda: seen.update(fresh=inst_utils.get_capture_content()))
            t.start()
            t.join()

        contextvars.copy_context().run(body)
        assert seen["own_context"] is True  # explicit call honored in its own context
        assert seen["fresh"] is False  # but a fresh thread stays private

    def test_false_still_propagates_to_fresh_thread(self, monkeypatch):
        monkeypatch.setattr(inst_utils, "_capture_content_default", True)
        seen: dict[str, bool] = {}

        def body():
            inst_utils.set_capture_content(False)
            t = threading.Thread(target=lambda: seen.update(fresh=inst_utils.get_capture_content()))
            t.start()
            t.join()

        contextvars.copy_context().run(body)
        assert seen["fresh"] is False


# ---------------------------------------------------------------------------
# M09: shutdown hygiene
# ---------------------------------------------------------------------------
class TestShutdown:
    def test_shutdown_closes_transport_unregisters_atexit_idempotent(self):
        with (
            patch("disseqt_agentic_sdk.client.client.HTTPTransport") as ht,
            patch("disseqt_agentic_sdk.client.client.TraceBuffer"),
            patch("disseqt_agentic_sdk.client.client.atexit") as ax,
        ):
            client = DisseqtAgenticClient(
                api_key="k",
                project_id="p",
                service_name="s",
                endpoint="http://x/v1/traces",
                application_id="app",
            )
            ax.register.assert_called_once_with(client.shutdown)
            client.shutdown()
            client.shutdown()
        assert ax.unregister.call_args_list[0].args == (client.shutdown,)
        assert ht.return_value.close.called

    def test_transport_close_closes_session_and_is_idempotent(self):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        t.session = MagicMock()
        t.close()
        t.close()
        assert t.session.close.call_count == 2

    def test_buffer_stop_is_prompt(self):
        import time

        buf = TraceBuffer(MagicMock(send_spans_with_failures=lambda s: []), flush_interval=30)
        start = time.monotonic()
        buf.stop()
        assert time.monotonic() - start < 1.5
        assert not buf._flush_thread.is_alive()

    def test_buffer_stop_logs_join_timeout(self):
        buf = TraceBuffer(MagicMock(send_spans_with_failures=lambda s: []), flush_interval=30)
        real = buf._flush_thread
        buf._flush_thread = Mock(is_alive=lambda: True, join=lambda timeout=None: None)
        with patch("disseqt_agentic_sdk.buffer.buffer.logger") as log:
            buf.stop()
        assert log.warning.called
        real.join(timeout=2)


# ---------------------------------------------------------------------------
# M15: auth-failure stderr latch
# ---------------------------------------------------------------------------
class TestAuthStderrLatch:
    @pytest.fixture(autouse=True)
    def _clock(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(http_mod, "time", SimpleNamespace(monotonic=lambda: self.now))
        monkeypatch.setattr(http_mod, "_SILENCE_AUTH_STDERR", False)
        http_mod._reset_auth_failure_latch()

    def _emit(self, capsys) -> int:
        http_mod._write_auth_failure_to_stderr(401, "http://x")
        return capsys.readouterr().err.count("CRITICAL")

    def test_first_emits_then_throttled_with_growing_interval_then_reset(self, capsys):
        assert self._emit(capsys) == 1
        assert self._emit(capsys) == 0
        self.now += 30
        assert self._emit(capsys) == 0
        self.now += 31  # > 60s since first
        assert self._emit(capsys) == 1
        self.now += 100  # interval doubled to 120s
        assert self._emit(capsys) == 0
        self.now += 21
        assert self._emit(capsys) == 1
        http_mod._reset_auth_failure_latch()  # what a successful send does
        assert self._emit(capsys) == 1

    def test_success_resets_latch(self, capsys):
        assert self._emit(capsys) == 1
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        ok = Mock(status_code=200)
        ok.raise_for_status.return_value = None
        t.session.post = lambda *a, **k: ok
        assert t.send_spans([_span()]) is True
        assert self._emit(capsys) == 1

    def test_thread_safe_single_emit(self, capsys):
        threads = [
            threading.Thread(target=http_mod._write_auth_failure_to_stderr, args=(401, "e"))
            for _ in range(20)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert capsys.readouterr().err.count("CRITICAL") == 1


# ---------------------------------------------------------------------------
# M16: version comparison
# ---------------------------------------------------------------------------
class TestVersionLt:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("1.0.0rc1", "1.0.0", True),
            ("1.0.0b2", "1.0.0", True),
            ("1.0.0a1", "1.0.0b1", True),
            ("1.0.0", "1.0.0rc1", False),
            ("1.0.0.post1", "1.0.0", False),
            ("1.9.0", "1.10.0", True),
            ("2.0.0", "2.0.0", False),
        ],
    )
    def test_pep440(self, a, b, expected):
        assert _version_lt(a, b) is expected

    def test_falls_back_when_packaging_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "packaging.version", None)
        assert _version_lt("1.2.3", "1.10.0") is True
        assert _version_lt("1.10.0", "1.2.3") is False
        assert _naive_version_lt("1.0.0", "1.0.1") is True

    def test_invalid_version_falls_back(self):
        assert _version_lt("not-a-version", "1.0.0") is True


# ---------------------------------------------------------------------------
# M17: realtime_policy_id round trip
# ---------------------------------------------------------------------------
class TestRealtimePolicyRoundTrip:
    def test_to_from_dict(self):
        s = _span(realtime_policy_id="pol-1")
        d = s.to_dict()
        assert d["realtime_policy_id"] == "pol-1"
        assert EnrichedSpan.from_dict(d).realtime_policy_id == "pol-1"
        assert EnrichedSpan.from_dict({}).realtime_policy_id == ""

    def test_wire_payload_unchanged(self):
        t = HTTPTransport("http://x/v1/traces", api_key="k")
        payloads = []

        def fake_post(url, json=None, **kw):
            payloads.append(json)
            return Mock(status_code=200, raise_for_status=lambda: None)

        t.session.post = fake_post
        t.send_spans([_span(realtime_policy_id="pol-1")])
        body = payloads[0]
        assert body["resource"]["attributes"]["policy.id"] == "pol-1"
        assert "realtime_policy_id" not in json.dumps(body)


# ---------------------------------------------------------------------------
# M18: span logging
# ---------------------------------------------------------------------------
class TestSpanLogging:
    def test_buffer_failure_logged_via_package_logger(self):
        client = MagicMock()
        client.buffer.add_span.side_effect = RuntimeError("kaput")
        span = DisseqtSpan(trace_id="t", name="n", kind="MODEL_EXEC", client=client)
        with patch("disseqt_agentic_sdk.span.span.logger") as log:
            with span:
                pass
        assert log.warning.called
        assert log.warning.call_args.kwargs["extra"]["span_id"] == span.span_id
        assert "kaput" in log.warning.call_args.kwargs["extra"]["error"]


# ---------------------------------------------------------------------------
# M20: redaction recursion + exc_text
# ---------------------------------------------------------------------------
class TestRedaction:
    def test_recurses_into_containers(self):
        out = redact_field(
            "payload",
            {
                "user": "bob@example.com",
                "password": "hunter2",
                "items": ["call 123456789012", ("a@b.co",), {"api_key": "zzz"}],
                "tags": {"x@y.io"},
                "n": 5,
            },
        )
        assert out["user"] == "[EMAIL]"
        assert out["password"] == SENSITIVE_VALUE_TOKEN
        assert out["items"][0] == "call [PHONE]"
        assert out["items"][1] == ("[EMAIL]",)
        assert out["items"][2]["api_key"] == SENSITIVE_VALUE_TOKEN
        assert out["tags"] == ["[EMAIL]"]
        assert out["n"] == 5

    def test_sensitive_parent_redacts_nested_strings(self):
        out = redact_field("session", {"a": {"b": ["plain"]}})
        assert out == {"a": {"b": [SENSITIVE_VALUE_TOKEN]}}

    def test_depth_cap_fails_closed(self):
        deep: dict = {}
        cur = deep
        for _ in range(30):
            cur["k"] = {}
            cur = cur["k"]
        cur["k"] = "leaf@example.com"
        out = redact_field("x", deep)
        assert "leaf@example.com" not in json.dumps(out)
        assert SENSITIVE_VALUE_TOKEN in json.dumps(out)

    def test_digest_untouched(self):
        d = disseqt_logging.digest("hello 123456789012")
        assert redact_field("payload_digest", {"d": d})["d"] is d

    def test_exception_text_redacted_when_enabled(self):
        buf = io.StringIO()
        disseqt_logging.configure(level="debug", fmt="json", stream=buf, redact=True)
        try:
            log = disseqt_logging.stdlib_logger("disseqt_agentic_sdk.test_m20")
            try:
                raise RuntimeError("failed for bob@example.com")
            except RuntimeError:
                log.error("boom", exc_info=True)
        finally:
            disseqt_logging.disable()
        line = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert "bob@example.com" not in line["exception"]
        assert "[EMAIL]" in line["exception"]

    def test_exception_text_untouched_when_redaction_off(self):
        buf = io.StringIO()
        disseqt_logging.configure(level="debug", fmt="json", stream=buf, redact=False)
        try:
            log = disseqt_logging.stdlib_logger("disseqt_agentic_sdk.test_m20b")
            try:
                raise RuntimeError("failed for bob@example.com")
            except RuntimeError:
                log.error("boom", exc_info=True)
        finally:
            disseqt_logging.disable()
        line = json.loads(buf.getvalue().strip().splitlines()[-1])
        assert "bob@example.com" in line["exception"]


class TestSendTraceDoesNotDuplicateDeliveredSpans:
    """M14: send_trace must skip spans end() already delivered incrementally."""

    def _client(self):
        from unittest.mock import MagicMock

        from disseqt_agentic_sdk import DisseqtAgenticClient

        client = DisseqtAgenticClient.__new__(DisseqtAgenticClient)
        client.buffer = MagicMock()
        return client

    def test_delivered_spans_are_skipped(self):

        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.trace import DisseqtTrace

        client = self._client()
        trace = DisseqtTrace(name="t", project_id="p", client=client)
        with trace.start_span("delivered", SpanKind.AGENT_EXEC):
            pass
        assert client.buffer.add_span.call_count == 1
        pending = trace.start_span("pending", SpanKind.AGENT_EXEC)
        pending._client = None  # never ended/delivered incrementally
        client.send_trace(trace)
        sent = [s.name for c in client.buffer.add_spans.call_args_list for s in c.args[0]]
        assert sent == ["pending"]

    def test_nothing_sent_when_all_delivered(self):
        from disseqt_agentic_sdk.enums import SpanKind
        from disseqt_agentic_sdk.trace import DisseqtTrace

        client = self._client()
        trace = DisseqtTrace(name="t", project_id="p", client=client)
        with trace.start_span("only", SpanKind.AGENT_EXEC):
            pass
        client.send_trace(trace)
        client.buffer.add_spans.assert_not_called()
