"""M11: policies are evaluated in parallel (bounded), in order, under one deadline."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from disseqt_sdk import Client, any_blocking, is_error
from disseqt_sdk.client import _MAX_PARALLEL_POLICIES, HTTPError
from disseqt_sdk.models.input_validation import InputValidationRequest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _PolicyServer:
    """Threaded server: per-policy delay/status, records peak concurrency."""

    def __init__(self, delay=0.0, per_policy=None):
        self.delay = delay
        self.per_policy = per_policy or {}
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.hits = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                pid = self.path.split("/policies/")[1].split("/")[0]
                with outer.lock:
                    outer.active += 1
                    outer.peak = max(outer.peak, outer.active)
                    outer.hits.append(pid)
                try:
                    cfg = outer.per_policy.get(pid, {})
                    time.sleep(cfg.get("delay", outer.delay))
                    status = cfg.get("status", 200)
                    body = json.dumps(
                        {"status": "success", "data": {"policy_id": pid, "decision": "PASS"}}
                        if status == 200
                        else {"error": "x"}
                    ).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with outer.lock:
                        outer.active -= 1

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def serve():
    made = []

    def make(**kw):
        s = _PolicyServer(**kw)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop()


def _client(srv, timeout=10):
    return Client(
        project_id="p",
        api_key="k",
        base_url=srv.url,
        realtime_policy_base_url=srv.url,
        application_name="app",
        timeout=timeout,
    )


def _req():
    return InputValidationRequest(prompt="hello")


def _ids(n):
    return [f"pol-{i}" for i in range(n)]


class TestParallelPolicies:
    def test_results_keep_policy_order(self, serve):
        # Later policies finish first; output order must still match input.
        ids = _ids(4)
        srv = serve(per_policy={ids[0]: {"delay": 0.3}, ids[1]: {"delay": 0.2}})
        with _client(srv) as c:
            out = c.validate(_req(), policies=ids)
        assert [e["data"]["policy_id"] for e in out["policies"]] == ids

    def test_runs_in_parallel_not_sequentially(self, serve):
        srv = serve(delay=0.3)
        with _client(srv) as c:
            t0 = time.monotonic()
            c.validate(_req(), policies=_ids(4))
            took = time.monotonic() - t0
        assert srv.peak >= 2
        assert took < 0.3 * 4 * 0.8  # clearly faster than sequential (1.2s)

    def test_concurrency_is_capped(self, serve):
        srv = serve(delay=0.2)
        with _client(srv) as c:
            out = c.validate(_req(), policies=_ids(10))
        assert len(out["policies"]) == 10
        assert srv.peak <= _MAX_PARALLEL_POLICIES

    def test_single_policy_does_not_need_threads(self, serve):
        srv = serve()
        with _client(srv) as c:
            out = c.validate(_req(), policies=["only"])
        assert out["policies"][0]["data"]["policy_id"] == "only"

    def test_partial_failure_keeps_other_results_in_order(self, serve):
        ids = _ids(3)
        srv = serve(per_policy={ids[1]: {"status": 500}})
        # Default fail-closed: ANY policy failure raises so the
        # documented ``except HTTPError: return False`` pattern keeps
        # working. The opt-in ``partial_ok=True`` path still returns
        # earlier envelopes with error entries for the failed ones —
        # this test exercises that opt-in branch.
        with _client(srv) as c:
            out = c.validate(_req(), policies=ids, partial_ok=True)
        assert out["policies"][0]["data"]["policy_id"] == ids[0]
        assert is_error(out["policies"][1]) and out["policies"][1]["policy_id"] == ids[1]
        assert out["policies"][2]["data"]["policy_id"] == ids[2]
        assert any_blocking(out) is True  # errored policy fails closed

    def test_all_failing_still_raises(self, serve):
        ids = _ids(3)
        srv = serve(per_policy={i: {"status": 500} for i in ids})
        with _client(srv) as c, pytest.raises(HTTPError) as ei:
            c.validate(_req(), policies=ids)
        assert ei.value.status_code == 500
        assert len(ei.value.partial_result["policies"]) == 3

    def test_overall_deadline_bounds_total_wait(self, serve):
        # 5 policies, cap 4. First 4 return instantly; the 5th stalls
        # for 3s. Overall deadline is now
        # timeout * ceil(5/4) = 1s * 2 = 2s (TP-2128 round-3 PR #39
        # review: per-slot deadline), so the 5th gets at most ~1s of
        # its own budget inside the second slot before the overall
        # deadline forces an error-entry. First 4 succeed.
        ids = _ids(5)
        srv = serve(per_policy={ids[4]: {"delay": 3.0}})
        with _client(srv, timeout=1) as c:
            t0 = time.monotonic()
            out = c.validate(_req(), policies=ids, partial_ok=True)
            took = time.monotonic() - t0
        # 2s deadline + a little jitter — must be well under the
        # delayed policy's own 3s delay.
        assert took < 2.6, f"total wait {took:.2f}s exceeded the 2s per-slot deadline"
        assert [is_error(e) for e in out["policies"]] == [False] * 4 + [True]
        late = out["policies"][4]
        assert late["policy_id"] == ids[4]
        # The 5th hits either its own per-request read timeout (1s < 3s
        # delay) OR the overall deadline (2s) — either way we end up
        # with an error entry, which is the behaviour we want to pin.
        assert late["error"]["status_code"] == 0
        assert any_blocking(out) is True
