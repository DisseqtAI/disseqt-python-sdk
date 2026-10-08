"""M10: validator Client reuses one Session and retries only HTTP 429 (bounded)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from disseqt_sdk import Client, HTTPError
from disseqt_sdk.client import _MAX_RETRY_AFTER_S


class _Server:
    """Local HTTP server replaying a scripted list of (status, headers, body)."""

    def __init__(self, script):
        self.script = list(script)
        self.hits = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                i = min(outer.hits, len(outer.script) - 1)
                outer.hits += 1
                status, headers, body = outer.script[i]
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


OK = (200, {}, '{"ok": true}')


def _client(url, **kw):
    return Client(project_id="p", api_key="k", base_url=url, timeout=5, **kw)


def _post(client):
    return client._session.post(client.base_url + "/x", json={}, timeout=5)


@pytest.fixture
def serve():
    servers = []

    def make(script):
        s = _Server(script)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.stop()


class TestRetryPolicy:
    def test_429_then_200_is_retried(self, serve):
        srv = serve([(429, {"Retry-After": "0"}, "{}"), OK])
        with _client(srv.url) as c:
            assert _post(c).status_code == 200
        assert srv.hits == 2

    def test_429_exhausts_and_returns_the_429(self, serve):
        srv = serve([(429, {"Retry-After": "0"}, "{}")])
        with _client(srv.url, max_retries=2) as c:
            assert _post(c).status_code == 429
        assert srv.hits == 3  # 1 try + 2 retries

    def test_retry_after_beyond_cap_is_not_waited_out(self, serve):
        srv = serve([(429, {"Retry-After": str(int(_MAX_RETRY_AFTER_S) + 60)}, "{}"), OK])
        with _client(srv.url) as c:
            assert _post(c).status_code == 429
        assert srv.hits == 1

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 401, 400])
    def test_other_statuses_are_never_retried(self, serve, status):
        srv = serve([(status, {}, "{}"), OK])
        with _client(srv.url) as c:
            assert _post(c).status_code == status
        assert srv.hits == 1

    def test_max_retries_zero_disables_retry(self, serve):
        srv = serve([(429, {"Retry-After": "0"}, "{}"), OK])
        with _client(srv.url, max_retries=0) as c:
            assert _post(c).status_code == 429
        assert srv.hits == 1

    def test_connection_refused_is_not_retried(self):
        # Nothing listening: fail fast, single attempt, SDK error (status 0).
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        c = _client(f"http://127.0.0.1:{port}")
        with pytest.raises(Exception) as ei:
            _post(c)
        assert ei.type.__name__ in {"ConnectionError", "ConnectTimeout"}

    @pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
    def test_invalid_max_retries_rejected(self, bad):
        with pytest.raises(ValueError):
            Client(project_id="p", api_key="k", max_retries=bad)


class TestSessionLifecycle:
    def test_one_session_is_reused_across_calls(self, serve):
        srv = serve([OK])
        c = _client(srv.url)
        s1 = c._session
        _post(c)
        _post(c)
        assert c._session is s1
        c.close()

    def test_close_is_idempotent_and_context_manager_closes(self, serve):
        srv = serve([OK])
        with _client(srv.url) as c:
            _post(c)
        c.close()
        c.close()

    def test_validate_raises_httperror_on_final_429(self, serve):
        srv = serve([(429, {"Retry-After": "0"}, '{"error":"slow down"}')])
        c = _client(srv.url, max_retries=1, realtime_policy_base_url=srv.url)
        with pytest.raises(HTTPError) as ei:
            c._post_policy_evaluate("pol", {"prompt": "x"}, "app")
        assert ei.value.status_code == 429
        assert srv.hits == 2
