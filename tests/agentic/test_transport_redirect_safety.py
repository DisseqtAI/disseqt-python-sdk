"""
Real-socket reproduction of the cross-host redirect credential leak.

No mocks: two real TCP listeners on 127.0.0.1 and localhost (different
hostname strings, so requests classifies a redirect between them as
cross-host), and the transport's own real ``requests.Session``. This
exists specifically because a mocked ``session.post`` assertion cannot
prove what actually crosses the wire on a redirect -- only a real
socket can.

Bug: ``requests``' default ``allow_redirects=True`` follows a
301/302/303 automatically. ``rebuild_auth`` strips ONLY the
``Authorization`` header on a cross-host redirect -- custom headers
like ``X-Api-Key`` are not stripped and follow to
the new host in cleartext, even though the JSON body (and its
``resource.attributes["api.key"]`` copy) is dropped for that same
redirect class (POST -> GET conversion). Pre-fix, that redirect
scenario leaked the live API key to whatever host answered the
redirect. Fix: ``allow_redirects=False`` on the trace-ingestion POST
(transport/http.py) -- there is no product reason this internal
machine-to-machine request should ever follow a redirect, so the
correct, complete fix is that the second host is never contacted at
all.
"""

from __future__ import annotations

import socket
import threading
import time
from uuid import uuid4

import pytest

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.transport.http import HTTPTransport


def _make_span() -> EnrichedSpan:
    return EnrichedSpan(
        trace_id=str(uuid4()),
        span_id=str(uuid4()),
        name="probe",
        kind="MODEL_EXEC",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_000_000_000,
        duration_ns=1_000_000_000,
        status_code="OK",
        service_name="probe-service",
        realtime_policy_id="",
    )


def _bind_ephemeral(host: str) -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, 0))  # OS-assigned free port -- avoids CI port collisions
    srv.listen(1)
    return srv


def _read_request(conn: socket.socket) -> bytes:
    conn.settimeout(5)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    header_blob, _, rest = data.partition(b"\r\n\r\n")
    content_length = 0
    for line in header_blob.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            content_length = int(line.split(b":", 1)[1].strip())
    body = rest
    while len(body) < content_length:
        chunk = conn.recv(4096)
        if not chunk:
            break
        body += chunk
    return header_blob


@pytest.fixture
def redirect_pair():
    """
    Server A (127.0.0.1) answers the real POST with a 302 to Server B
    (localhost) -- a different hostname string, so requests treats this
    as cross-host. Returns (port_a, port_b, captured) where captured is
    populated with whatever each server actually received.
    """
    srv_a = _bind_ephemeral("127.0.0.1")
    srv_b = _bind_ephemeral("127.0.0.1")  # "localhost" resolves here too
    port_a = srv_a.getsockname()[1]
    port_b = srv_b.getsockname()[1]
    captured: dict[str, bytes | None] = {"A": None, "B": None}

    def run_a():
        srv_a.settimeout(5)
        try:
            conn, _ = srv_a.accept()
        except TimeoutError:
            return
        captured["A"] = _read_request(conn)
        resp = (
            f"HTTP/1.1 302 Found\r\n"
            f"Location: http://localhost:{port_b}/redirected\r\n"
            f"Content-Length: 0\r\nConnection: close\r\n\r\n"
        ).encode()
        conn.sendall(resp)
        conn.close()
        srv_a.close()

    def run_b():
        srv_b.settimeout(1.5)  # short: we're proving this DOESN'T get hit
        try:
            conn, _ = srv_b.accept()
        except TimeoutError:
            return
        captured["B"] = _read_request(conn)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
        conn.close()
        srv_b.close()

    ta = threading.Thread(target=run_a, daemon=True)
    tb = threading.Thread(target=run_b, daemon=True)
    ta.start()
    tb.start()
    time.sleep(0.2)

    yield port_a, port_b, captured, ta, tb

    srv_a.close()
    srv_b.close()


class TestRedirectDoesNotLeakCredential:
    def test_redirect_target_is_never_contacted(self, redirect_pair):
        """
        The complete fix: with allow_redirects=False, requests never
        issues a second request at all, so the redirect target receives
        nothing -- not the header, not the body, not a connection.
        """
        port_a, port_b, captured, ta, tb = redirect_pair

        transport = HTTPTransport(
            endpoint=f"http://127.0.0.1:{port_a}/v1/traces",
            api_key="FAKE-not-a-real-secret-redirect-test",
            application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558",
        )
        transport.send_spans([_make_span()])

        ta.join(timeout=5)
        tb.join(timeout=3)

        assert captured["A"] is not None, "transport never even reached Server A"
        assert captured["B"] is None, (
            "Server B (the redirect target, a different host) received a "
            "connection -- the API key header would have reached it. "
            "allow_redirects=False on the session.post call should have "
            "prevented requests from following the redirect at all."
        )
