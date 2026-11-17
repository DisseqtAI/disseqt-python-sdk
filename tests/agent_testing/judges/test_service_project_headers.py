"""The service transport sends project credentials like disseqt_sdk, never mixed with bearer."""

import email.message
import io
import urllib.request
import urllib.response
from typing import Any

import pytest

from disseqt_agent_testing.judges.models.service import HttpTransport, ServiceModel

_seen: dict[str, str] = {}


@pytest.fixture()
def server(monkeypatch: pytest.MonkeyPatch) -> str:
    """No socket is opened: the HTTP(S) handlers' open is replaced and records the headers."""

    def fake_open(handler: Any, request: urllib.request.Request) -> Any:
        _seen.clear()
        _seen.update({k.lower(): v for k, v in request.header_items()})
        resp = urllib.response.addinfourl(
            io.BytesIO(b"{}"), email.message.Message(), request.full_url, 200
        )
        resp.msg = "OK"  # type: ignore[attr-defined]
        return resp

    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", fake_open)
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_open)
    return "https://svc.invalid"


def _send(base: str, **kw: Any) -> dict[str, str]:
    HttpTransport(base, **kw)._send("/v1/judge/complete", {"a": 1}, 5.0)
    return dict(_seen)


def test_project_mode_sends_project_headers_not_bearer(server: str) -> None:
    seen = _send(server, api_key="k1", project_id="p1")
    assert seen["x-project-id"] == "p1" and seen["x-api-key"] == "k1"
    assert "authorization" not in seen


def test_legacy_key_still_sent_as_bearer(server: str) -> None:
    seen = _send(server, api_key="svc")
    assert seen["authorization"] == "Bearer svc"
    assert "x-project-id" not in seen and "x-api-key" not in seen


def test_project_without_key_sends_no_secret(server: str) -> None:
    seen = _send(server, project_id="p1")
    assert seen["x-project-id"] == "p1"
    assert "x-api-key" not in seen and "authorization" not in seen


def test_env_project_mode_uses_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISSEQT_JUDGE_SERVICE_URL", "https://svc.invalid")
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "p9")
    monkeypatch.setenv("DISSEQT_API_KEY", "k9")
    monkeypatch.setenv("DISSEQT_JUDGE_SERVICE_KEY", "legacy-ignored")
    t = ServiceModel("m")._transport
    assert isinstance(t, HttpTransport) and t._project == "p9" and t._key == "k9"
