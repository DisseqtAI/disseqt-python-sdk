"""The ``service:`` provider: the judge model is reached through an API of another service.

Wire contract (v1), the one the real service must implement::

    POST {base_url}/v1/judge/complete
    request : {"model", "messages", "output_schema", "schema_name", "params", "sample_index",
               "kind", "order"}
    success : 200 {"text", "stop_reason", "usage": {"input_tokens", "output_tokens"},
                   "cost_usd", "model_id", "system_fingerprint"}
    failure : {status} with {"error": {"message", "retry_after_s"}}; 429/5xx are retried,
              401/403 open the circuit, other 4xx are bad requests.

Until the real service exists, ``FakeServiceTransport`` implements this contract in-process so
the whole pipeline (retry, limits, cache, pricing) is testable. Nothing falls back to the fake
silently: a ``service:`` model without a transport or ``DISSEQT_JUDGE_SERVICE_URL`` fails closed.
"""

import asyncio
import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ..codes import JudgeConfigError
from .base import (
    JudgeRequest,
    JudgeResponse,
    Price,
    ProviderAuth,
    ProviderBadRequest,
    ProviderTransient,
)
from .registry import register_provider

ENDPOINT = "/v1/judge/complete"
URL_ENV = "DISSEQT_JUDGE_SERVICE_URL"
KEY_ENV = "DISSEQT_JUDGE_SERVICE_KEY"
PROJECT_ENV = "DISSEQT_PROJECT_ID"
API_KEY_ENV = "DISSEQT_API_KEY"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_RETRY_AFTER_S = 60.0
_LOOPBACK = ("localhost", "127.0.0.1", "::1")


def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` (bools, strings of garbage, NaN, inf, lists)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _clean_usage(usage: Any) -> dict[str, int] | None:
    """Non-negative integer token counts; anything unparseable counts as 0 (never raises)."""
    if not isinstance(usage, Mapping):
        return None
    out: dict[str, int] = {}
    for key, raw in usage.items():
        number = _finite(raw)
        out[str(key)] = int(number) if number is not None and number >= 0 else 0
    return out


def _clean_retry_after(value: Any) -> float | None:
    """Seconds clamped to [0, MAX_RETRY_AFTER_S]; unparseable or NaN means "no hint"."""
    number = _finite(value)
    if number is None:
        return None
    return min(max(number, 0.0), MAX_RETRY_AFTER_S)


def normalize_base_url(url: str | None) -> str | None:
    """The one place a base URL is normalised (trailing slashes), for every provider path."""
    if not url:
        return None
    text = str(url).strip().rstrip("/")
    return text or None


@runtime_checkable
class ServiceTransport(Protocol):
    """Sends one JSON request and returns ``(http_status, json_body)``. Must not raise on HTTP
    errors; connection failures raise ``ProviderTransient``."""

    base_url: str | None

    async def post(
        self, path: str, payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, Any]]: ...


def request_payload(model: str, req: JudgeRequest) -> dict[str, Any]:
    return {
        "model": model, "messages": [dict(m) for m in req.messages],
        "output_schema": dict(req.output_schema), "schema_name": req.schema_name,
        "params": dict(req.params), "sample_index": req.sample_index, "kind": req.kind,
        "order": req.order,
    }  # fmt: skip


class ServiceModel:
    """``JudgeModel`` over a ``ServiceTransport``."""

    provider = "service"

    def __init__(
        self,
        model_id: str,
        *,
        transport: ServiceTransport | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        price: Price | None = None,
        **_unused: Any,
    ):
        self.id = model_id
        if transport is None:
            url = base_url or os.environ.get(URL_ENV)
            if not url:
                raise JudgeConfigError(
                    "service_not_configured",
                    f"service:{model_id} needs a transport=, base_url= or ${URL_ENV}",
                )
            project = project_id or os.environ.get(PROJECT_ENV)
            # Project mode (same credentials as disseqt_sdk): X-Project-ID + X-API-Key.
            # Without a project the legacy service key is sent as a bearer token.
            key = api_key or os.environ.get(API_KEY_ENV if project else KEY_ENV)
            transport = HttpTransport(url, key, project_id=project)
        self._transport = transport
        self.base_url = normalize_base_url(transport.base_url)
        self.networked = not isinstance(transport, FakeServiceTransport)
        self._price = price

    def price(self) -> Price | None:
        return self._price

    async def complete(self, req: JudgeRequest) -> JudgeResponse:
        status, body = await self._transport.post(
            ENDPOINT, request_payload(self.id, req), req.attempt_timeout_s
        )
        if status != 200:
            raw_err = body.get("error")
            err: Mapping[str, Any] = raw_err if isinstance(raw_err, Mapping) else {}
            text = str(err.get("message") or f"HTTP {status}")
            if status == 429 or status >= 500:
                raise ProviderTransient(
                    text, status=status, retry_after_s=_clean_retry_after(err.get("retry_after_s"))
                )
            if status in (401, 403):
                raise ProviderAuth(text)
            raise ProviderBadRequest(text)
        usage = _clean_usage(body.get("usage"))
        cost = _finite(body.get("cost_usd"))
        if cost is None or cost < 0:
            cost = (
                self._price.cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                if self._price and usage
                else 0.0
            )
        return JudgeResponse(
            str(body.get("text", "")), None, str(body.get("stop_reason", "stop")), usage,
            cost, str(body.get("model_id") or self.id), body.get("system_fingerprint"),
        )  # fmt: skip

    def secrets(self) -> tuple[str, ...]:
        getter = getattr(self._transport, "secrets", None)
        return tuple(getter()) if callable(getter) else ()


def configured_base_url(options: Mapping[str, Any]) -> str | None:
    """The base URL a ``service:`` model will use, known WITHOUT building it (replay must not
    construct a provider). Mirrors ``ServiceModel.__init__`` so the judge identity is the same
    before and after the lazily built model exists."""
    transport = options.get("transport")
    if transport is not None:
        return normalize_base_url(getattr(transport, "base_url", None))
    return normalize_base_url(options.get("base_url") or os.environ.get(URL_ENV))


def _clean_secret(value: str | None, what: str) -> str | None:
    """Strip, then reject control characters. The message never echoes the value."""
    if value is None:
        return None
    text = str(value).strip()
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise JudgeConfigError("service_credentials", f"{what} contains control characters")
    return text or None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follows a redirect: credentials must not travel to another URL. The 3xx surfaces as
    an HTTP status (a bad request for the model)."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class HttpTransport:
    """stdlib HTTP transport for the real service (blocking call run in a worker thread).
    Written against the contract above; not yet exercised against a live service."""

    def __init__(self, base_url: str, api_key: str | None = None, *, project_id: str | None = None):
        self._key = _clean_secret(api_key, "the API key")
        self._project = _clean_secret(project_id, "the project id")
        url = normalize_base_url(base_url) or ""
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise JudgeConfigError("service_url", "the service URL must be http(s)://host[:port]")
        if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK:
            raise JudgeConfigError(
                "service_url",
                "prompts and credentials are only sent over https "
                "(http is allowed for localhost only)",
            )
        self.base_url: str | None = url
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return f"HttpTransport(base_url={self.base_url!r})"

    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self._key, self._project) if s)

    def _send(self, path: str, payload: Mapping[str, Any], timeout_s: float) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json"}
        if self._project:
            headers["X-Project-ID"] = self._project
            if self._key:
                headers["X-API-Key"] = self._key
        elif self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        request = urllib.request.Request(  # noqa: S310  (scheme checked in __init__)
            f"{self.base_url}{path}", json.dumps(payload).encode(), headers, method="POST"
        )
        try:
            with self._opener.open(request, timeout=timeout_s) as resp:  # noqa: S310
                return resp.status, self._parse(resp.read(MAX_BODY_BYTES + 1))
        except urllib.error.HTTPError as exc:
            try:
                body = self._parse(exc.read(MAX_BODY_BYTES + 1))
            except ProviderBadRequest:
                body = {}
            if exc.code in (429, 503):
                body = self._with_retry_after(body, exc.headers.get("Retry-After"))
            return exc.code, body
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderTransient(f"connection failed: {exc}") from exc
        except ValueError:  # e.g. an invalid header value: never echo what it contained
            raise ProviderBadRequest("the service request could not be sent") from None

    @staticmethod
    def _parse(raw: bytes) -> Any:
        if len(raw) > MAX_BODY_BYTES:
            raise ProviderBadRequest("the service response exceeds the 4 MiB limit")
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            raise ProviderBadRequest("the service response is not valid JSON") from None

    @staticmethod
    def _with_retry_after(body: Any, header: str | None) -> Any:
        seconds = _clean_retry_after(header) if header else None
        if seconds is None or not isinstance(body, Mapping):
            return body
        raw_err = body.get("error")
        err = dict(raw_err) if isinstance(raw_err, Mapping) else {}
        if err.get("retry_after_s") is None:
            err["retry_after_s"] = seconds
        return {**body, "error": err}

    async def post(
        self, path: str, payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, Any]]:
        status, body = await asyncio.to_thread(self._send, path, payload, timeout_s)
        return status, body if isinstance(body, Mapping) else {}


class FakeServiceTransport:
    """An in-process stand-in for the service. ``handler(payload) -> body | (status, body)``;
    ``script`` (a list) is consumed in order instead. Records every payload in ``.requests``."""

    base_url: str | None = "http://fake-judge-service.invalid"

    def __init__(self, handler: Any = None, *, script: Any = None, model_id: str | None = None):
        self._handler = handler
        self._script = list(script) if script is not None else None
        self._model_id = model_id
        self.requests: list[dict[str, Any]] = []

    async def post(
        self, path: str, payload: Mapping[str, Any], timeout_s: float
    ) -> tuple[int, Mapping[str, Any]]:
        self.requests.append(dict(payload))
        if path != ENDPOINT:
            return 404, {"error": {"message": f"no route {path}"}}
        if self._script is not None:
            if not self._script:
                raise AssertionError("FakeServiceTransport script exhausted")
            item = self._script.pop(0)
        elif self._handler is not None:
            item = self._handler(payload)
        else:
            raise AssertionError("FakeServiceTransport needs a handler or a script")
        if isinstance(item, tuple):
            return int(item[0]), item[1]
        body = dict(item)
        if "text" not in body:  # a bare verdict dict: wrap it like the service would
            body = {"text": json.dumps(body), "stop_reason": "stop"}
        body.setdefault("model_id", self._model_id or str(payload["model"]))
        return 200, body


def _factory(model_id: str, **options: Any) -> ServiceModel:
    return ServiceModel(model_id, **options)


register_provider("service", _factory, override=True)
