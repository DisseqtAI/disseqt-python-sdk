"""Authenticated HTTP wrapper for CLI subcommands hitting the dataset gateway.

Sends the user's key (``X-API-Key``, plus ``X-Project-Id`` only when set); the
gateway validates it and injects the internal identity headers itself.
Unwraps the ``{"status": "success", "data": ...}`` envelope and raises
:class:`APIError` (click exits 1) on HTTP / network / error-envelope
failures. Credentials come from the env or ``disseqt login`` (see
:func:`._common.require_credentials`); missing ones are a config error (exit 2).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import click
import requests

from .._version import sdk_identity_headers
from ..api_client import (
    DEFAULT_POLICIES_BASE_URL,
    ENV_POLICIES_BASE_URL,
    HTTPError,
    unwrap_envelope,
)
from ..auth import DEFAULT_BASE_URL, AuthConfigPermissionError, resolve_base_url
from ..models.redteam import TERMINAL_RUN_STATES
from ._common import ENV_BASE_URL, _fail, require_credentials

DEFAULT_TIMEOUT_SECS = 60


class APIError(click.ClickException):
    """HTTP / network failure. click prints it at the command boundary."""

    exit_code = 1

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


def base_url(env_key: str = ENV_BASE_URL, default: str = DEFAULT_BASE_URL) -> str:
    """Dataset gateway: env, then the ``disseqt login --base-url`` config, then default."""
    if env_key != ENV_BASE_URL:
        return (os.environ.get(env_key) or default).rstrip("/")
    try:
        return resolve_base_url()
    except AuthConfigPermissionError as exc:
        _fail(str(exc))


def policies_base_url() -> str:
    return base_url(ENV_POLICIES_BASE_URL, DEFAULT_POLICIES_BASE_URL)


def _headers() -> dict[str, str]:
    project_id, api_key = require_credentials()
    headers = {"X-API-Key": api_key, "Content-Type": "application/json", **sdk_identity_headers()}
    if project_id:
        headers["X-Project-Id"] = project_id
    return headers


def request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    base: str | None = None,
    raw: bool = False,
) -> Any:
    """Authenticated request; returns the unwrapped ``data``, raw text for non-JSON,
    or the body text untouched with ``raw=True`` (CSV exports)."""
    url = f"{base or base_url()}{path}"
    headers = _headers()
    if extra_headers:
        headers.update(extra_headers)
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
    except requests.RequestException as exc:
        raise APIError(f"network error calling {url}: {exc}") from exc
    if not resp.ok:
        raise APIError(f"HTTP {resp.status_code} from {url}: {resp.text[:512]}", resp.status_code)
    if raw:
        return resp.text
    if not resp.text:
        return None
    try:
        raw = resp.json()
    except json.JSONDecodeError:
        return resp.text
    try:
        return unwrap_envelope(raw, resp.status_code)
    except HTTPError as exc:
        raise APIError(str(exc), exc.status_code) from exc


def _status(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    # MR jobs nest it: GET /mr-jailbreak/jobs/:id -> {job: {status}, interactions}.
    job = payload.get("job")
    nested = job.get("status") if isinstance(job, dict) else None
    return str(payload.get("status") or nested or "")


def wait_terminal(
    path: str,
    poll_interval: float,
    max_wait: float,
    on_abort: Callable[[], Any] | None = None,
    hint: str = "",
    status_of: Callable[[Any], str] = _status,
) -> Any:
    """Poll ``path`` until its status is terminal. On Ctrl-C or timeout, cancel
    the run via ``on_abort`` (best effort), print ``hint``, exit 130 / 1."""
    deadline = time.monotonic() + max_wait
    try:
        while time.monotonic() < deadline:
            payload = request("GET", path)
            if status_of(payload).lower() in TERMINAL_RUN_STATES:
                return payload
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        _abort(on_abort, "interrupted", hint)
        raise SystemExit(130) from None
    _abort(on_abort, f"not finished within {max_wait}s", hint)
    raise click.ClickException(f"{path} did not finish within {max_wait}s")


def _abort(on_abort: Callable[[], Any] | None, why: str, hint: str) -> None:
    if on_abort is None:
        click.secho(f"{why}. {hint}", fg="yellow", err=True)
        return
    try:
        on_abort()
    except APIError as exc:
        click.secho(f"warning: cancel failed: {exc}", fg="yellow", err=True)
    click.secho(f"{why}; cancel requested. {hint}", fg="yellow", err=True)
