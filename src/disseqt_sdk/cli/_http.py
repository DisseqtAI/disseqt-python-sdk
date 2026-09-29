"""Authenticated HTTP wrapper for CLI subcommands hitting the dataset gateway.

Sends only the user's project key (``X-API-Key`` + ``X-Project-Id``); the
gateway validates it and injects the internal identity headers itself.
Unwraps the ``{"status": "success", "data": ...}`` envelope and raises
:class:`APIError` (click exits 1) on HTTP / network / error-envelope
failures. Missing credentials are a config error (``_fail`` → exit 2).
"""

from __future__ import annotations

import json
import os
from typing import Any

import click
import requests

from .._version import sdk_identity_headers
from ..api_client import HTTPError, unwrap_envelope
from ._common import ENV_API_KEY, ENV_BASE_URL, ENV_PROJECT_ID, _fail

DEFAULT_BASE_URL = "https://api.disseqt.ai/dataset"
DEFAULT_TIMEOUT_SECS = 60


class APIError(click.ClickException):
    """HTTP / network failure. click prints it at the command boundary."""

    exit_code = 1

    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


def base_url(env_key: str = ENV_BASE_URL, default: str = DEFAULT_BASE_URL) -> str:
    return (os.environ.get(env_key) or default).rstrip("/")


def _headers() -> dict[str, str]:
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    if not project_id or not api_key:
        _fail(f"set {ENV_PROJECT_ID} and {ENV_API_KEY} in the environment")
    return {
        "X-API-Key": api_key,
        "X-Project-Id": project_id,
        "Content-Type": "application/json",
        **sdk_identity_headers(),
    }


def request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    base: str | None = None,
) -> Any:
    """Authenticated request; returns the unwrapped ``data`` (or raw text for non-JSON)."""
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
