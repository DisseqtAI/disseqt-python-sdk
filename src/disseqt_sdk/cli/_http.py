"""Minimal authenticated HTTP wrapper for CLI subcommands that hit
red-team / policy-management endpoints not yet surfaced on
:class:`~disseqt_sdk.client.Client`.

Reads the same env vars as :mod:`._common` — see ``ENV_*`` constants.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from .._version import sdk_identity_headers
from ._common import ENV_API_KEY, ENV_PROJECT_ID, _fail

DEFAULT_TIMEOUT_SECS = 60


def _base_from(env_key: str, default: str) -> str:
    return (os.environ.get(env_key) or default).rstrip("/")


def _headers() -> dict[str, str]:
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    if not project_id or not api_key:
        _fail(f"set {ENV_PROJECT_ID} and {ENV_API_KEY} in the environment")
    return {
        "X-Service-API-Key": api_key,
        "X-API-Key": api_key,
        "X-Project-Id": project_id,
        "Content-Type": "application/json",
        **sdk_identity_headers(),
    }


def _org_project_headers() -> dict[str, str]:
    """Extra headers the policy-management BFF injects downstream."""
    hdrs: dict[str, str] = {}
    org = os.environ.get("DISSEQT_ORGANIZATION_ID")
    proj = os.environ.get("DISSEQT_PROJECT_ID")
    if org:
        hdrs["X-Organization-ID"] = org
    if proj:
        hdrs["X-Project-ID"] = proj
    return hdrs


def request(
    method: str,
    base_env: str,
    default_base: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Any:
    """Make an authenticated HTTP request and return parsed JSON."""
    url = f"{_base_from(base_env, default_base)}{path}"
    headers = _headers()
    headers.update(_org_project_headers())
    if extra_headers:
        headers.update(extra_headers)
    # requests picks the right Content-Type for multipart when `files` is set.
    if files is not None:
        headers.pop("Content-Type", None)
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            json=json_body,
            params=params,
            files=files,
            timeout=DEFAULT_TIMEOUT_SECS,
        )
    except requests.RequestException as exc:
        _fail(f"network error calling {url}: {exc}")
    if not resp.ok:
        _fail(f"HTTP {resp.status_code} from {url}: {resp.text[:512]}")
    if not resp.text:
        return None
    try:
        return resp.json()
    except json.JSONDecodeError:
        return resp.text
