"""``disseqt login`` / ``disseqt logout`` — pragmatic PAT-paste auth.

No new backend endpoints. Login prompts for a PAT, smoke-tests it against
the existing ``GET /api/v1/users/me/api-keys`` (cheapest authenticated
call), and stores it in ``~/.disseqt/config.json`` (``0600``). Logout
revokes the stored key server-side via ``DELETE /api/v1/users/me/api-keys/:id``
and clears the local file.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click
import requests

from .._version import sdk_identity_headers
from ..auth import (
    AuthConfigPermissionError,
)
from ..auth import (
    clear as auth_clear,
)
from ..auth import (
    load as auth_load,
)
from ..auth import (
    save as auth_save,
)
from ._common import _fail

# Auth-service endpoints live on the standard SDK gateway.
_DEFAULT_BASE_URL = "https://api.disseqt.ai/realtime-validations"
_API_KEYS_PATH = "/api/v1/users/me/api-keys"
# Public key-management page — printed for humans, never fetched.
_KEY_MGMT_URL = "https://app.disseqt.ai/settings/api-keys"
_KEY_PREFIX_LEN = 12
_TIMEOUT_S = 30


def _mask(api_key: str) -> str:
    """``sk_live_ab...`` — safe to print / log."""
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:8]}..."


def _auth_headers(api_key: str, project_id: str) -> dict[str, str]:
    return {
        "X-API-Key": api_key,
        "X-Project-Id": project_id,
        "Content-Type": "application/json",
        **sdk_identity_headers(),
    }


def _list_api_keys(base_url: str, api_key: str, project_id: str) -> requests.Response:
    """Smoke-test creds via the cheapest authenticated call."""
    url = f"{base_url.rstrip('/')}{_API_KEYS_PATH}"
    return requests.get(url, headers=_auth_headers(api_key, project_id), timeout=_TIMEOUT_S)


@click.command("login")
@click.option("--api-key", "api_key_flag", default=None, help="Skip prompt; use this key.")
@click.option(
    "--project-id", "project_id_flag", default=None, help="Skip prompt; use this project id."
)
@click.option(
    "--base-url",
    default=_DEFAULT_BASE_URL,
    show_default=True,
    help="Gateway base URL (rarely needed).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable status envelope.")
def login(
    api_key_flag: str | None,
    project_id_flag: str | None,
    base_url: str,
    as_json: bool,
) -> None:
    """Store a Disseqt API key + project id locally, after verifying them."""
    interactive = api_key_flag is None or project_id_flag is None
    if interactive and not sys.stdin.isatty():
        _fail(
            "non-interactive shell detected. For CI, set DISSEQT_API_KEY and "
            "DISSEQT_PROJECT_ID as env vars, or pass --api-key / --project-id."
        )

    if api_key_flag is None:
        click.echo(f"Create or copy an API key at: {_KEY_MGMT_URL}")
        api_key = click.prompt("Paste your API key", hide_input=True)
    else:
        api_key = api_key_flag
    project_id = project_id_flag or click.prompt("Enter your project ID")

    api_key = (api_key or "").strip()
    project_id = (project_id or "").strip()
    if not api_key or not project_id:
        _fail("api key and project id are both required")

    try:
        resp = _list_api_keys(base_url, api_key, project_id)
    except requests.RequestException as exc:
        # Never include the token in an error path.
        _fail(f"could not reach {base_url}: {type(exc).__name__}")

    if resp.status_code == 401 or resp.status_code == 403:
        _fail("invalid API key or project ID", code=2)
    if not resp.ok:
        _fail(f"unexpected HTTP {resp.status_code} verifying credentials", code=2)

    auth_save({"api_key": api_key, "project_id": project_id, "base_url": base_url})

    if as_json:
        click.echo(
            json.dumps(
                {
                    "status": "logged_in",
                    "api_key_prefix": _mask(api_key),
                    "project_id": project_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        click.echo(f"logged in as project {project_id} (key {_mask(api_key)})")


@click.command("logout")
@click.option(
    "--local-only",
    is_flag=True,
    help="Only clear ~/.disseqt/config.json; do not revoke the key server-side.",
)
def logout(local_only: bool) -> None:
    """Revoke the stored API key server-side, then clear the local config."""
    try:
        stored = auth_load()
    except AuthConfigPermissionError as exc:
        _fail(str(exc))
    if stored is None:
        click.echo("not logged in; nothing to do")
        return

    api_key = stored.get("api_key") or ""
    project_id = stored.get("project_id") or ""
    base_url = stored.get("base_url") or _DEFAULT_BASE_URL

    if local_only or not (api_key and project_id):
        auth_clear()
        click.echo("local config cleared")
        return

    revoked = _try_revoke(base_url, api_key, project_id)
    auth_clear()
    if revoked:
        click.echo("logged out (server-side key revoked, local config cleared)")
    else:
        click.echo(
            "local config cleared; server-side revocation failed "
            "(key may already be revoked — verify in the dashboard)"
        )


def _try_revoke(base_url: str, api_key: str, project_id: str) -> bool:
    """Best-effort revoke of the currently-stored key. Returns True on success.

    Failures are non-fatal: we still clear the local config so the CLI
    stops using the credential.
    """
    try:
        resp = _list_api_keys(base_url, api_key, project_id)
    except requests.RequestException:
        return False
    if not resp.ok:
        return False
    try:
        payload = resp.json()
    except ValueError:
        return False
    entries = _extract_key_list(payload)
    prefix = api_key[:_KEY_PREFIX_LEN]
    match_id = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_prefix = str(entry.get("key_prefix") or entry.get("prefix") or "")
        if entry_prefix and prefix.startswith(entry_prefix):
            match_id = entry.get("id") or entry.get("api_key_id")
            break
    if not match_id:
        return False
    delete_url = f"{base_url.rstrip('/')}{_API_KEYS_PATH}/{match_id}"
    try:
        del_resp = requests.delete(
            delete_url,
            headers=_auth_headers(api_key, project_id),
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException:
        return False
    return del_resp.ok


def _extract_key_list(payload: Any) -> list[Any]:
    """The list endpoint may return ``[...]`` or ``{"data": [...]}``."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []
