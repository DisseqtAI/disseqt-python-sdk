"""``disseqt login`` / ``disseqt logout`` — pragmatic API-key-paste auth.

Login prompts for a project API key + project id, smoke-tests them with the
cheapest authenticated dataset call (``GET /api/v1/testing/attack-techniques``)
and stores them in ``~/.disseqt/config.json`` (``0600``). Logout only clears
that file — the gateway exposes no key-revocation route for API-key callers.
"""

from __future__ import annotations

import json
import sys

import click
import requests

from .._version import sdk_identity_headers
from ..auth import AuthConfigPermissionError
from ..auth import clear as auth_clear
from ..auth import load as auth_load
from ..auth import save as auth_save
from . import _http
from ._common import _fail

_SMOKE_PATH = "/api/v1/testing/attack-techniques"
# Public key-management page — printed for humans, never fetched.
_KEY_MGMT_URL = "https://app.disseqt.ai/settings/api-keys"
_TIMEOUT_S = 30


def _mask(api_key: str) -> str:
    """``sk_live_ab...`` — safe to print / log."""
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:8]}..."


def _smoke_test(base_url: str, api_key: str, project_id: str) -> requests.Response:
    headers = {
        "X-API-Key": api_key,
        "X-Project-Id": project_id,
        **sdk_identity_headers(),
    }
    return requests.get(f"{base_url.rstrip('/')}{_SMOKE_PATH}", headers=headers, timeout=_TIMEOUT_S)


@click.command("login")
@click.option("--api-key", "api_key_flag", default=None, help="Skip prompt; use this key.")
@click.option(
    "--project-id", "project_id_flag", default=None, help="Skip prompt; use this project id."
)
@click.option(
    "--base-url",
    default=None,
    help=f"Dataset gateway base URL (default: $DISSEQT_BASE_URL or {_http.DEFAULT_BASE_URL}).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit a machine-readable status envelope.")
def login(
    api_key_flag: str | None,
    project_id_flag: str | None,
    base_url: str | None,
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

    base_url = base_url or _http.base_url()
    try:
        resp = _smoke_test(base_url, api_key, project_id)
    except requests.RequestException as exc:
        # Never include the token in an error path.
        _fail(f"could not reach {base_url}: {type(exc).__name__}", code=1)

    if resp.status_code in (401, 403):
        _fail("invalid API key or project ID")
    if not resp.ok:
        _fail(f"unexpected HTTP {resp.status_code} verifying credentials", code=1)

    auth_save({"api_key": api_key, "project_id": project_id})

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
def logout() -> None:
    """Clear the locally stored credentials (~/.disseqt/config.json)."""
    try:
        stored = auth_load()
    except AuthConfigPermissionError as exc:
        _fail(str(exc))
    if stored is None:
        click.echo("not logged in; nothing to do")
        return
    auth_clear()
    click.echo("local config cleared")
