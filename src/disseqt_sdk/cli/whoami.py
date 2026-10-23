"""``disseqt whoami`` — the identity the gateway derives from the API key.

Calls the gateway's ``GET /api/v1/whoami``; on a 404 (gateway without the
route) it falls back to the locally resolved credentials. The key is masked.
"""

from __future__ import annotations

import json
from typing import Any

import click

from . import _http
from ._common import load_credentials, require_credentials


def _mask_api_key(api_key: str | None) -> str | None:
    """Show only a shape hint: ``sk_...abc``. Never echo the full key."""
    if not api_key:
        return None
    if len(api_key) <= 7:
        return "***"
    return f"{api_key[:3]}...{api_key[-3:]}"


def server_identity() -> dict[str, Any]:
    """``{project_id, organization_id, user_id}`` from the gateway."""
    data = _http.request("GET", "/api/v1/whoami")
    return data if isinstance(data, dict) else {}


def resolve_project_id() -> str:
    """Configured project, else the one the gateway derives from the key."""
    project_id, _ = require_credentials()
    project_id = project_id or server_identity().get("project_id")
    if not project_id:
        raise click.ClickException("could not resolve project_id from the API key")
    return str(project_id)


def _collect() -> dict[str, Any]:
    project_id, api_key, source = load_credentials()
    require_credentials()
    local = {"project_id": project_id, "organization_id": None, "user_id": None}
    try:
        server = server_identity()
        identity = {k: server.get(k) or v for k, v in local.items()}
    except _http.APIError as exc:
        if exc.status_code != 404:
            raise
        click.secho("warning: gateway has no /api/v1/whoami; showing local credentials", err=True)
        identity, source = local, "local"
    return {**identity, "api_key": _mask_api_key(api_key), "source": source}


@click.command("whoami")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def whoami(as_json: bool) -> None:
    """Show which Disseqt identity the CLI is using."""
    info = _collect()
    if as_json:
        click.echo(json.dumps(info, indent=2, sort_keys=True))
        return
    for key, value in info.items():
        click.echo(f"{key}: {value if value is not None else '(unset)'}")
