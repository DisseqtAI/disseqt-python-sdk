"""``disseqt whoami`` — introspect the credentials the SDK will use.

Reads the same env vars every other CLI command reads and prints them
(with the API key masked). No backend ``/auth/whoami`` endpoint exists
today; if one lands later we can switch to a live call here.
"""

from __future__ import annotations

import json
import os

import click

ENV_PROJECT_ID = "DISSEQT_PROJECT_ID"
ENV_API_KEY = "DISSEQT_API_KEY"
ENV_USER_EMAIL = "DISSEQT_USER_EMAIL"
ENV_ORGANIZATION_ID = "DISSEQT_ORGANIZATION_ID"


def _mask_api_key(api_key: str | None) -> str | None:
    """Show only a shape hint: ``sk_...abc``. Never echo the full key."""
    if not api_key:
        return None
    if len(api_key) <= 7:
        return "***"
    return f"{api_key[:3]}...{api_key[-3:]}"


def _collect() -> dict[str, str | None]:
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    source = "env"
    if not project_id and not api_key:
        # Fall through to ``~/.disseqt/config.json`` when env is empty.
        try:
            from ..auth import load as _load

            stored = _load()
        except Exception:
            stored = None
        if stored is not None:
            project_id = stored.get("project_id")
            api_key = stored.get("api_key")
            source = "config"
    return {
        "project_id": project_id,
        "api_key": _mask_api_key(api_key),
        "user_email": os.environ.get(ENV_USER_EMAIL),
        "organization_id": os.environ.get(ENV_ORGANIZATION_ID),
        "source": source,
    }


@click.command("whoami")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def whoami(as_json: bool) -> None:
    """Show which Disseqt identity the CLI is using (env-var driven)."""
    info = _collect()
    if as_json:
        click.echo(json.dumps(info, indent=2, sort_keys=True))
        return
    for key, value in info.items():
        click.echo(f"{key}: {value if value is not None else '(unset)'}")
