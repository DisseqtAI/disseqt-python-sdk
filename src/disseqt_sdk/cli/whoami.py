"""``disseqt whoami`` — introspect the credentials the CLI will use.

Same resolution as every other verb (env, then ``~/.disseqt/config.json``);
prints them with the API key masked. No backend ``/auth/whoami`` exists.
"""

from __future__ import annotations

import json
import os

import click

from ._common import load_credentials

ENV_ORGANIZATION_ID = "DISSEQT_ORGANIZATION_ID"


def _mask_api_key(api_key: str | None) -> str | None:
    """Show only a shape hint: ``sk_...abc``. Never echo the full key."""
    if not api_key:
        return None
    if len(api_key) <= 7:
        return "***"
    return f"{api_key[:3]}...{api_key[-3:]}"


def _collect() -> dict[str, str | None]:
    project_id, api_key, source = load_credentials()
    return {
        "project_id": project_id,
        "api_key": _mask_api_key(api_key),
        "organization_id": os.environ.get(ENV_ORGANIZATION_ID),
        "source": source,
    }


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
