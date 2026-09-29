"""Shared CLI helpers — env var names, error printer, JSON output.

``build_client()`` / ``resolve_input()`` / ``ENV_POLICY_BASE_URL`` were
removed alongside the ``disseqt validate`` and ``disseqt run`` commands
they served — both depended on the policy-evaluate path that no in-scope
backend registers.
"""

from __future__ import annotations

import json
import os
from typing import Any, NoReturn

import click

from ..auth import AuthConfigPermissionError
from ..auth import load as auth_load

ENV_PROJECT_ID = "DISSEQT_PROJECT_ID"
ENV_API_KEY = "DISSEQT_API_KEY"
ENV_BASE_URL = "DISSEQT_BASE_URL"


def _fail(message: str, code: int = 2) -> NoReturn:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(code)


def load_credentials() -> tuple[str | None, str | None, str]:
    """``(project_id, api_key, source)`` from env, else ``~/.disseqt/config.json``
    (written by ``disseqt login``). A config file with permissions wider than
    0600 is a hard error, never silently skipped."""
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    if project_id and api_key:
        return project_id, api_key, "env"
    try:
        stored = auth_load()
    except AuthConfigPermissionError as exc:
        _fail(str(exc))
    if stored is None:
        return project_id, api_key, "env"
    return project_id or stored.get("project_id"), api_key or stored.get("api_key"), "config"


def require_credentials() -> tuple[str, str]:
    project_id, api_key, _ = load_credentials()
    if not project_id or not api_key:
        _fail(f"no credentials: set {ENV_PROJECT_ID} and {ENV_API_KEY}, or run `disseqt login`")
    return project_id, api_key


def echo_json(payload: Any) -> None:
    """Print pretty JSON to stdout."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_error(exc: Exception) -> None:
    click.secho(f"error: {exc}", fg="red", err=True)
