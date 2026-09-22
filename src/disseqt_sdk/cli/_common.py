"""Shared CLI helpers — env var names, error printer, JSON output.

``build_client()`` / ``resolve_input()`` / ``ENV_POLICY_BASE_URL`` were
removed alongside the ``disseqt validate`` and ``disseqt run`` commands
they served — both depended on the policy-evaluate path that no in-scope
backend registers.
"""

from __future__ import annotations

import json
from typing import Any, NoReturn

import click

ENV_PROJECT_ID = "DISSEQT_PROJECT_ID"
ENV_API_KEY = "DISSEQT_API_KEY"
ENV_BASE_URL = "DISSEQT_BASE_URL"


def _fail(message: str, code: int = 2) -> NoReturn:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(code)


def echo_json(payload: Any) -> None:
    """Print pretty JSON to stdout."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_error(exc: Exception) -> None:
    click.secho(f"error: {exc}", fg="red", err=True)
