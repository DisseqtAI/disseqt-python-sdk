"""Shared CLI helpers — env-driven :class:`Client` construction, JSON output."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, NoReturn

import click

from ..client import Client

ENV_PROJECT_ID = "DISSEQT_PROJECT_ID"
ENV_API_KEY = "DISSEQT_API_KEY"
ENV_BASE_URL = "DISSEQT_BASE_URL"
ENV_POLICY_BASE_URL = "DISSEQT_POLICY_BASE_URL"
ENV_APP_NAME = "DISSEQT_APPLICATION_NAME"


def _fail(message: str, code: int = 2) -> NoReturn:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(code)


def build_client(application_name: str | None = None) -> Client:
    """Instantiate a :class:`Client` from the standard env vars.

    Fails fast with a clear message if required vars are missing —
    exit code 2 (usage error), leaving 1 free for BLOCK verdicts.
    """
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    if not project_id or not api_key:
        _fail(
            f"set {ENV_PROJECT_ID} and {ENV_API_KEY} in the environment "
            "(the CLI never reads credentials from flags)"
        )
    base_url = os.environ.get(ENV_BASE_URL) or "https://api.disseqt.ai/realtime-validations"
    policy_base_url = os.environ.get(ENV_POLICY_BASE_URL) or base_url
    app_name = application_name or os.environ.get(ENV_APP_NAME) or "disseqt-cli"
    return Client(
        project_id=project_id,
        api_key=api_key,
        base_url=base_url,
        realtime_policy_base_url=policy_base_url,
        application_name=app_name,
    )


def echo_json(payload: Any) -> None:
    """Print pretty JSON to stdout."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_error(exc: Exception) -> None:
    click.secho(f"error: {exc}", fg="red", err=True)


def resolve_input(input_arg: str | None, input_file: str | None) -> str:
    """Return the input string from either the flag or a file, or stdin."""
    if input_arg is not None:
        return input_arg
    if input_file is not None:
        if input_file == "-":
            return sys.stdin.read()
        with open(input_file, encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    _fail("provide --input, --input-file, or pipe content on stdin")
