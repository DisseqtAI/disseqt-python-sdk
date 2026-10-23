"""Shared CLI helpers — env var names, error printer, JSON output.

``build_client()`` / ``resolve_input()`` / ``ENV_POLICY_BASE_URL`` were
removed alongside the ``disseqt validate`` and ``disseqt run`` commands
they served — both depended on the policy-evaluate path that no in-scope
backend registers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, NoReturn

import click

from ..auth import AuthConfigPermissionError
from ..auth import resolve as auth_resolve

ENV_PROJECT_ID = "DISSEQT_PROJECT_ID"
ENV_API_KEY = "DISSEQT_API_KEY"
ENV_BASE_URL = "DISSEQT_BASE_URL"


def _fail(message: str, code: int = 2) -> NoReturn:
    click.secho(f"error: {message}", fg="red", err=True)
    raise SystemExit(code)


def load_credentials() -> tuple[str | None, str | None, str]:
    """``(project_id, api_key, source)``: env beats ``~/.disseqt/config.json``
    (same precedence as the library). A config wider than 0600 is a hard error."""
    try:
        api_key, project_id, source = auth_resolve()
    except AuthConfigPermissionError as exc:
        _fail(str(exc))
    return project_id, api_key, source


def require_credentials() -> tuple[str | None, str]:
    """``(project_id or None, api_key)``; the project is optional (derived from the key)."""
    project_id, api_key, _ = load_credentials()
    if not api_key:
        _fail(
            f"no credentials: set {ENV_API_KEY} (optionally {ENV_PROJECT_ID}), or run `disseqt login`"
        )
    return project_id, api_key


def mask_key(api_key: str) -> str:
    """At most 8 chars and never more than half the key (Node ``maskedPrefix``)."""
    return f"{api_key[: min(8, len(api_key) // 2)]}..."


def load_json_body(json_body: str | None) -> dict[str, Any]:
    """Parse a --json value (literal JSON or @file) into an object; bad input exits 2."""
    if not json_body:
        return {}
    try:
        if json_body.startswith("@"):
            with open(json_body[1:], encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(json_body)
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"invalid --json: {e}")
    if not isinstance(data, dict):
        _fail("--json body must be a JSON object")
    return data


def echo_json(payload: Any) -> None:
    """Print pretty JSON to stdout."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def print_error(exc: Exception) -> None:
    click.secho(f"error: {exc}", fg="red", err=True)


def target_options(*, legacy: bool) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """--app / --integration / --endpoint (+ deprecated --target when ``legacy``)."""
    opts = [
        click.option("--app", "app_id", help="AI System (application) id."),
        click.option("--integration", "integration_id", help="App integration id."),
        click.option("--endpoint", "endpoint", help="Target endpoint URL."),
    ]
    if legacy:
        opts.append(
            click.option(
                "--target", "legacy_target", help="Deprecated: use --app/--integration/--endpoint."
            )
        )

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        for opt in reversed(opts):
            f = opt(f)
        return f

    return wrap


def pick_target(
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
    legacy_target: str | None = None,
    *,
    legacy_kind: Callable[[str], str] = lambda _: "app",
) -> tuple[str, str]:
    """``(kind, value)`` for exactly one target flag; kind is app|integration|endpoint|template."""
    if legacy_target:
        click.secho(
            "warning: --target is deprecated; use --app, --integration or --endpoint",
            fg="yellow",
            err=True,
        )
    given = [
        (kind, value)
        for kind, value in (
            ("app", app_id),
            ("integration", integration_id),
            ("endpoint", endpoint),
            ("legacy", legacy_target),
        )
        if value
    ]
    if len(given) != 1:
        raise click.UsageError("pass exactly one of --app, --integration or --endpoint")
    kind, value = given[0]
    return (legacy_kind(value), value) if kind == "legacy" else (kind, value)


def optional_target(
    app_id: str | None, integration_id: str | None, endpoint: str | None
) -> tuple[str, str] | None:
    """Like :func:`pick_target`, but no flag means "target is in the --json body"."""
    if not (app_id or integration_id or endpoint):
        return None
    return pick_target(app_id, integration_id, endpoint)
