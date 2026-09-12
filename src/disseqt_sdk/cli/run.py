"""``disseqt run config.yaml`` — YAML batch runner + CI gate (Phase 2d).

Config shape (all sections optional except ``default_policies``)::

    models: {}          # reserved — no-op today
    target: {}          # reserved
    system_config: {}   # reserved
    default_policies:
      - <policy-id>
      - <policy-id>
    cases:              # list of validation cases to run
      - name: prompt-01
        input: "hello"
        context: null
        response: null
        policies: []    # optional per-case override
    custom_validators: []  # reserved
"""

from __future__ import annotations

import json
from typing import Any

import click

from ..client import Client
from ..models.input_validation import InputValidationRequest
from ..policy import any_blocking
from ._common import build_client, echo_json, print_error


def _load_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML file. Falls back to JSON when PyYAML is unavailable."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — optional dep
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise click.ClickException(f"{path}: top-level must be a mapping")
    return loaded


def _run_case(client: Client, case: dict[str, Any], defaults: list[str]) -> dict[str, Any]:
    policies = case.get("policies") or defaults
    if not policies:
        raise click.ClickException(
            f"case {case.get('name', '?')} has no policies and no default_policies set"
        )
    req = InputValidationRequest(
        prompt=case.get("input", ""),
        context=case.get("context"),
        response=case.get("response"),
    )
    result = client.validate(req, policies=list(policies))
    return {
        "name": case.get("name", ""),
        "blocked": any_blocking(result),
        "result": result,
    }


@click.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--fail-on-block",
    is_flag=True,
    help="Exit non-zero if any case's verdict is BLOCK.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit per-case results as JSON.",
)
def run(config_path: str, fail_on_block: bool, as_json: bool) -> None:
    """Run every case in CONFIG_PATH through the SDK."""
    try:
        cfg = _load_yaml(config_path)
    except Exception as exc:  # noqa: BLE001
        print_error(exc)
        raise SystemExit(2) from exc

    defaults = cfg.get("default_policies") or []
    cases = cfg.get("cases") or []
    if not cases:
        click.secho("no cases in config — nothing to do", fg="yellow", err=True)
        return

    client = build_client()
    outcomes: list[dict[str, Any]] = []
    for case in cases:
        try:
            outcomes.append(_run_case(client, case, defaults))
        except Exception as exc:  # noqa: BLE001
            outcomes.append({"name": case.get("name", "?"), "error": str(exc)})

    if as_json:
        echo_json(outcomes)
    else:
        for o in outcomes:
            name = o.get("name", "?")
            if "error" in o:
                click.secho(f"ERROR  {name}: {o['error']}", fg="red")
            elif o.get("blocked"):
                click.secho(f"BLOCK  {name}", fg="red")
            else:
                click.secho(f"PASS   {name}", fg="green")

    blocked = any(o.get("blocked") for o in outcomes)
    errored = any("error" in o for o in outcomes)
    if fail_on_block and (blocked or errored):
        raise SystemExit(1)
