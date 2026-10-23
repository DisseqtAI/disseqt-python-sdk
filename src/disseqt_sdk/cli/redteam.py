"""``disseqt redteam`` command family.

Targets the dataset gateway's ``/api/v1/testing/*``, ``/api/v1/jailbreak/*``
and ``/api/v1/mr-jailbreak/*`` routes with the user's project key (see
:mod:`._http`). Request bodies mirror the Go structs in
disseqt-dataset-backend-service (``api/testing_types.go``,
``pkg/testing/pipeline.go``, ``api/mr_jailbreak_batch_automation.go``,
``api/testing_bot_types.go``); the file/line is cited next to each builder.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from typing import Any

import click

from ..models.redteam import has_more_pages
from . import _http
from ._common import echo_json, load_json_body, pick_target, target_options
from .whoami import resolve_project_id, scope_params

# ListJailbreakAgentsRequest requires page_id + kebab-case page-size (5..100).
# ponytail: first 100 agents only; add --page when a project has more.
_AGENTS_PAGE = {"page_id": 1, "page-size": 100}


def _resolve_id(payload: Any, *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys or ("id", "job_id", "run_id", "session_id"):
        val = payload.get(key)
        if val:
            return str(val)
    return None


def page_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """--page / --page-size / --all for ``{items, pagination}`` list routes."""
    f = click.option("--all", "fetch_all", is_flag=True, help="Fetch every page.")(f)
    f = click.option("--page-size", default=100, show_default=True, type=click.IntRange(1, 100))(f)
    return click.option("--page", default=1, show_default=True, type=click.IntRange(1))(f)


def _pages(
    path: str,
    page: int,
    page_size: int,
    fetch_all: bool,
    *,
    page_key: str = "page",
    list_key: str = "items",
) -> Any:
    """GET one page, or every page with ``fetch_all`` (pagination.has_next / total_count)."""

    def get(n: int) -> Any:
        return _http.request("GET", path, params={page_key: n, "page-size": page_size})

    data = get(page)
    if not fetch_all or not isinstance(data, dict):
        return data
    items = list(data.get(list_key) or [])
    while has_more_pages(data, page, len(items), list_key):
        page += 1
        data = get(page)
        items.extend(data.get(list_key) or [])
    return {**data, list_key: items}


def _testing_plan(
    pack_ids: list[str],
    techniques: list[str],
    validators: list[str],
    *,
    stop_on_first_breach: bool = False,
    max_total_prompts: int = 0,
) -> dict[str, Any]:
    """pkg/testing/pipeline.go TestingPlanConfig. Only the ``prompt_pack`` source
    and the ``single_turn_jailbreak`` strategy are registered server-side
    (pkg/task/testing_run_task.go); the strategy reads ``config.techniques``."""
    return {
        "prompt_sources": [{"type": "prompt_pack", "config": {"pack_ids": pack_ids}}],
        "attack_strategies": [
            {
                "type": "single_turn_jailbreak",
                "techniques": techniques,
                "config": {"techniques": techniques},
            }
        ],
        "validators": validators,
        "execution": {
            "mode": "sequential",
            "stop_on_first_breach": stop_on_first_breach,
            "max_total_prompts": max_total_prompts,
        },
    }


def _session_body(
    name: str, app_name: str, app_type: str, target: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    """api/testing_types.go CreateTestingSessionRequest — all four keys required.
    application_context needs name + type or the session stays ``draft``
    (determineSessionStatus, api/testing_handlers.go)."""
    return {
        "name": name,
        "application_context": {"name": app_name, "type": app_type},
        "target_config": target,
        "testing_plan": plan,
    }


def _run_body(run_name: str) -> dict[str, Any]:
    """api/testing_types.go CreateTestingRunRequest."""
    return {"run_name": run_name, "trigger_metadata": {"source": "disseqt-cli"}}


def _create_session_and_run(session_body: dict[str, Any], run_name: str) -> tuple[Any, Any]:
    session = _http.request("POST", "/api/v1/testing/sessions", json_body=session_body)
    session_id = _resolve_id(session, "id", "session_id")
    if not session_id:
        raise click.ClickException(f"could not resolve session id from response: {session!r}")
    run = _http.request(
        "POST", f"/api/v1/testing/sessions/{session_id}/runs", json_body=_run_body(run_name)
    )
    return session, run


def _interactive_menu() -> None:
    """Human-friendly landing screen shown when stdin is a TTY."""
    click.secho("disseqt redteam", fg="cyan", bold=True)
    click.echo("Full red-team surface. Pick a verb:\n")
    verbs = [
        ("run [config.yaml]", "Run a YAML-driven suite (interactive if no file)"),
        ("validate", "One-shot red-team validation (technique + vulnerability)"),
        ("attack", "Run one attack end-to-end (single-turn or multi-turn)"),
        ("status <job_id>", "Poll status of a run/job"),
        ("cancel <job_id>", "Cancel a running job"),
        ("results <job_id>", "Fetch results for a completed job"),
        ("report <job_id>", "Export a report (json | csv | markdown)"),
        ("session list|get", "Inspect testing sessions"),
        ("list-attacks", "List techniques + agents (raw)"),
        ("list-techniques", "List techniques with single/multi-turn filter"),
        ("list-personas", "Enumerate persona agents"),
        ("vuln-test", "Run the polling vuln-test endpoint"),
    ]
    for name, desc in verbs:
        click.echo(f"  {click.style(name, fg='green'):<28} {desc}")
    click.echo("\nQuickstart: " + click.style("disseqt redteam list-attacks", fg="yellow"))
    click.echo("Full help:  " + click.style("disseqt redteam --help", fg="yellow"))


@click.group("redteam", invoke_without_command=True)
@click.pass_context
def redteam(ctx: click.Context) -> None:
    """Run red-team attacks (single-turn, multi-turn, vuln tests)."""
    if ctx.invoked_subcommand is not None:
        return
    # Bare invocation.
    if sys.stdin.isatty():
        _interactive_menu()
    else:
        click.echo(ctx.get_help())


@redteam.command("list-attacks")
@click.option("--kind", type=click.Choice(["single", "multi", "agents", "all"]), default="all")
@page_options
def list_attacks(kind: str, page: int, page_size: int, fetch_all: bool) -> None:
    """List available attack techniques (single-turn, multi-turn, agents)."""
    out: dict[str, Any] = {}
    if kind in ("single", "all"):
        out["single_turn"] = _pages("/api/v1/testing/attack-techniques", page, page_size, fetch_all)
    # mr-jailbreak lists page with page_id and return {techniques|agents, total_count}.
    if kind in ("multi", "all"):
        out["multi_turn"] = _pages(
            "/api/v1/mr-jailbreak/techniques",
            page,
            max(page_size, 5),
            fetch_all,
            page_key="page_id",
            list_key="techniques",
        )
    if kind in ("agents", "all"):
        out["agents"] = _pages(
            "/api/v1/mr-jailbreak/agents",
            page,
            max(page_size, 5),
            fetch_all,
            page_key="page_id",
            list_key="agents",
        )
    echo_json(out)


_SINGLE_TURN_KEYS = {
    "app": "application_id",
    "integration": "app_integration_id",
    "endpoint": "endpoint_url",
}
_TARGET_KEYS = tuple(_SINGLE_TURN_KEYS.values())
_NIL_UUID = "00000000-0000-0000-0000-000000000000"


def _is_template(value: str) -> bool:
    return value.lstrip().startswith(("{", "@"))


@redteam.command("attack")
@click.option("--single-turn", "single_turn", is_flag=True, help="Run a single-turn attack.")
@click.option("--multi-turn", "multi_turn", is_flag=True, help="Run a multi-turn attack.")
@click.option(
    "--technique",
    required=True,
    help="Single-turn: attack technique key. Multi-turn: MR jailbreak technique UUID.",
)
@target_options(legacy=True)
@click.option("--pack", "pack_ids", multiple=True, help="Single-turn: prompt pack id (repeatable).")
@click.option(
    "--validator", "validators", multiple=True, help="Single-turn: validator name (repeatable)."
)
@click.option(
    "--prompt", "prompts", multiple=True, help="Multi-turn: target prompt (repeatable, 1..10)."
)
@click.option("--app-name", default=None, help="Multi-turn app_name (default: template name).")
@click.option("--app-description", default="", help="Multi-turn: short app description.")
@click.option("--app-type", default="chatbot", show_default=True, help="Multi-turn app_type.")
@click.option(
    "--max-depth", default=5, type=click.IntRange(1, 10), help="Multi-turn: conversation depth."
)
@click.option("--poll-interval", default=2.0, help="Seconds between status polls.")
@click.option("--max-wait", default=300.0, help="Max seconds to wait; the run is cancelled after.")
def attack(
    single_turn: bool,
    multi_turn: bool,
    technique: str,
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
    legacy_target: str | None,
    pack_ids: tuple[str, ...],
    validators: tuple[str, ...],
    prompts: tuple[str, ...],
    app_name: str | None,
    app_description: str,
    app_type: str,
    max_depth: int,
    poll_interval: float,
    max_wait: float,
) -> None:
    """Run one red-team attack end to end against exactly one target."""
    if single_turn == multi_turn:
        raise click.UsageError("pass exactly one of --single-turn or --multi-turn")
    kind, value = pick_target(
        app_id,
        integration_id,
        endpoint,
        legacy_target,
        legacy_kind=lambda v: "template" if _is_template(v) else "app",
    )
    if multi_turn:
        batch = _multi_turn(
            kind, value, technique, prompts, app_name, app_description, app_type, max_depth
        )
        _poll_jobs(batch, poll_interval, max_wait)
        return
    if kind == "template":
        raise click.UsageError(
            "single-turn takes --app, --integration or --endpoint, not a template"
        )
    if not pack_ids or not validators:
        raise click.UsageError("--single-turn needs at least one --pack and one --validator")
    name = f"cli-{technique}-{int(time.time())}"
    plan = _testing_plan(list(pack_ids), [technique], list(validators))
    target = {_SINGLE_TURN_KEYS[kind]: value}
    _, run = _create_session_and_run(_session_body(name, name, "web", target, plan), name)
    run_id = _resolve_id(run, "id", "run_id")
    if not run_id:
        raise click.ClickException(f"could not resolve run id from response: {run!r}")
    status = _http.wait_terminal(
        f"/api/v1/testing/runs/{run_id}",
        poll_interval,
        max_wait,
        on_abort=lambda: _http.request("POST", f"/api/v1/testing/runs/{run_id}/cancel"),
        hint=f"Inspect with: disseqt redteam status {run_id}",
    )
    results = _http.request("GET", f"/api/v1/testing/runs/{run_id}/results")
    echo_json({"status": status, "results": results})


def _multi_turn(
    kind: str,
    value: str,
    technique: str,
    prompts: tuple[str, ...],
    app_name: str | None,
    app_description: str,
    app_type: str,
    max_depth: int,
) -> Any:
    """POST batch-automate (api/mr_jailbreak_batch_automation.go BatchAutomateJailbreakRequest)."""
    if not 1 <= len(prompts) <= 10:
        raise click.UsageError("--multi-turn needs 1..10 --prompt values")
    if kind == "endpoint" and not _is_template(value):
        raise click.UsageError(
            "multi-turn --endpoint takes an app-integration template (JSON or @file); "
            "use --integration for a saved target"
        )
    template = load_json_body(value) if kind in ("endpoint", "template") else None
    if template is not None:
        missing = [
            k for k in ("name", "base_url", "integration_type", "send_step") if k not in template
        ]
        if missing:
            raise click.UsageError(f"template is missing {missing}")
    name = app_name or (template or {}).get("name") or "disseqt-cli"
    config: dict[str, Any] = {
        "project_id": resolve_project_id(),
        "job_name_prefix": "cli",
        "app_name": name,
        "app_description_short": app_description or f"{name} (disseqt CLI)",
        "app_type": app_type,
        "max_depth": max_depth,
        "orchestration_mode": "single",
        "technique_id": technique,
    }
    body: dict[str, Any] = {
        "target_prompts": list(prompts),
        "jailbreak_config": config,
        "ecid_prefix": "cli",
        "ecid_start_number": 1,
    }
    if template is not None:
        body["app_integration_template"] = template
    elif kind == "integration":
        body["app_integration_id"] = value
    else:
        config["application_id"] = value
    return _http.request("POST", "/api/v1/mr-jailbreak/batch-automate", json_body=body)


def _poll_jobs(batch: Any, poll_interval: float, max_wait: float) -> None:
    """Poll each MR job; no MR cancel route, so Ctrl-C/timeout only prints a hint."""
    results = (batch or {}).get("results") if isinstance(batch, dict) else None
    # Failed rows carry the nil UUID (TargetPromptResult.JobID is not a pointer).
    job_ids = [
        str(r["job_id"])
        for r in results or []
        if isinstance(r, dict) and r.get("job_id") and r["job_id"] != _NIL_UUID
    ]
    jobs = [
        _http.wait_terminal(
            f"/api/v1/mr-jailbreak/jobs/{job_id}",
            poll_interval,
            max_wait,
            hint=f"Jobs keep running server-side; inspect with: disseqt redteam status {job_id}",
        )
        for job_id in job_ids
    ]
    echo_json({"batch": batch, "jobs": jobs})


@redteam.group("session")
def session() -> None:
    """Inspect red-team sessions."""


@session.command("list")
def session_list() -> None:
    """List red-team sessions."""
    echo_json(_http.request("GET", "/api/v1/testing/sessions"))


@session.command("get")
@click.argument("session_id")
def session_get(session_id: str) -> None:
    """Fetch one session by id."""
    echo_json(_http.request("GET", f"/api/v1/testing/sessions/{session_id}"))


@redteam.command("vuln-test")
@click.option("--vulnerability", "vuln_id", required=True, help="Vulnerability id to test.")
@target_options(legacy=True)
@click.option(
    "--organization-id",
    envvar="DISSEQT_ORGANIZATION_ID",
    help="Organization id (query param); defaults to the key's organization.",
)
def vuln_test(
    vuln_id: str,
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
    legacy_target: str | None,
    organization_id: str | None,
) -> None:
    """POST /api/v1/vulnerabilities/{id}/test/poll (api/vulnerability_types.go
    VulnerabilityTestRequest; app integrations only, ids travel as query params)."""
    # Old --target meant app_integration_id here; keep that meaning.
    kind, value = pick_target(
        app_id, integration_id, endpoint, legacy_target, legacy_kind=lambda _: "integration"
    )
    if kind == "endpoint":
        raise click.UsageError(
            "vuln-test supports app integrations only: use --app or --integration"
        )
    body = {"application_id" if kind == "app" else "app_integration_id": value}
    params = scope_params(organization_id)
    echo_json(
        _http.request(
            "POST", f"/api/v1/vulnerabilities/{vuln_id}/test/poll", json_body=body, params=params
        )
    )


# ---------------------------------------------------------------------------
# Follow-up additions: run / validate / status / cancel / results /
# list-personas / list-techniques / report
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML config; JSON configs work without PyYAML."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        if path.lower().endswith((".yaml", ".yml")):
            raise click.ClickException(
                f"{path}: reading YAML needs PyYAML — pip install 'disseqt-ai-sdk[cli]'"
            ) from None
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise click.ClickException(f"{path}: top-level must be a mapping") from None
        return loaded
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise click.ClickException(f"{path}: top-level must be a mapping")
    return loaded


_ENV_REF = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` from ``os.environ``; an unset VAR is a usage error."""
    if isinstance(value, str):
        missing = [v for v in _ENV_REF.findall(value) if v not in os.environ]
        if missing:
            raise click.UsageError(f"config references unset env var(s): {', '.join(missing)}")
        return _ENV_REF.sub(lambda m: os.environ[m.group(1)], value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _prompt_run_config() -> dict[str, Any]:
    """Interactive fallback when no config file is supplied and stdin is a TTY."""
    click.secho("Interactive red-team run", fg="cyan", bold=True)
    return {
        "target": {"application_id": click.prompt("Application id", type=str)},
        "prompt_packs": [click.prompt("Prompt pack id", type=str)],
        "techniques": [click.prompt("Attack technique key", type=str)],
        "validators": [click.prompt("Validator name", type=str)],
    }


@redteam.command("run")
@click.argument("config_path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit the raw session + run payloads.")
def run(config_path: str | None, as_json: bool) -> None:
    """Create a testing session + run from a YAML CONFIG_PATH.

    See docs/redteam-example.yaml for the key → request-field mapping. With
    no argument and an interactive TTY, prompts for the minimal set.
    """
    if config_path:
        cfg = _load_yaml(config_path)
    elif sys.stdin.isatty():
        cfg = _prompt_run_config()
    else:
        raise click.UsageError("provide a config path or run in an interactive terminal")

    cfg = _expand_env(cfg)
    name = str(cfg.get("name") or f"cli-run-{int(time.time())}")
    app = cfg.get("application") or {}
    plan = _testing_plan(
        list(cfg.get("prompt_packs") or []),
        list(cfg.get("techniques") or []),
        list(cfg.get("validators") or []),
        stop_on_first_breach=cfg.get("stop_on_first_breach") is True,
        max_total_prompts=int(cfg.get("max_total_prompts") or 0),
    )
    target = cfg.get("target") or {}
    if not isinstance(target, dict) or sum(bool(target.get(k)) for k in _TARGET_KEYS) != 1:
        raise click.UsageError(f"target: needs exactly one of {', '.join(_TARGET_KEYS)}")
    session_body = _session_body(
        name, str(app.get("name") or name), str(app.get("type") or "web"), target, plan
    )
    session, launched = _create_session_and_run(session_body, str(cfg.get("run_name") or name))
    if as_json:
        echo_json({"session": session, "run": launched})
        return
    click.secho(
        f"session={_resolve_id(session, 'id', 'session_id')} run={_resolve_id(launched, 'id', 'run_id') or '?'}",
        fg="green",
    )
    click.echo(
        f"track: disseqt redteam status {_resolve_id(launched, 'id', 'run_id') or '<run id>'}"
    )


@redteam.command("validate")
@click.option("--input", "input_text", required=True, help="Prompt / LLM input to score.")
@click.option("--output", "output_text", default="", help="LLM output to score (may be empty).")
@click.option(
    "--validator",
    "validators",
    multiple=True,
    required=True,
    help="Validator name (repeatable). At least one required; up to 32.",
)
@click.option("--input-context", default="", help="Optional context passed to the validator.")
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True),
    help="Override per-validator threshold (0 < t <= 1). Absent = server default.",
)
def validate(
    input_text: str,
    output_text: str,
    validators: tuple[str, ...],
    input_context: str,
    threshold: float | None,
) -> None:
    """POST /api/v1/testing/validate — one-shot single-turn validation.

    Wire contract matches disseqt-dataset-backend-service PR #794:
    {input, output, validators, input_context, threshold?}.
    """
    body: dict[str, Any] = {
        "input": input_text,
        "output": output_text,
        "validators": list(validators),
        "input_context": input_context,
    }
    if threshold is not None:
        body["threshold"] = threshold
    echo_json(_http.request("POST", "/api/v1/testing/validate", json_body=body))


def _testing_or_mr(testing_path: str, mr_path: str) -> Any:
    """Try the testing surface; on 404 only, fall back to mr-jailbreak."""
    try:
        return _http.request("GET", testing_path)
    except _http.APIError as exc:
        if exc.status_code != 404:
            raise
        return _http.request("GET", mr_path)


@redteam.command("status")
@click.argument("job_id")
def status(job_id: str) -> None:
    """Fetch status for a running/completed job or run."""
    echo_json(
        _testing_or_mr(f"/api/v1/testing/runs/{job_id}", f"/api/v1/mr-jailbreak/jobs/{job_id}")
    )


@redteam.command("cancel")
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel a running job or run."""
    echo_json(_http.request("POST", f"/api/v1/testing/runs/{job_id}/cancel"))


@redteam.command("results")
@click.argument("job_id")
@page_options
def results(job_id: str, page: int, page_size: int, fetch_all: bool) -> None:
    """Results for a run (``{items, pagination}``); falls back to MR job interactions on 404."""
    try:
        payload = _pages(f"/api/v1/testing/runs/{job_id}/results", page, page_size, fetch_all)
    except _http.APIError as exc:
        if exc.status_code != 404:
            raise
        payload = _http.request("GET", f"/api/v1/mr-jailbreak/jobs/{job_id}/interactions")
    echo_json(payload)


@redteam.command("list-personas")
@click.option("--attack-type", help="Server-side attack_type filter.")
def list_personas(attack_type: str | None) -> None:
    """Enumerate persona agents (api/jailbreak_agents_handlers.go ListJailbreakAgentsRequest)."""
    params: dict[str, Any] = {
        **_AGENTS_PAGE,
        **({"attack_type": attack_type} if attack_type else {}),
    }
    echo_json(_http.request("GET", "/api/v1/mr-jailbreak/agents", params=params))


@redteam.command("list-techniques")
@click.option("--single-turn", "single_turn", is_flag=True, help="Only single-turn techniques.")
@click.option("--multi-turn", "multi_turn", is_flag=True, help="Only multi-turn techniques.")
def list_techniques(single_turn: bool, multi_turn: bool) -> None:
    """List techniques, optionally filtered by turn kind."""
    if single_turn and multi_turn:
        raise click.UsageError("pass at most one of --single-turn or --multi-turn")
    # Every page: the routes default to 20 items.
    out: dict[str, Any] = {}
    if single_turn or not multi_turn:
        out["single_turn"] = _pages("/api/v1/testing/attack-techniques", 1, 100, True)
    if multi_turn or not single_turn:
        out["multi_turn"] = _pages(
            "/api/v1/mr-jailbreak/techniques",
            1,
            100,
            True,
            page_key="page_id",
            list_key="techniques",
        )
    echo_json(out)


def _rows(payload: Any) -> list[Any]:
    """Rows of a ``{items, pagination}`` page (api/testing_types.go ListTestingResultsResponse)."""
    if isinstance(payload, list):
        return payload
    rows = (payload or {}).get("items") if isinstance(payload, dict) else None
    return rows if isinstance(rows, list) else []


def _results_to_markdown(payload: Any) -> str:
    rows = _rows(payload)
    if not isinstance(rows, list) or not rows:
        return "_no results_\n"
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    headers.append(k)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        if isinstance(row, dict):
            lines.append("| " + " | ".join(_stringify(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def _stringify(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return str(v)


@redteam.command("report")
@click.argument("run_id", required=False)
@click.option("--session", "session_id", help="Session id — required with --format csv.")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    help="csv = server-side per-session CSV; json/markdown = per-run results, formatted locally.",
)
def report(run_id: str | None, session_id: str | None, fmt: str) -> None:
    """Export a report for a completed run (json/markdown) or session (csv)."""
    if fmt == "csv":
        if not session_id:
            raise click.UsageError("--format csv needs --session <id>")
        csv_text = _http.request(
            "GET", f"/api/v1/testing/sessions/{session_id}/report/csv", raw=True
        )
        click.echo(csv_text, nl=False)
        return
    if not run_id:
        raise click.UsageError("RUN_ID is required for --format json|markdown")
    payload = _pages(f"/api/v1/testing/runs/{run_id}/results", 1, 100, True)
    if fmt == "json":
        echo_json(payload)
    else:  # markdown
        click.echo(_results_to_markdown(payload))
