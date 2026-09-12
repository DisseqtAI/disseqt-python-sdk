"""``disseqt redteam`` command family (Phase 4b + follow-up expansion).

Targets the newer ``/api/v1/testing/*`` surface + the batch endpoint
``/api/v1/mr-jailbreak/batch-automate``. All routes currently require
session cookies (see plan §4a — API-key middleware still pending);
setting ``DISSEQT_REDTEAM_BASE_URL`` lets you point at a mock during
that gap.

Follow-up adds ``run``, ``validate``, ``status``, ``cancel``, ``results``,
``list-personas``, ``list-techniques`` and ``report`` so the whole
red-team surface lives under one command group.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from typing import Any

import click

from . import _http
from ._common import echo_json

BASE_ENV = "DISSEQT_REDTEAM_BASE_URL"
DEFAULT_BASE = "https://api.disseqt.ai/dataset"

_TERMINAL_STATES = {"completed", "failed", "cancelled", "error", "success", "done"}


def _resolve_id(payload: Any, *keys: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in keys or ("id", "job_id", "run_id", "session_id"):
        val = payload.get(key)
        if val:
            return str(val)
    return None


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
def list_attacks(kind: str) -> None:
    """List available attack techniques (single-turn, multi-turn, agents)."""
    out: dict[str, Any] = {}
    if kind in ("single", "all"):
        out["single_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/attack-techniques"
        )
    if kind in ("multi", "all"):
        out["multi_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/techniques"
        )
    if kind in ("agents", "all"):
        out["agents"] = _http.request("GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/agents")
    echo_json(out)


@redteam.command("attack")
@click.option("--single-turn", "single_turn", is_flag=True, help="Run a single-turn attack.")
@click.option("--multi-turn", "multi_turn", is_flag=True, help="Run a multi-turn attack.")
@click.option("--technique", required=True, help="Attack technique id or name.")
@click.option("--target", required=True, help="Target model identifier / integration id.")
@click.option("--prompt", help="Seed prompt (single-turn) or objective (multi-turn).")
@click.option("--poll-interval", default=2.0, help="Seconds between run status polls.")
@click.option("--max-wait", default=300.0, help="Max seconds to wait for the run to finish.")
def attack(
    single_turn: bool,
    multi_turn: bool,
    technique: str,
    target: str,
    prompt: str | None,
    poll_interval: float,
    max_wait: float,
) -> None:
    """Run one red-team attack end to end."""
    if single_turn == multi_turn:
        raise click.UsageError("pass exactly one of --single-turn or --multi-turn")

    if multi_turn:
        body = {"technique": technique, "target": target, "objective": prompt or ""}
        result = _http.request(
            "POST", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/batch-automate", json_body=body
        )
        echo_json(result)
        return

    # single-turn: create session, create run, poll, fetch results
    session = _http.request(
        "POST", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/sessions", json_body={"target": target}
    )
    session_id = _resolve_id(session, "id", "session_id")
    if not session_id:
        raise click.ClickException(f"could not resolve session id from response: {session!r}")

    run = _http.request(
        "POST",
        BASE_ENV,
        DEFAULT_BASE,
        f"/api/v1/testing/sessions/{session_id}/runs",
        json_body={"technique": technique, "prompt": prompt or ""},
    )
    run_id = _resolve_id(run, "id", "run_id")
    if not run_id:
        raise click.ClickException(f"could not resolve run id from response: {run!r}")

    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        status = _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{run_id}")
        state = (status or {}).get("status") or (status or {}).get("state") or ""
        if state.lower() in _TERMINAL_STATES:
            results = _http.request(
                "GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{run_id}/results"
            )
            echo_json({"status": status, "results": results})
            return
        time.sleep(poll_interval)
    raise click.ClickException(f"run {run_id} did not finish within {max_wait}s")


@redteam.group("session")
def session() -> None:
    """Inspect red-team sessions."""


@session.command("list")
def session_list() -> None:
    """List red-team sessions."""
    echo_json(_http.request("GET", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/sessions"))


@session.command("get")
@click.argument("session_id")
def session_get(session_id: str) -> None:
    """Fetch one session by id."""
    echo_json(
        _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/sessions/{session_id}")
    )


@redteam.command("vuln-test")
@click.option("--vulnerability", "vuln_id", required=True, help="Vulnerability id to test.")
@click.option("--target", required=True, help="Target model identifier / integration id.")
def vuln_test(vuln_id: str, target: str) -> None:
    """Run the polling vuln-test endpoint for one vulnerability."""
    body = {"target": target}
    echo_json(
        _http.request(
            "POST",
            BASE_ENV,
            DEFAULT_BASE,
            f"/api/v1/vulnerabilities/{vuln_id}/test/poll",
            json_body=body,
        )
    )


# ---------------------------------------------------------------------------
# Follow-up additions: run / validate / status / cancel / results /
# list-personas / list-techniques / report
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    """Parse a YAML config; fall back to JSON when PyYAML isn't installed."""
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


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders using ``os.environ``."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _prompt_run_config() -> dict[str, Any]:
    """Interactive fallback when no config file is supplied and stdin is a TTY."""
    click.secho("Interactive red-team run", fg="cyan", bold=True)
    target = click.prompt("Target model / integration id", type=str)
    technique = click.prompt("Attack technique", type=str)
    vulnerability = click.prompt("Vulnerability id (leave blank to skip)", default="", type=str)
    return {
        "target": {"id": target},
        "techniques": [technique],
        "vulnerabilities": [vulnerability] if vulnerability else [],
    }


@redteam.command("run")
@click.argument("config_path", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit the raw job payload.")
def run(config_path: str | None, as_json: bool) -> None:
    """Run a full red-team suite from a YAML CONFIG_PATH.

    With no argument and an interactive TTY, prompts for target +
    technique + vulnerability and submits a minimal suite.
    """
    if config_path:
        cfg = _load_yaml(config_path)
    elif sys.stdin.isatty():
        cfg = _prompt_run_config()
    else:
        raise click.UsageError("provide a config path or run in an interactive terminal")

    cfg = _expand_env(cfg)
    target = cfg.get("target") or {}
    session_payload = {
        "target": target,
        "vulnerabilities": cfg.get("vulnerabilities") or [],
        "personas": cfg.get("personas") or [],
        "concurrency": cfg.get("concurrency"),
        "max_depth": cfg.get("max_depth"),
        "stop_on_first_success": cfg.get("stop_on_first_success"),
    }
    session = _http.request(
        "POST", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/sessions", json_body=session_payload
    )
    session_id = _resolve_id(session, "id", "session_id")
    if not session_id:
        raise click.ClickException(f"could not resolve session id from response: {session!r}")

    run_payload = {
        "techniques": cfg.get("techniques") or [],
        "personas": cfg.get("personas") or [],
        "vulnerabilities": cfg.get("vulnerabilities") or [],
    }
    launched = _http.request(
        "POST",
        BASE_ENV,
        DEFAULT_BASE,
        f"/api/v1/testing/sessions/{session_id}/runs",
        json_body=run_payload,
    )
    if as_json:
        echo_json({"session": session, "run": launched})
        return
    run_id = _resolve_id(launched, "id", "run_id") or "?"
    click.secho(f"session={session_id} run={run_id}", fg="green")
    click.echo(f"track: disseqt redteam status {run_id}")


@redteam.command("validate")
@click.option("--input", "input_text", required=True, help="Prompt to test.")
@click.option("--technique", required=True, help="Attack technique id or name.")
@click.option("--vulnerability", required=True, help="Vulnerability id or name.")
@click.option("--target", default="default", help="Target model / integration id.")
def validate(input_text: str, technique: str, vulnerability: str, target: str) -> None:
    """Fire a single red-team-flavored validation (one attack + one judge)."""
    body = {
        "prompt": input_text,
        "technique": technique,
        "vulnerability": vulnerability,
        "target": target,
    }
    echo_json(
        _http.request("POST", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/validate", json_body=body)
    )


def _status_probe(job_id: str) -> Any:
    """Try the testing surface first, fall back to mr-jailbreak."""
    try:
        return _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{job_id}")
    except SystemExit:
        return _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/mr-jailbreak/jobs/{job_id}")


@redteam.command("status")
@click.argument("job_id")
def status(job_id: str) -> None:
    """Fetch status for a running/completed job or run."""
    echo_json(_status_probe(job_id))


@redteam.command("cancel")
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel a running job or run."""
    echo_json(
        _http.request("POST", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{job_id}/cancel")
    )


@redteam.command("results")
@click.argument("job_id")
def results(job_id: str) -> None:
    """Pretty-print results for a completed job or run."""
    try:
        payload = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{job_id}/results"
        )
    except SystemExit:
        payload = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/mr-jailbreak/jobs/{job_id}/interactions"
        )
    echo_json(payload)


@redteam.command("list-personas")
@click.option("--attack-type", help="Filter personas by attack_type field if present.")
def list_personas(attack_type: str | None) -> None:
    """Enumerate persona agents (`/api/v1/mr-jailbreak/agents`)."""
    payload = _http.request("GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/agents")
    if attack_type and isinstance(payload, list):
        payload = [
            p for p in payload if isinstance(p, dict) and p.get("attack_type") == attack_type
        ]
    echo_json(payload)


@redteam.command("list-techniques")
@click.option("--single-turn", "single_turn", is_flag=True, help="Only single-turn techniques.")
@click.option("--multi-turn", "multi_turn", is_flag=True, help="Only multi-turn techniques.")
def list_techniques(single_turn: bool, multi_turn: bool) -> None:
    """List techniques, optionally filtered by turn kind."""
    if single_turn and multi_turn:
        raise click.UsageError("pass at most one of --single-turn or --multi-turn")
    out: dict[str, Any] = {}
    if single_turn or not multi_turn:
        out["single_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/testing/attack-techniques"
        )
    if multi_turn or not single_turn:
        out["multi_turn"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, "/api/v1/mr-jailbreak/techniques"
        )
    echo_json(out)


def _results_to_csv(payload: Any) -> str:
    """Flatten a results payload into CSV. Best-effort — dicts/list only."""
    rows = payload if isinstance(payload, list) else (payload or {}).get("results") or []
    if not isinstance(rows, list) or not rows:
        return ""
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        if isinstance(row, dict):
            writer.writerow({k: _stringify(v) for k, v in row.items()})
    return buf.getvalue()


def _results_to_markdown(payload: Any) -> str:
    rows = payload if isinstance(payload, list) else (payload or {}).get("results") or []
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
@click.argument("job_id")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    help="Report format. CSV hits the server-side CSV endpoint; others format locally.",
)
def report(job_id: str, fmt: str) -> None:
    """Export a report for a completed job."""
    if fmt == "csv":
        # Server-side CSV renderer if available.
        payload = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/sessions/{job_id}/report/csv"
        )
        if isinstance(payload, str):
            click.echo(payload)
            return
        # Server returned JSON — flatten locally.
        click.echo(_results_to_csv(payload))
        return

    payload = _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/testing/runs/{job_id}/results")
    if fmt == "json":
        echo_json(payload)
    else:  # markdown
        click.echo(_results_to_markdown(payload))


# ---------------------------------------------------------------------------
# Follow-up additions (batch 2):
#   - analytics [--summary|--prompts-stats]
#   - recommend {packs,attacks,validators}
#   - parse-curl / test-connection
#   - eval-csv / eval-single-turn
# ---------------------------------------------------------------------------

_JAILBREAK_BASE = "/api/v1/jailbreak"
_BOT_BASE = "/api/v1/testing/bot"


def _render_kv_table(title: str, payload: Any) -> None:
    """Best-effort rich table for a dict payload; fall back to JSON."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:  # pragma: no cover — rich is a first-party dep
        click.secho(title, fg="cyan", bold=True)
        echo_json(payload)
        return
    if not isinstance(payload, dict):
        # Non-dict (list/scalar) — render as JSON under the title.
        click.secho(title, fg="cyan", bold=True)
        echo_json(payload)
        return
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("field")
    table.add_column("value")
    for key, value in payload.items():
        table.add_row(str(key), _stringify(value))
    Console().print(table)


@redteam.command("analytics")
@click.option("--summary", "summary_only", is_flag=True, help="Only fetch the summary endpoint.")
@click.option(
    "--prompts-stats", "prompts_only", is_flag=True, help="Only fetch the prompts-stats endpoint."
)
@click.option(
    "--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def analytics(summary_only: bool, prompts_only: bool, fmt: str) -> None:
    """Show jailbreak analytics (summary + prompts-stats)."""
    if summary_only and prompts_only:
        raise click.UsageError("pass at most one of --summary or --prompts-stats")

    want_summary = summary_only or not prompts_only
    want_prompts = prompts_only or not summary_only

    out: dict[str, Any] = {}
    if want_summary:
        out["summary"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"{_JAILBREAK_BASE}/analytics/summary"
        )
    if want_prompts:
        out["prompts_stats"] = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"{_JAILBREAK_BASE}/analytics/prompts-stats"
        )

    if fmt == "json":
        # Unwrap single-key output when only one endpoint was requested.
        echo_json(next(iter(out.values())) if len(out) == 1 else out)
        return
    for title, payload in out.items():
        _render_kv_table(title.replace("_", " ").title(), payload)


_RECOMMEND_PATHS = {
    "packs": f"{_BOT_BASE}/recommend-packs",
    "attacks": f"{_BOT_BASE}/recommend-attacks",
    "validators": f"{_BOT_BASE}/recommend-validators",
}


@redteam.command("recommend")
@click.argument("kind", type=click.Choice(sorted(_RECOMMEND_PATHS)))
@click.option("--context", "context", help="Free-form context string.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON body to send instead of --context.",
)
def recommend(kind: str, context: str | None, config_path: str | None) -> None:
    """Ask the bot to recommend packs / attacks / validators."""
    if not context and not config_path:
        raise click.UsageError("pass --context or --config FILE")
    if context and config_path:
        raise click.UsageError("pass exactly one of --context or --config")
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            body = json.load(f)
        if not isinstance(body, dict):
            raise click.ClickException(f"{config_path}: top-level must be a JSON object")
    else:
        body = {"context": context}
    echo_json(_http.request("POST", BASE_ENV, DEFAULT_BASE, _RECOMMEND_PATHS[kind], json_body=body))


@redteam.command("parse-curl")
@click.argument("source", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--stdin", "from_stdin", is_flag=True, help="Read curl string from stdin.")
def parse_curl(source: str | None, from_stdin: bool) -> None:
    """Parse a curl command into a structured request payload."""
    if source == "-" or from_stdin or (source is None and not sys.stdin.isatty()):
        curl_text = sys.stdin.read()
    elif source:
        with open(source, encoding="utf-8") as f:
            curl_text = f.read()
    else:
        raise click.UsageError("provide a FILE path, --stdin, or pipe curl on stdin")
    curl_text = curl_text.strip()
    if not curl_text:
        raise click.UsageError("empty curl input")
    echo_json(
        _http.request(
            "POST", BASE_ENV, DEFAULT_BASE, f"{_BOT_BASE}/parse-curl", json_body={"curl": curl_text}
        )
    )


@redteam.command("test-connection")
@click.option(
    "--target",
    help="Target as provider/model (e.g., openai/gpt-4o). Falls back to active profile.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON body with a full target dict.",
)
def test_connection(target: str | None, config_path: str | None) -> None:
    """Ping the target model through the bot connectivity endpoint."""
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            body = json.load(f)
        if not isinstance(body, dict):
            raise click.ClickException(f"{config_path}: top-level must be a JSON object")
    elif target:
        if "/" in target:
            provider, _, model = target.partition("/")
            body = {"target": {"provider": provider, "model": model}}
        else:
            body = {"target": {"id": target}}
    else:
        # ponytail: no profile store yet; fall back to env var if set,
        # otherwise fail with a clear hint. Add a profile lookup when
        # multi-target profiles ship.
        env_target = os.environ.get("DISSEQT_REDTEAM_TARGET")
        if not env_target:
            raise click.UsageError(
                "pass --target provider/model, --config FILE, " "or set DISSEQT_REDTEAM_TARGET"
            )
        provider, _, model = env_target.partition("/")
        body = (
            {"target": {"provider": provider, "model": model}}
            if model
            else {"target": {"id": env_target}}
        )
    echo_json(
        _http.request(
            "POST", BASE_ENV, DEFAULT_BASE, f"{_BOT_BASE}/test-connection", json_body=body
        )
    )


@redteam.command("eval-csv")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_path", type=click.Path(dir_okay=False), help="Save results JSON.")
@click.option("--wait", "wait_for_completion", is_flag=True, help="Poll until the job finishes.")
@click.option("--poll-interval", default=2.0, help="Seconds between poll requests.")
@click.option("--max-wait", default=600.0, help="Max seconds to wait for completion.")
def eval_csv(
    path: str,
    output_path: str | None,
    wait_for_completion: bool,
    poll_interval: float,
    max_wait: float,
) -> None:
    """Upload a CSV to the bulk-evaluate endpoint and optionally poll for results."""
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh.read(), "text/csv")}
    submit = _http.request(
        "POST",
        BASE_ENV,
        DEFAULT_BASE,
        f"{_JAILBREAK_BASE}/evaluate-csv",
        files=files,
    )
    job_id = _resolve_id(submit, "job_id", "id")

    if not wait_for_completion:
        echo_json(submit)
        return
    if not job_id:
        raise click.ClickException(f"could not resolve job id from response: {submit!r}")

    deadline = time.monotonic() + max_wait
    last: Any = None
    while time.monotonic() < deadline:
        last = _http.request(
            "GET", BASE_ENV, DEFAULT_BASE, f"{_JAILBREAK_BASE}/jobs/{job_id}/process"
        )
        state = (last or {}).get("status") or (last or {}).get("state") or ""
        if str(state).lower() in _TERMINAL_STATES:
            if output_path:
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(last, out, indent=2, default=str)
                click.secho(f"wrote {output_path}", fg="green")
            else:
                echo_json(last)
            return
        time.sleep(poll_interval)
    raise click.ClickException(f"job {job_id} did not finish within {max_wait}s")


@redteam.command("eval-single-turn")
@click.option("--input", "input_text", required=True, help="Prompt to evaluate.")
@click.option("--technique", help="Attack technique to attribute the prompt to.")
@click.option("--vulnerability", help="Vulnerability to score against.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def eval_single_turn(
    input_text: str, technique: str | None, vulnerability: str | None, fmt: str
) -> None:
    """Evaluate one prompt against the single-turn jailbreak scorer."""
    body: dict[str, Any] = {"input": input_text}
    if technique:
        body["technique"] = technique
    if vulnerability:
        body["vulnerability"] = vulnerability
    payload = _http.request(
        "POST", BASE_ENV, DEFAULT_BASE, f"{_JAILBREAK_BASE}/single-turn-evaluate", json_body=body
    )
    if fmt == "json":
        echo_json(payload)
        return
    if isinstance(payload, dict):
        verdict = payload.get("verdict") or payload.get("decision") or "?"
        reason = payload.get("reason") or payload.get("rationale") or ""
        color = "green" if str(verdict).upper() in ("PASS", "SAFE", "OK") else "red"
        click.secho(f"verdict: {verdict}", fg=color, bold=True)
        if reason:
            click.echo(f"reason: {reason}")
    else:
        echo_json(payload)
