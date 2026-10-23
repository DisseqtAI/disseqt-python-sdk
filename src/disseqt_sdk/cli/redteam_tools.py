"""``disseqt redteam`` admin verbs: analytics, recommend, parse-curl, test-connection,
technique / prompt / generated-prompt CRUD and breach lookups (api/jailbreak_routes.go,
api/testing_bot_types.go)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import click

from . import _http
from ._common import echo_json, load_json_body
from .redteam import _pages, _rows, _stringify, page_options, redteam

# ---------------------------------------------------------------------------
# analytics / recommend / parse-curl / test-connection
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
        out["summary"] = _http.request("GET", f"{_JAILBREAK_BASE}/analytics/summary")
    if want_prompts:
        # Backend registers /prompts-stats (not /analytics/prompts-stats) at
        # jailbreak_routes.go:57. /analytics/ only prefixes /analytics/summary.
        out["prompts_stats"] = _http.request("GET", f"{_JAILBREAK_BASE}/prompts-stats")

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
_MIN_APP_DESCRIPTION = 10  # api/testing_bot_types.go RecommendPacksRequest binding min=10


@redteam.command("recommend")
@click.argument("kind", type=click.Choice(sorted(_RECOMMEND_PATHS)))
@click.option("--app-name", help="Application name (packs only).")
@click.option("--app-description", help="Application description, at least 10 characters.")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Full JSON body to send instead of the flags.",
)
def recommend(
    kind: str, app_name: str | None, app_description: str | None, config_path: str | None
) -> None:
    """Ask the bot to recommend packs / attacks / validators
    (api/testing_bot_types.go Recommend*Request)."""
    if config_path:
        if app_name or app_description:
            raise click.UsageError("pass --config alone, or --app-name/--app-description")
        with open(config_path, encoding="utf-8") as f:
            body = json.load(f)
        if not isinstance(body, dict):
            raise click.ClickException(f"{config_path}: top-level must be a JSON object")
    else:
        if not app_description or len(app_description) < _MIN_APP_DESCRIPTION:
            raise click.UsageError(
                f"--app-description of at least {_MIN_APP_DESCRIPTION} characters is required"
            )
        body = {"app_description": app_description}
        if kind == "packs":
            if not app_name:
                raise click.UsageError("recommend packs needs --app-name")
            body["app_name"] = app_name
    echo_json(_http.request("POST", _RECOMMEND_PATHS[kind], json_body=body))


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
    # api/testing_bot_handlers.go parseCurlForBot binds {"curl_command"}.
    echo_json(
        _http.request("POST", f"{_BOT_BASE}/parse-curl", json_body={"curl_command": curl_text})
    )


@redteam.command("test-connection")
@click.option(
    "--endpoint", default="", help="Target endpoint URL (blank for name-routed providers)."
)
@click.option("--provider", default="", help="Provider name, e.g. openai.")
@click.option("--model", default="", help="Model name, e.g. gpt-4o.")
@click.option("--api-key", "api_key", help="Target credential (prefer --api-key-env).")
@click.option("--api-key-env", "api_key_env", help="Env var holding the target credential.")
def test_connection(
    endpoint: str, provider: str, model: str, api_key: str | None, api_key_env: str | None
) -> None:
    """Ping a target through the bot connectivity endpoint
    (api/testing_bot_handlers.go testTargetConnectionDirect: flat body, api_key required)."""
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise click.UsageError(f"{api_key_env} is not set")
    if not api_key:
        raise click.UsageError("pass --api-key or --api-key-env")
    body = {"endpoint": endpoint, "provider": provider, "model": model, "api_key": api_key}
    echo_json(_http.request("POST", f"{_BOT_BASE}/test-connection", json_body=body))


# ---------------------------------------------------------------------------
# Jailbreak technique / prompt CRUD + generated-prompt mark-successful +
# breach get. Wire contract: dataset-backend api/jailbreak_routes.go on stage.
# ---------------------------------------------------------------------------


@redteam.group("technique")
def technique() -> None:
    """Jailbreak techniques CRUD (/api/v1/jailbreak/techniques)."""


@technique.command("create")
@click.option("--json", "json_body", help="Technique payload as JSON literal or @file.")
def technique_create(json_body: str | None) -> None:
    echo_json(
        _http.request(
            "POST",
            f"{_JAILBREAK_BASE}/techniques",
            json_body=load_json_body(json_body),
        )
    )


@technique.command("update")
@click.argument("technique_id")
@click.option("--json", "json_body", help="Patch payload as JSON literal or @file.")
def technique_update(technique_id: str, json_body: str | None) -> None:
    echo_json(
        _http.request(
            "PATCH",
            f"{_JAILBREAK_BASE}/techniques/{technique_id}",
            json_body=load_json_body(json_body),
        )
    )


@technique.command("delete")
@click.argument("technique_id")
def technique_delete(technique_id: str) -> None:
    echo_json(
        _http.request(
            "DELETE",
            f"{_JAILBREAK_BASE}/techniques/{technique_id}",
        )
    )


@redteam.group("prompt")
def prompt() -> None:
    """Jailbreak prompts CRUD (/api/v1/jailbreak/prompts)."""


@prompt.command("create")
@click.option("--json", "json_body", help="Prompt payload as JSON literal or @file.")
def prompt_create(json_body: str | None) -> None:
    echo_json(
        _http.request(
            "POST",
            f"{_JAILBREAK_BASE}/prompts",
            json_body=load_json_body(json_body),
        )
    )


@prompt.command("get")
@click.argument("prompt_id")
def prompt_get(prompt_id: str) -> None:
    echo_json(_http.request("GET", f"{_JAILBREAK_BASE}/prompts/{prompt_id}"))


@prompt.command("update")
@click.argument("prompt_id")
@click.option("--json", "json_body", help="Patch payload as JSON literal or @file.")
def prompt_update(prompt_id: str, json_body: str | None) -> None:
    echo_json(
        _http.request(
            "PATCH",
            f"{_JAILBREAK_BASE}/prompts/{prompt_id}",
            json_body=load_json_body(json_body),
        )
    )


@prompt.command("delete")
@click.argument("prompt_id")
def prompt_delete(prompt_id: str) -> None:
    echo_json(_http.request("DELETE", f"{_JAILBREAK_BASE}/prompts/{prompt_id}"))


@redteam.group("generated-prompt")
def generated_prompt() -> None:
    """Generated jailbreak prompts."""


@generated_prompt.command("mark-successful")
@click.argument("generated_prompt_id")
def generated_prompt_mark_successful(generated_prompt_id: str) -> None:
    """PATCH /generated-prompts/:id/success — flag a generated prompt as successful."""
    echo_json(
        _http.request(
            "PATCH",
            f"{_JAILBREAK_BASE}/generated-prompts/{generated_prompt_id}/success",
            json_body={"success": True},
        )
    )


@redteam.group("breach")
def breach() -> None:
    """Breach ("finding") lookup on a red-team run."""


@breach.command("list")
@click.argument("run_id")
@page_options
def breach_list(run_id: str, page: int, page_size: int, fetch_all: bool) -> None:
    """List breaches for a run (GET /testing/runs/:id/results/breaches)."""
    echo_json(_pages(f"/api/v1/testing/runs/{run_id}/results/breaches", page, page_size, fetch_all))


@breach.command("get")
@click.argument("run_id")
@click.argument("breach_id")
def breach_get(run_id: str, breach_id: str) -> None:
    """Get one breach by id — client-side filter (no /findings/:id server route)."""
    payload = _pages(f"/api/v1/testing/runs/{run_id}/results/breaches", 1, 100, True)
    for row in _rows(payload):
        if isinstance(row, dict) and str(row.get("id", "")) == breach_id:
            echo_json(row)
            return
    raise click.ClickException(f"breach {breach_id} not found in run {run_id}")
