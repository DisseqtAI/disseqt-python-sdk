"""CLI groups ``plan`` (/api/v1/test-plans), ``plan-run`` (/api/v1/test-plan-runs) and
``app`` (policy-management applications + dataset application-links)."""

from __future__ import annotations

from typing import Any

import click

from ..resources import plan_run_target
from . import _http
from ._common import _fail, echo_json, load_json_body, optional_target, target_options
from .resources import _delete, _get, _json_option, _patch, _post

# ---------------------------------------------------------------------------
# Test Plans (T3)   /api/v1/test-plans
# ---------------------------------------------------------------------------


@click.group("plan")
def plan() -> None:
    """Test plans (T3) — versioned red-team plan templates."""


@plan.command("create")
@_json_option("Plan payload JSON (recipe, inputs, categories...).")
def plan_create(json_body: str | None) -> None:
    _post("/api/v1/test-plans", load_json_body(json_body))


@plan.command("list")
def plan_list() -> None:
    _get("/api/v1/test-plans")


@plan.command("gallery")
def plan_gallery() -> None:
    _get("/api/v1/test-plans/gallery")


@plan.command("list-deleted")
def plan_list_deleted() -> None:
    _get("/api/v1/test-plans/deleted")


@plan.command("options")
@click.argument("ref")
def plan_options(ref: str) -> None:
    _get(f"/api/v1/test-plans/options/{ref}")


@plan.command("get")
@click.argument("plan_id")
def plan_get(plan_id: str) -> None:
    _get(f"/api/v1/test-plans/{plan_id}")


@plan.command("summary")
@click.argument("plan_id")
def plan_summary(plan_id: str) -> None:
    _get(f"/api/v1/test-plans/{plan_id}/summary")


@plan.command("update")
@click.argument("plan_id")
@_json_option("PATCH body (must include expected_recipe_revision or expected_metadata_revision).")
def plan_update(plan_id: str, json_body: str | None) -> None:
    _patch(f"/api/v1/test-plans/{plan_id}", load_json_body(json_body))


@plan.command("delete")
@click.argument("plan_id")
def plan_delete(plan_id: str) -> None:
    _delete(f"/api/v1/test-plans/{plan_id}")


@plan.command("restore")
@click.argument("plan_id")
def plan_restore(plan_id: str) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/restore")


@plan.command("copy")
@click.argument("plan_id")
@_json_option('Optional copy payload (e.g. {"name": "..."}).')
def plan_copy(plan_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/copy", load_json_body(json_body))


@plan.command("versions")
@click.argument("plan_id")
def plan_versions(plan_id: str) -> None:
    _get(f"/api/v1/test-plans/{plan_id}/versions")


@plan.command("create-version")
@click.argument("plan_id")
@_json_option("New version payload (recipe + optional copy_inputs_from_version).")
def plan_create_version(plan_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/versions", load_json_body(json_body))


@plan.command("publish")
@click.argument("plan_id")
@_json_option("Publish payload with sharing_scope + expected_sharing_scope.")
def plan_publish(plan_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/publish", load_json_body(json_body))


@plan.command("generate-inputs")
@_json_option("Payload with app_description + subcategories + organization_id.")
def plan_generate_inputs(json_body: str | None) -> None:
    _post("/api/v1/test-plans/generate-inputs", load_json_body(json_body))


@plan.command("generate-inputs-status")
@click.argument("job_id")
def plan_generate_inputs_status(job_id: str) -> None:
    _get(f"/api/v1/test-plans/generate-inputs/{job_id}")


# ---------------------------------------------------------------------------
# Test Plan Runs (T6)   /api/v1/test-plan-runs
# ---------------------------------------------------------------------------


@click.group("plan-run")
def plan_run() -> None:
    """Test plan runs (T6) — execute a plan, poll status, fetch reports."""


@plan_run.command("create")
@click.argument("plan_id")
@_json_option("Run payload (target + optional version_id).")
@target_options(legacy=False)
@click.option("--wait", is_flag=True, help="Poll until the run reaches a terminal status.")
@click.option(
    "--interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Polling interval in seconds when --wait is set.",
)
@click.option(
    "--max-wait",
    type=float,
    default=900.0,
    show_default=True,
    help="Give up (exit 1) after this many seconds when --wait is set.",
)
def plan_run_create(
    plan_id: str,
    json_body: str | None,
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
    wait: bool,
    interval: float,
    max_wait: float,
) -> None:
    """Target flags build ``target`` (api/test_plan_runs_create_types.go)."""
    body = load_json_body(json_body)
    picked = optional_target(app_id, integration_id, endpoint)
    if picked:
        kind, value = picked
        body = {
            **body,
            "target": plan_run_target(
                value if kind == "app" else None,
                value if kind == "integration" else None,
                value if kind == "endpoint" else None,
            ),
        }
    resp = _http.request("POST", f"/api/v1/test-plans/{plan_id}/runs", json_body=body)
    if not wait:
        echo_json(resp)
        return
    run_id = (resp.get("run_id") or resp.get("id")) if isinstance(resp, dict) else None
    if not isinstance(run_id, str) or not run_id:
        _fail("--wait requires a run_id in the create response")
    # ponytail: naive poll loop, no jitter/backoff; upgrade if runs commonly outlive 30-60 min.
    echo_json(
        _http.wait_terminal(
            f"/api/v1/test-plan-runs/{run_id}",
            interval,
            max_wait,
            on_abort=lambda: _http.request("POST", f"/api/v1/test-plan-runs/{run_id}/cancel"),
            hint=f"Inspect with: disseqt plan-run get {run_id}",
            status_of=_plan_run_status,
        )
    )


def _plan_run_status(latest: Any) -> str:
    """Run status lives at ``status`` or ``header.status``."""
    if not isinstance(latest, dict):
        return ""
    header = latest.get("header")
    return str(latest.get("status") or (header or {}).get("status") or "")


@plan_run.command("list")
@click.argument("plan_id")
def plan_run_list(plan_id: str) -> None:
    _get(f"/api/v1/test-plans/{plan_id}/runs")


@plan_run.command("list-deleted")
def plan_run_list_deleted() -> None:
    _get("/api/v1/test-plan-runs/deleted")


@plan_run.command("get")
@click.argument("run_id")
def plan_run_get(run_id: str) -> None:
    _get(f"/api/v1/test-plan-runs/{run_id}")


@plan_run.command("stage")
@click.argument("run_id")
@click.argument("stage_key")
def plan_run_stage(run_id: str, stage_key: str) -> None:
    _get(f"/api/v1/test-plan-runs/{run_id}/stages/{stage_key}")


@plan_run.command("trace")
@click.argument("run_id")
@click.option("--prompt-ref", required=True, help="Prompt reference UUID.")
def plan_run_trace(run_id: str, prompt_ref: str) -> None:
    _get(f"/api/v1/test-plan-runs/{run_id}/trace", params={"prompt_ref": prompt_ref})


@plan_run.command("report")
@click.argument("run_id")
@click.option("--stage-key", help="Stage key (required by backend for the results page).")
def plan_run_report(run_id: str, stage_key: str | None) -> None:
    params = {"stage_key": stage_key} if stage_key else None
    _get(f"/api/v1/test-plan-runs/{run_id}/report", params=params)


@plan_run.command("prompts")
@click.argument("run_id")
def plan_run_prompts(run_id: str) -> None:
    _get(f"/api/v1/test-plan-runs/{run_id}/prompts")


@plan_run.command("cancel")
@click.argument("run_id")
def plan_run_cancel(run_id: str) -> None:
    _post(f"/api/v1/test-plan-runs/{run_id}/cancel")


@plan_run.command("delete")
@click.argument("run_id")
def plan_run_delete(run_id: str) -> None:
    _delete(f"/api/v1/test-plan-runs/{run_id}")


@plan_run.command("restore")
@click.argument("run_id")
def plan_run_restore(run_id: str) -> None:
    _post(f"/api/v1/test-plan-runs/{run_id}/restore")


@plan_run.command("reveal")
@click.argument("run_id")
@click.argument("result_id")
def plan_run_reveal(run_id: str, result_id: str) -> None:
    _post(f"/api/v1/test-plan-runs/{run_id}/results/{result_id}/reveal")


# ---------------------------------------------------------------------------
# AI Systems (applications): list via policy-management, link via dataset
# ---------------------------------------------------------------------------


@click.group("app")
def app() -> None:
    """AI Systems: list them and link one to an app integration."""


@app.command("list")
def app_list() -> None:
    """GET /api/v1/applications on policy-management (DISSEQT_POLICIES_BASE_URL)."""
    echo_json(_http.request("GET", "/api/v1/applications", base=_http.policies_base_url()))


@app.command("link")
@click.argument("application_id")
@click.option("--integration", "integration_id", required=True, help="App integration id.")
@click.option(
    "--role",
    type=click.Choice(["primary", "fallback", "candidate", "staging"]),
    help="Link role (server default: primary).",
)
def app_link(application_id: str, integration_id: str, role: str | None) -> None:
    """Link an AI System to an app integration (api/application_links.go)."""
    body = {
        "application_id": application_id,
        "integration_type": "app_integration",
        "integration_id": integration_id,
    }
    _post("/api/v1/llm/application-links", {**body, **({"role": role} if role else {})})


@app.command("unlink")
@click.argument("application_id")
def app_unlink(application_id: str) -> None:
    """Delete every link of the AI System (any integration kind); exit 1 if none."""
    links = _http.request(
        "GET", "/api/v1/llm/application-links", params={"application_id": application_id}
    )
    ids = [row["id"] for bucket in _LINK_BUCKETS for row in (links or {}).get(bucket) or []]
    if not ids:
        raise click.ClickException(f"AI System {application_id} has no linked integration")
    for link_id in ids:
        _delete(f"/api/v1/llm/application-links/{link_id}")


_LINK_BUCKETS = ("app_integrations", "llm_integrations", "rag_integrations", "mcp_integrations")


__all__ = ["app", "plan", "plan_run"]
