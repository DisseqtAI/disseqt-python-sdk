"""CLI resource groups: ``target``, ``rag-target``, ``mcp-target``,
``pack``, ``pp-run``, ``output-validation``, ``rag-validation``,
``session``, ``byov``, ``vulnerability``, ``plan``, ``plan-run``.

Thin wrappers over :mod:`._http` — one command per backend endpoint.
No new deps, no new env vars beyond the shared ``DISSEQT_DATASET_BASE_URL``
(falls back to the same host as ``DISSEQT_REDTEAM_BASE_URL``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

import click

from ..api_client import DisseqtAPIClient
from . import _http
from ._common import ENV_API_KEY, ENV_PROJECT_ID, _fail, echo_json

BASE_ENV = "DISSEQT_DATASET_BASE_URL"
DEFAULT_BASE = "https://api.disseqt.ai/dataset"


def _json_option(
    help_text: str = "JSON body (use @path/to/file.json to read from disk).",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Reusable --json option that either reads a literal string or a file."""
    return click.option("--json", "json_body", help=help_text)


def _load_json_body(json_body: str | None) -> dict[str, Any]:
    """Parse a --json value: literal JSON, or @file for a JSON file."""
    if not json_body:
        return {}
    if json_body.startswith("@"):
        with open(json_body[1:], encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(json_body)
        except json.JSONDecodeError as e:
            _fail(f"invalid --json: {e}")
    if not isinstance(data, dict):
        _fail("--json body must be a JSON object")
    return data


def _get(path: str, params: dict[str, Any] | None = None) -> None:
    echo_json(_http.request("GET", BASE_ENV, DEFAULT_BASE, path, params=params))


def _post(path: str, body: dict[str, Any] | None = None) -> None:
    echo_json(_http.request("POST", BASE_ENV, DEFAULT_BASE, path, json_body=body))


def _patch(path: str, body: dict[str, Any]) -> None:
    echo_json(_http.request("PATCH", BASE_ENV, DEFAULT_BASE, path, json_body=body))


def _delete(path: str) -> None:
    echo_json(_http.request("DELETE", BASE_ENV, DEFAULT_BASE, path))


def _api_client() -> DisseqtAPIClient:
    """Construct a :class:`DisseqtAPIClient` from the CLI env vars."""
    project_id = os.environ.get(ENV_PROJECT_ID)
    api_key = os.environ.get(ENV_API_KEY)
    if not project_id or not api_key:
        _fail(f"set {ENV_PROJECT_ID} and {ENV_API_KEY} in the environment")
    base_url = os.environ.get(BASE_ENV) or DEFAULT_BASE
    return DisseqtAPIClient(project_id=project_id, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Targets  (/api/v1/llm/app-integrations)
# ---------------------------------------------------------------------------


@click.group("target")
def target() -> None:
    """LLM targets (app-integrations)."""


@target.command("create")
@_json_option("Target payload as JSON literal or @file.")
def target_create(json_body: str | None) -> None:
    _post("/api/v1/llm/app-integrations", _load_json_body(json_body))


@target.command("list")
def target_list() -> None:
    _get("/api/v1/llm/app-integrations")


@target.command("get")
@click.argument("target_id")
def target_get(target_id: str) -> None:
    _get(f"/api/v1/llm/app-integrations/{target_id}")


@target.command("update")
@click.argument("target_id")
@_json_option("Patch payload as JSON literal or @file.")
def target_update(target_id: str, json_body: str | None) -> None:
    _patch(f"/api/v1/llm/app-integrations/{target_id}", _load_json_body(json_body))


@target.command("delete")
@click.argument("target_id")
def target_delete(target_id: str) -> None:
    _delete(f"/api/v1/llm/app-integrations/{target_id}")


@target.command("test")
@click.argument("target_id")
@_json_option("Optional test-payload JSON.")
def target_test(target_id: str, json_body: str | None) -> None:
    """Test a saved integration by id."""
    # Backend registers /:id/test-connection, not /:id/test (server.go:811).
    _post(
        f"/api/v1/llm/app-integrations/{target_id}/test-connection",
        _load_json_body(json_body),
    )


@target.command("probe")
@_json_option("Full target-integration JSON (unsaved).")
def target_probe(json_body: str | None) -> None:
    """Test-connection without saving (dry run)."""
    _post("/api/v1/llm/app-integrations/test-connection", _load_json_body(json_body))


@target.command("parse-curl")
@click.argument("curl_source", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--curl", "curl_text", help="Curl string literal.")
def target_parse_curl(curl_source: str | None, curl_text: str | None) -> None:
    """Parse a curl command into a target-integration preview."""
    if not curl_source and not curl_text:
        _fail("pass a FILE path or --curl STRING")
    if curl_source and curl_text:
        _fail("pass exactly one of FILE or --curl")
    if curl_source:
        with open(curl_source, encoding="utf-8") as f:
            curl_text = f.read()
    _post("/api/v1/llm/app-integrations/parse-curl", {"curl": (curl_text or "").strip()})


# ---------------------------------------------------------------------------
# RAG / MCP targets
# ---------------------------------------------------------------------------


def _make_crud_group(name: str, base_path: str, help_text: str) -> click.Group:
    """Factory for pure-CRUD resource groups (create/list/get/update/delete).

    ponytail: factory is fine here — three separate groups share the exact
    same five-verb shape. Break it back apart if any of them grows
    resource-specific verbs.
    """

    @click.group(name)
    def group() -> None:
        pass

    group.help = help_text

    @group.command("create")
    @_json_option()
    def _create(json_body: str | None) -> None:
        _post(base_path, _load_json_body(json_body))

    @group.command("list")
    def _list() -> None:
        _get(base_path)

    @group.command("get")
    @click.argument("resource_id")
    def _get_one(resource_id: str) -> None:
        _get(f"{base_path}/{resource_id}")

    @group.command("update")
    @click.argument("resource_id")
    @_json_option()
    def _update(resource_id: str, json_body: str | None) -> None:
        _patch(f"{base_path}/{resource_id}", _load_json_body(json_body))

    @group.command("delete")
    @click.argument("resource_id")
    def _del(resource_id: str) -> None:
        _delete(f"{base_path}/{resource_id}")

    return group


rag_target = _make_crud_group("rag-target", "/api/v1/llm/rag-integrations", "RAG targets.")
mcp_target = _make_crud_group("mcp-target", "/api/v1/llm/mcp-integrations", "MCP targets.")


# ---------------------------------------------------------------------------
# Packs  (/api/v1/prompt-packs)
# ---------------------------------------------------------------------------


@click.group("pack")
def pack() -> None:
    """Prompt-pack CRUD, prompts, publish/unpublish, ratings, reviews."""


@pack.command("create")
@_json_option()
def pack_create(json_body: str | None) -> None:
    _post("/api/v1/prompt-packs", _load_json_body(json_body))


@pack.command("list")
def pack_list() -> None:
    _get("/api/v1/prompt-packs")


@pack.command("get")
@click.argument("pack_id")
def pack_get(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}")


@pack.command("update")
@click.argument("pack_id")
@_json_option()
def pack_update(pack_id: str, json_body: str | None) -> None:
    _patch(f"/api/v1/prompt-packs/{pack_id}", _load_json_body(json_body))


@pack.command("delete")
@click.argument("pack_id")
def pack_delete(pack_id: str) -> None:
    _delete(f"/api/v1/prompt-packs/{pack_id}")


@pack.command("restore")
@click.argument("pack_id")
def pack_restore(pack_id: str) -> None:
    _post(f"/api/v1/prompt-packs/{pack_id}/restore")


@pack.command("prompts")
@click.argument("pack_id")
def pack_prompts(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/prompts")


@pack.command("add-prompts")
@click.argument("pack_id")
@_json_option("JSON with a 'prompts' array, or @file.")
def pack_add_prompts(pack_id: str, json_body: str | None) -> None:
    body = _load_json_body(json_body)
    if "prompts" not in body:
        _fail("payload must contain a 'prompts' array")
    _post(f"/api/v1/prompt-packs/{pack_id}/prompts/bulk", body)


@pack.command("duplicate")
@click.argument("pack_id")
@_json_option("Optional duplicate-config JSON.")
def pack_duplicate(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/{pack_id}/duplicate", _load_json_body(json_body))


@pack.command("publish")
@click.argument("pack_id")
def pack_publish(pack_id: str) -> None:
    # Backend PATCHes /api/v1/prompt-packs/:id/publish (server.go:2237).
    _patch(f"/api/v1/prompt-packs/{pack_id}/publish", {})


@pack.command("unpublish")
@click.argument("pack_id")
def pack_unpublish(pack_id: str) -> None:
    # See pack_publish; PATCH at server.go:2238.
    _patch(f"/api/v1/prompt-packs/{pack_id}/unpublish", {})


@pack.command("import-status")
@click.argument("pack_id")
def pack_import_status(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/import-status")


@pack.command("export")
@click.argument("pack_id")
def pack_export(pack_id: str) -> None:
    """Export pack contents (backend returns CSV or a download URL)."""
    # Delegates to PacksResource.download() so the SDK stays the source of truth.
    raw = _api_client().packs.download(pack_id)
    echo_json(raw.decode("utf-8", errors="replace"))


@pack.command("rate")
@click.argument("pack_id")
@_json_option("Rating JSON, e.g. '{\"rating\": 5}'.")
def pack_rate(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/{pack_id}/ratings", _load_json_body(json_body))


@pack.command("review")
@click.argument("pack_id")
@_json_option("Review JSON.")
def pack_review(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/{pack_id}/reviews", _load_json_body(json_body))


# ---------------------------------------------------------------------------
# Runs (nested under packs)
# Named `pp-run` (prompt-pack run) to avoid clashing with the existing
# top-level `disseqt run` command.
# ---------------------------------------------------------------------------


@click.group("pp-run")
def pp_run() -> None:
    """Prompt-pack runs (create/list/get/delete/cancel/trace/report)."""


@pp_run.command("create")
@click.argument("pack_id")
@_json_option("Run payload JSON.")
def pp_run_create(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/{pack_id}/runs", _load_json_body(json_body))


@pp_run.command("list")
@click.argument("pack_id")
def pp_run_list(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/runs")


@pp_run.command("get")
@click.argument("run_id")
def pp_run_get(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}")


@pp_run.command("delete")
@click.argument("run_id")
def pp_run_delete(run_id: str) -> None:
    _delete(f"/api/v1/prompt-packs/runs/{run_id}")


@pp_run.command("stats")
@click.argument("pack_id")
def pp_run_stats(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/runs/stats")


@pp_run.command("compare")
@click.argument("pack_id")
def pp_run_compare(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/runs/compare")


@pp_run.command("outputs")
@click.argument("run_id")
def pp_run_outputs(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}/outputs")


@pp_run.command("retrieval-traces")
@click.argument("run_id")
def pp_run_retrieval_traces(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}/retrieval-traces")


@pp_run.command("cancel")
@click.argument("run_id")
def pp_run_cancel(run_id: str) -> None:
    _post(f"/api/v1/prompt-packs/runs/{run_id}/cancel")


@pp_run.command("trace")
@click.argument("run_id")
def pp_run_trace(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}/trace")


@pp_run.command("report")
@click.argument("run_id")
def pp_run_report(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}/report")


@pp_run.command("reveal")
@click.argument("run_id")
@click.argument("output_id")
def pp_run_reveal(run_id: str, output_id: str) -> None:
    """Reveal a masked run output."""
    _post(f"/api/v1/prompt-packs/runs/{run_id}/results/{output_id}/reveal")


@pp_run.command("add-to-pack")
@click.argument("run_id")
@click.option(
    "--output-id",
    "output_ids",
    multiple=True,
    required=True,
    help="Output ID to add. Repeat --output-id for multiple.",
)
@click.option("--pack-id", required=True, help="Destination prompt pack ID.")
def pp_run_add_to_pack(run_id: str, output_ids: tuple[str, ...], pack_id: str) -> None:
    """Add selected run outputs to a prompt pack."""
    _post(
        f"/api/v1/prompt-packs/{pack_id}/prompts/add",
        {"run_id": run_id, "output_ids": list(output_ids)},
    )


# ---------------------------------------------------------------------------
# Output validations
# ---------------------------------------------------------------------------


@click.group("output-validation")
def output_validation() -> None:
    """Prompt-pack output-validations."""


@output_validation.command("create")
@click.argument("run_id")
@_json_option()
def ov_create(run_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/runs/{run_id}/validate-outputs", _load_json_body(json_body))


@output_validation.command("list-for-pack")
@click.argument("pack_id")
def ov_list_for_pack(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/output-validations")


@output_validation.command("get")
@click.argument("validation_id")
def ov_get(validation_id: str) -> None:
    _get(f"/api/v1/prompt-packs/output-validations/{validation_id}")


@output_validation.command("summary")
@click.argument("validation_id")
def ov_summary(validation_id: str) -> None:
    _get(f"/api/v1/prompt-packs/output-validations/{validation_id}/summary")


@output_validation.command("rca-status")
@click.argument("validation_id")
def ov_rca_status(validation_id: str) -> None:
    _get(f"/api/v1/prompt-packs/output-validations/{validation_id}/rca-status")


@output_validation.command("results-csv")
@click.argument("validation_id")
def ov_results_csv(validation_id: str) -> None:
    _get(f"/api/v1/prompt-packs/output-validations/{validation_id}/results/csv")


@output_validation.command("compare")
@click.argument("pack_id")
def ov_compare(pack_id: str) -> None:
    _get(f"/api/v1/prompt-packs/{pack_id}/validations/compare")


@output_validation.command("delete")
@click.argument("validation_id")
def ov_delete(validation_id: str) -> None:
    _delete(f"/api/v1/prompt-packs/output-validations/{validation_id}")


@output_validation.command("cancel")
@click.argument("validation_id")
def ov_cancel(validation_id: str) -> None:
    _post(f"/api/v1/prompt-packs/output-validations/{validation_id}/cancel")


# ---------------------------------------------------------------------------
# RAG validations
# ---------------------------------------------------------------------------


@click.group("rag-validation")
def rag_validation() -> None:
    """RAG-validations on prompt-pack runs."""


@rag_validation.command("create")
@click.argument("run_id")
@_json_option()
def rv_create(run_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/prompt-packs/runs/{run_id}/rag-validate", _load_json_body(json_body))


@rag_validation.command("list-for-run")
@click.argument("run_id")
def rv_list_for_run(run_id: str) -> None:
    _get(f"/api/v1/prompt-packs/runs/{run_id}/rag-validations")


@rag_validation.command("get")
@click.argument("validation_id")
def rv_get(validation_id: str) -> None:
    _get(f"/api/v1/prompt-packs/rag-validations/{validation_id}")


@rag_validation.command("cancel")
@click.argument("validation_id")
def rv_cancel(validation_id: str) -> None:
    _post(f"/api/v1/prompt-packs/rag-validations/{validation_id}/cancel")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@click.group("session")
def session() -> None:
    """Red-team / testing sessions."""


@session.command("create")
@_json_option()
def session_create(json_body: str | None) -> None:
    _post("/api/v1/testing/sessions", _load_json_body(json_body))


@session.command("list")
def session_list() -> None:
    _get("/api/v1/testing/sessions")


@session.command("get")
@click.argument("session_id")
def session_get(session_id: str) -> None:
    _get(f"/api/v1/testing/sessions/{session_id}")


@session.command("delete")
@click.argument("session_id")
def session_delete(session_id: str) -> None:
    _delete(f"/api/v1/testing/sessions/{session_id}")


@session.command("create-run")
@click.argument("session_id")
@_json_option()
def session_create_run(session_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/testing/sessions/{session_id}/runs", _load_json_body(json_body))


@session.command("list-runs")
@click.argument("session_id")
def session_list_runs(session_id: str) -> None:
    _get(f"/api/v1/testing/sessions/{session_id}/runs")


@session.command("report-csv")
@click.argument("session_id")
def session_report_csv(session_id: str) -> None:
    _get(f"/api/v1/testing/sessions/{session_id}/report/csv")


@session.command("run-get")
@click.argument("run_id")
def session_run_get(run_id: str) -> None:
    _get(f"/api/v1/testing/runs/{run_id}")


@session.command("run-results")
@click.argument("run_id")
def session_run_results(run_id: str) -> None:
    _get(f"/api/v1/testing/runs/{run_id}/results")


@session.command("run-breaches")
@click.argument("run_id")
def session_run_breaches(run_id: str) -> None:
    _get(f"/api/v1/testing/runs/{run_id}/results/breaches")


@session.command("run-cancel")
@click.argument("run_id")
def session_run_cancel(run_id: str) -> None:
    _post(f"/api/v1/testing/runs/{run_id}/cancel")


# ---------------------------------------------------------------------------
# BYOV Validators
# ---------------------------------------------------------------------------


@click.group("byov")
def byov() -> None:
    """Bring-your-own-validator (custom validators)."""


@byov.command("create")
@_json_option()
def byov_create(json_body: str | None) -> None:
    _post("/api/v1/llm/custom-validators", _load_json_body(json_body))


@byov.command("list")
def byov_list() -> None:
    _get("/api/v1/llm/custom-validators")


@byov.command("get")
@click.argument("validator_id")
def byov_get(validator_id: str) -> None:
    _get(f"/api/v1/llm/custom-validators/{validator_id}")


@byov.command("update")
@click.argument("validator_id")
@_json_option()
def byov_update(validator_id: str, json_body: str | None) -> None:
    _patch(f"/api/v1/llm/custom-validators/{validator_id}", _load_json_body(json_body))


@byov.command("delete")
@click.argument("validator_id")
def byov_delete(validator_id: str) -> None:
    _delete(f"/api/v1/llm/custom-validators/{validator_id}")


@byov.command("test")
@click.argument("validator_id")
@_json_option()
def byov_test(validator_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/llm/custom-validators/{validator_id}/test", _load_json_body(json_body))


@byov.command("probe")
@_json_option()
def byov_probe(json_body: str | None) -> None:
    """Test-connection for a BYOV validator without saving."""
    _post("/api/v1/llm/custom-validators/test-connection", _load_json_body(json_body))


# ---------------------------------------------------------------------------
# Vulnerabilities
# ---------------------------------------------------------------------------


@click.group("vulnerability")
def vulnerability() -> None:
    """Read-only vulnerabilities catalog + a run-test endpoint."""


@vulnerability.command("list")
def vuln_list() -> None:
    _get("/api/v1/vulnerabilities")


@vulnerability.command("get")
@click.argument("vuln_id")
def vuln_get(vuln_id: str) -> None:
    _get(f"/api/v1/vulnerabilities/{vuln_id}")


@vulnerability.command("test")
@click.argument("vuln_id")
@_json_option()
def vuln_test(vuln_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/vulnerabilities/{vuln_id}/test", _load_json_body(json_body))


# ---------------------------------------------------------------------------
# Test Plans (T3)   /api/v1/test-plans
# ---------------------------------------------------------------------------


@click.group("plan")
def plan() -> None:
    """Test plans (T3) — versioned red-team plan templates."""


@plan.command("create")
@_json_option("Plan payload JSON (recipe, inputs, categories...).")
def plan_create(json_body: str | None) -> None:
    _post("/api/v1/test-plans", _load_json_body(json_body))


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
    _patch(f"/api/v1/test-plans/{plan_id}", _load_json_body(json_body))


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
    _post(f"/api/v1/test-plans/{plan_id}/copy", _load_json_body(json_body))


@plan.command("versions")
@click.argument("plan_id")
def plan_versions(plan_id: str) -> None:
    _get(f"/api/v1/test-plans/{plan_id}/versions")


@plan.command("create-version")
@click.argument("plan_id")
@_json_option("New version payload (recipe + optional copy_inputs_from_version).")
def plan_create_version(plan_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/versions", _load_json_body(json_body))


@plan.command("publish")
@click.argument("plan_id")
@_json_option("Publish payload with sharing_scope + expected_sharing_scope.")
def plan_publish(plan_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/test-plans/{plan_id}/publish", _load_json_body(json_body))


@plan.command("generate-inputs")
@_json_option("Payload with app_description + subcategories + organization_id.")
def plan_generate_inputs(json_body: str | None) -> None:
    _post("/api/v1/test-plans/generate-inputs", _load_json_body(json_body))


@plan.command("generate-inputs-status")
@click.argument("job_id")
def plan_generate_inputs_status(job_id: str) -> None:
    _get(f"/api/v1/test-plans/generate-inputs/{job_id}")


# ---------------------------------------------------------------------------
# Test Plan Runs (T6)   /api/v1/test-plan-runs
# ---------------------------------------------------------------------------

# Terminal statuses per pkg/testplan/walk.go + stage.go.
_PLAN_RUN_TERMINAL = frozenset({"completed", "completed_with_errors", "cancelled", "failed"})


@click.group("plan-run")
def plan_run() -> None:
    """Test plan runs (T6) — execute a plan, poll status, fetch reports."""


@plan_run.command("create")
@click.argument("plan_id")
@_json_option("Run payload (target + optional version_id).")
@click.option("--wait", is_flag=True, help="Poll until the run reaches a terminal status.")
@click.option(
    "--interval",
    type=float,
    default=5.0,
    show_default=True,
    help="Polling interval in seconds when --wait is set.",
)
def plan_run_create(plan_id: str, json_body: str | None, wait: bool, interval: float) -> None:
    resp = _http.request(
        "POST",
        BASE_ENV,
        DEFAULT_BASE,
        f"/api/v1/test-plans/{plan_id}/runs",
        json_body=_load_json_body(json_body),
    )
    if not wait:
        echo_json(resp)
        return
    run_id = resp.get("run_id") if isinstance(resp, dict) else None
    if not isinstance(run_id, str) or not run_id:
        _fail("--wait requires a run_id in the create response")
    # ponytail: naive poll loop, no jitter/backoff; upgrade if runs commonly outlive 30-60 min.
    while True:
        latest = _http.request("GET", BASE_ENV, DEFAULT_BASE, f"/api/v1/test-plan-runs/{run_id}")
        status = (latest.get("status") if isinstance(latest, dict) else None) or (
            latest.get("header", {}).get("status")
            if isinstance(latest, dict) and isinstance(latest.get("header"), dict)
            else None
        )
        if status in _PLAN_RUN_TERMINAL:
            echo_json(latest)
            return
        time.sleep(interval)


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


__all__ = [
    "byov",
    "mcp_target",
    "output_validation",
    "pack",
    "plan",
    "plan_run",
    "pp_run",
    "rag_target",
    "rag_validation",
    "session",
    "target",
    "vulnerability",
]
