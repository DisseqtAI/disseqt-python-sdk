"""CLI resource groups: ``target``, ``rag-target``, ``mcp-target``,
``pack``, ``pp-run``, ``output-validation``, ``session``, ``byov``,
``vulnerability`` (``plan``, ``plan-run``, ``app``: :mod:`.plans`).

Prompt-pack verbs use the service-key mount ``/api/v1/sdk/prompt-packs``
(api/server.go sdkPromptPackRoutes); the browser-session mount is not
reachable with an API key.

Thin wrappers over :mod:`._http` — one command per backend endpoint,
all on the dataset gateway (``DISSEQT_BASE_URL``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click

from ..api_client import DisseqtAPIClient
from ..resources import PACK_SHARING_SCOPES
from . import _http
from ._common import (
    _fail,
    echo_json,
    load_json_body,
    optional_target,
    require_credentials,
    target_options,
)
from .whoami import resolve_project_id, scope_params


def _json_option(
    help_text: str = "JSON body (use @path/to/file.json to read from disk).",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Reusable --json option that either reads a literal string or a file."""
    return click.option("--json", "json_body", help=help_text)


def _get(path: str, params: dict[str, Any] | None = None) -> None:
    echo_json(_http.request("GET", path, params=params))


def _post(path: str, body: dict[str, Any] | None = None) -> None:
    echo_json(_http.request("POST", path, json_body=body))


def _patch(path: str, body: dict[str, Any]) -> None:
    echo_json(_http.request("PATCH", path, json_body=body))


def _delete(path: str) -> None:
    echo_json(_http.request("DELETE", path))


def _get_raw(path: str) -> None:
    """Print a non-JSON body (CSV) verbatim."""
    click.echo(_http.request("GET", path, raw=True), nl=False)


_TARGET_FIELDS = {
    "app": "application_id",
    "integration": "app_integration_id",
    "endpoint": "endpoint_url",
}


def _api_client() -> DisseqtAPIClient:
    """Construct a :class:`DisseqtAPIClient` from the CLI credentials."""
    project_id, api_key = require_credentials()
    # No base_url: it resolves env > login config > default, same as _http.
    return DisseqtAPIClient(project_id=project_id, api_key=api_key)


# ---------------------------------------------------------------------------
# Targets  (/api/v1/llm/app-integrations)
# ---------------------------------------------------------------------------


@click.group("target")
def target() -> None:
    """LLM targets (app-integrations)."""


@target.command("create")
@_json_option("Target payload as JSON literal or @file.")
def target_create(json_body: str | None) -> None:
    _post("/api/v1/llm/app-integrations", load_json_body(json_body))


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
    _patch(f"/api/v1/llm/app-integrations/{target_id}", load_json_body(json_body))


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
        load_json_body(json_body),
    )


@target.command("probe")
@_json_option("Full target-integration JSON (unsaved).")
def target_probe(json_body: str | None) -> None:
    """Test-connection without saving (dry run)."""
    _post("/api/v1/llm/app-integrations/test-connection", load_json_body(json_body))


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
    # api/app_integration_types.go CurlCommands `json:"curl_commands"`.
    _post("/api/v1/llm/app-integrations/parse-curl", {"curl_commands": (curl_text or "").strip()})


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
        _post(base_path, load_json_body(json_body))

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
        _patch(f"{base_path}/{resource_id}", load_json_body(json_body))

    @group.command("delete")
    @click.argument("resource_id")
    def _del(resource_id: str) -> None:
        _delete(f"{base_path}/{resource_id}")

    return group


rag_target = _make_crud_group("rag-target", "/api/v1/llm/rag-integrations", "RAG targets.")
mcp_target = _make_crud_group("mcp-target", "/api/v1/llm/mcp-integrations", "MCP targets.")


# ---------------------------------------------------------------------------
# Packs  (/api/v1/sdk/prompt-packs)
# ---------------------------------------------------------------------------


@click.group("pack")
def pack() -> None:
    """Prompt-pack CRUD, prompts, publish/unpublish, ratings, reviews."""


@pack.command("create")
@_json_option()
def pack_create(json_body: str | None) -> None:
    """Create a pack.

    'severity', 'interaction_mode' and 'task_type' are ignored by the server:
    it aggregates them from the pack's prompts, so don't expect them back.
    """
    _post("/api/v1/sdk/prompt-packs", load_json_body(json_body))


@pack.command("list")
@click.option("--marketplace", is_flag=True, help="List the public marketplace catalogue instead.")
def pack_list(marketplace: bool) -> None:
    """List your own packs (use --marketplace for the public catalogue)."""
    suffix = "" if marketplace else "/my-packs"
    _get(f"/api/v1/sdk/prompt-packs{suffix}")


@pack.command("get")
@click.argument("pack_id")
def pack_get(pack_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}")


@pack.command("update")
@click.argument("pack_id")
@_json_option()
def pack_update(pack_id: str, json_body: str | None) -> None:
    """Update a pack.

    'severity', 'interaction_mode' and 'task_type' are ignored by the server:
    they are aggregated from the pack's prompts.
    """
    _patch(f"/api/v1/sdk/prompt-packs/{pack_id}", load_json_body(json_body))


@pack.command("delete")
@click.argument("pack_id")
def pack_delete(pack_id: str) -> None:
    _delete(f"/api/v1/sdk/prompt-packs/{pack_id}")


@pack.command("prompts")
@click.argument("pack_id")
def pack_prompts(pack_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/prompts")


@pack.command("add-prompts")
@click.argument("pack_id")
@_json_option("JSON with a 'prompts' array, or @file.")
def pack_add_prompts(pack_id: str, json_body: str | None) -> None:
    body = load_json_body(json_body)
    if "prompts" not in body:
        _fail("payload must contain a 'prompts' array")
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/prompts/bulk", body)


@pack.command("duplicate")
@click.argument("pack_id")
@_json_option("Optional duplicate-config JSON.")
def pack_duplicate(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/duplicate", load_json_body(json_body))


@pack.command("publish")
@click.argument("pack_id")
@click.option(
    "--sharing-scope",
    type=click.Choice([s.lower() for s in PACK_SHARING_SCOPES], case_sensitive=False),
    default="private",
    show_default=True,
    help="Visibility to publish the pack with (required by the backend).",
)
def pack_publish(pack_id: str, sharing_scope: str) -> None:
    """Publish a pack at the given sharing scope."""
    _patch(
        f"/api/v1/sdk/prompt-packs/{pack_id}/publish",
        {"sharing_scope": sharing_scope.upper()},
    )


@pack.command("unpublish")
@click.argument("pack_id")
def pack_unpublish(pack_id: str) -> None:
    _patch(f"/api/v1/sdk/prompt-packs/{pack_id}/unpublish", {})


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
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/ratings", load_json_body(json_body))


@pack.command("review")
@click.argument("pack_id")
@_json_option("Review JSON.")
def pack_review(pack_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/reviews", load_json_body(json_body))


# ---------------------------------------------------------------------------
# Runs (nested under packs)
# Named `pp-run` (prompt-pack run) to avoid clashing with `redteam run`.
# ---------------------------------------------------------------------------


@click.group("pp-run")
def pp_run() -> None:
    """Prompt-pack runs (create/list/get/delete/cancel/trace/report)."""


@pp_run.command("create")
@click.argument("pack_id")
@_json_option("Run payload JSON.")
@target_options(legacy=False)
def pp_run_create(
    pack_id: str,
    json_body: str | None,
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
) -> None:
    """Create a run; target flags set top-level ids (api/prompt_pack_runs_handlers.go
    PromptPackRunRequest). Without a flag the --json body must carry the target."""
    body = load_json_body(json_body)
    picked = optional_target(app_id, integration_id, endpoint)
    if picked:
        body = {**body, _TARGET_FIELDS[picked[0]]: picked[1]}
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/runs", body)


@pp_run.command("list")
@click.argument("pack_id")
def pp_run_list(pack_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/runs")


@pp_run.command("get")
@click.argument("run_id")
def pp_run_get(run_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/runs/{run_id}")


@pp_run.command("delete")
@click.argument("run_id")
def pp_run_delete(run_id: str) -> None:
    _delete(f"/api/v1/sdk/prompt-packs/runs/{run_id}")


@pp_run.command("stats")
@click.argument("pack_id")
def pp_run_stats(pack_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/runs/stats")


@pp_run.command("compare")
@click.argument("pack_id")
@click.option("--run1", "run1_id", required=True, help="First run id.")
@click.option("--run2", "run2_id", required=True, help="Second run id.")
def pp_run_compare(pack_id: str, run1_id: str, run2_id: str) -> None:
    params = {"run1_id": run1_id, "run2_id": run2_id}
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/runs/compare", params)


@pp_run.command("outputs")
@click.argument("run_id")
def pp_run_outputs(run_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/runs/{run_id}/outputs")


@pp_run.command("cancel")
@click.argument("run_id")
def pp_run_cancel(run_id: str) -> None:
    _post(f"/api/v1/sdk/prompt-packs/runs/{run_id}/cancel")


@pp_run.command("trace")
@click.argument("run_id")
def pp_run_trace(run_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/runs/{run_id}/trace")


@pp_run.command("report")
@click.argument("run_id")
def pp_run_report(run_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/runs/{run_id}/report")


@pp_run.command("validate")
@click.argument("run_id")
@_json_option("Validation payload JSON (validators, output_ids, etc.).")
def pp_run_validate(run_id: str, json_body: str | None) -> None:
    """Re-validate outputs of a run.

    Alias for ``output-validation create`` — POSTs to
    /api/v1/sdk/prompt-packs/runs/:run_id/validate-outputs. Kept on the pp-run
    group for discoverability alongside outputs / trace / report.
    """
    _post(f"/api/v1/sdk/prompt-packs/runs/{run_id}/validate-outputs", load_json_body(json_body))


@pp_run.command("add-to-pack")
@click.argument("run_id")
@click.option(
    "--entry-id",
    "entry_ids",
    multiple=True,
    required=True,
    help="An output's prompt_pack_entry_id to copy (repeatable).",
)
@click.option("--pack-id", required=True, help="Destination prompt pack ID.")
def pp_run_add_to_pack(run_id: str, entry_ids: tuple[str, ...], pack_id: str) -> None:
    """Copy the pack entries behind a run's outputs into another pack (``/prompts/add``)."""
    _post(f"/api/v1/sdk/prompt-packs/{pack_id}/prompts/add", {"entry_ids": list(entry_ids)})


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
    _post(f"/api/v1/sdk/prompt-packs/runs/{run_id}/validate-outputs", load_json_body(json_body))


@output_validation.command("list-for-pack")
@click.argument("pack_id")
def ov_list_for_pack(pack_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/output-validations")


@output_validation.command("get")
@click.argument("validation_id")
def ov_get(validation_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}")


@output_validation.command("summary")
@click.argument("validation_id")
def ov_summary(validation_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}/summary")


@output_validation.command("rca-status")
@click.argument("validation_id")
def ov_rca_status(validation_id: str) -> None:
    _get(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}/rca-status")


@output_validation.command("results-csv")
@click.argument("validation_id")
def ov_results_csv(validation_id: str) -> None:
    _get_raw(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}/results/csv")


@output_validation.command("compare")
@click.argument("pack_id")
@click.option("--validation1", "validation1_id", required=True, help="First validation id.")
@click.option("--validation2", "validation2_id", required=True, help="Second validation id.")
def ov_compare(pack_id: str, validation1_id: str, validation2_id: str) -> None:
    params = {"validation1_id": validation1_id, "validation2_id": validation2_id}
    _get(f"/api/v1/sdk/prompt-packs/{pack_id}/validations/compare", params)


@output_validation.command("delete")
@click.argument("validation_id")
def ov_delete(validation_id: str) -> None:
    _delete(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}")


@output_validation.command("cancel")
@click.argument("validation_id")
def ov_cancel(validation_id: str) -> None:
    _post(f"/api/v1/sdk/prompt-packs/output-validations/{validation_id}/cancel")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@click.group("session")
def session() -> None:
    """Red-team / testing sessions."""


@session.command("create")
@_json_option()
def session_create(json_body: str | None) -> None:
    _post("/api/v1/testing/sessions", load_json_body(json_body))


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
    _post(f"/api/v1/testing/sessions/{session_id}/runs", load_json_body(json_body))


@session.command("list-runs")
@click.argument("session_id")
def session_list_runs(session_id: str) -> None:
    _get(f"/api/v1/testing/sessions/{session_id}/runs")


@session.command("report-csv")
@click.argument("session_id")
def session_report_csv(session_id: str) -> None:
    _get_raw(f"/api/v1/testing/sessions/{session_id}/report/csv")


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
    _post("/api/v1/llm/custom-validators", load_json_body(json_body))


@byov.command("list")
def byov_list() -> None:
    _get("/api/v1/llm/custom-validators", {"project_id": resolve_project_id()})


@byov.command("get")
@click.argument("validator_id")
def byov_get(validator_id: str) -> None:
    _get(f"/api/v1/llm/custom-validators/{validator_id}")


@byov.command("update")
@click.argument("validator_id")
@_json_option()
def byov_update(validator_id: str, json_body: str | None) -> None:
    _patch(f"/api/v1/llm/custom-validators/{validator_id}", load_json_body(json_body))


@byov.command("delete")
@click.argument("validator_id")
def byov_delete(validator_id: str) -> None:
    _delete(f"/api/v1/llm/custom-validators/{validator_id}")


@byov.command("test")
@click.argument("validator_id")
@_json_option()
def byov_test(validator_id: str, json_body: str | None) -> None:
    _post(f"/api/v1/llm/custom-validators/{validator_id}/test", load_json_body(json_body))


@byov.command("probe")
@_json_option()
def byov_probe(json_body: str | None) -> None:
    """Test-connection for a BYOV validator without saving."""
    _post("/api/v1/llm/custom-validators/test-connection", load_json_body(json_body))


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
@target_options(legacy=False)
@click.option(
    "--organization-id",
    envvar="DISSEQT_ORGANIZATION_ID",
    help="Organization id (query param); defaults to the key's organization.",
)
def vuln_test(
    vuln_id: str,
    json_body: str | None,
    app_id: str | None,
    integration_id: str | None,
    endpoint: str | None,
    organization_id: str | None,
) -> None:
    """Test a vulnerability against an app (--app) or app integration (--integration)."""
    body = load_json_body(json_body)
    picked = optional_target(app_id, integration_id, endpoint)
    if picked and picked[0] == "endpoint":
        raise click.UsageError("vulnerability test supports --app or --integration only")
    if picked:
        body = {**body, _TARGET_FIELDS[picked[0]]: picked[1]}
    params = scope_params(organization_id)
    echo_json(
        _http.request(
            "POST", f"/api/v1/vulnerabilities/{vuln_id}/test", json_body=body, params=params
        )
    )


__all__ = [
    "byov",
    "mcp_target",
    "output_validation",
    "pack",
    "pp_run",
    "rag_target",
    "session",
    "target",
    "vulnerability",
]
