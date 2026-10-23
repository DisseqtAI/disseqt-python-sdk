"""Resource clients for the Disseqt dataset-backend REST surface.

Additive to :class:`~disseqt_sdk.api_client.DisseqtAPIClient`: mounted as
lazy attribute-style resources (``client.targets``, ``client.packs``,
``client.runs``, ``client.output_validations``,
``client.sessions``, ``client.byov_validators``, ``client.rag_targets``,
``client.mcp_targets``, ``client.vulnerabilities``). Each resource is a
thin façade — one method per backend endpoint, ``dict`` in / ``dict`` out.

Auth + headers + base URL all reuse the parent
:class:`DisseqtAPIClient`; only the URL path prefix changes (these
endpoints live on ``/api/v1/*`` directly rather than under the prompt-packs
Kong route).
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

from .models.redteam import TestingRunStatus, TestingValidateResult, has_more_pages

if TYPE_CHECKING:
    from .api_client import DisseqtAPIClient

# Server-side enum for pack publishing (api/prompt_packs_handlers.go
# publishPromptPack: binding:"required,oneof=PRIVATE PROJECT ORGANIZATION PUBLIC").
PACK_SHARING_SCOPES = ("PRIVATE", "PROJECT", "ORGANIZATION", "PUBLIC")


def _one_target(**ids: str | None) -> tuple[str, str] | None:
    """The single ``(field, value)`` among target kwargs; >1 is a ValueError."""
    given = [(k, v) for k, v in ids.items() if v]
    if len(given) > 1:
        raise ValueError(f"pass at most one target, got {[k for k, _ in given]}")
    return given[0] if given else None


def _with_fields(payload: dict[str, Any] | None, **fields: Any) -> dict[str, Any]:
    """Shallow copy of ``payload`` with ``fields`` added (never mutates the caller's dict)."""
    return {**(payload or {}), **fields}


class _Base:
    """Shared plumbing: hold the parent client, forward to its ``_request_abs``."""

    def __init__(self, api_client: DisseqtAPIClient) -> None:
        self._client = api_client

    def _req(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client._request_abs(method, path, json_payload=json_payload, params=params)

    def _req_bytes(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        return self._client._request_abs_bytes(method, path, params=params)


class TargetsResource(_Base):
    """LLM targets — ``app_integrations`` table.

    Endpoints: ``/api/v1/llm/app-integrations`` (+ ``/test``,
    ``/test-connection``, ``/parse-curl``).
    """

    _BASE = "/api/v1/llm/app-integrations"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def get(self, target_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{target_id}")

    def update(self, target_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{target_id}", json_payload=payload)

    def delete(self, target_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{target_id}")

    def test(self, target_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Test a saved integration by id.

        Backend route: POST /api/v1/llm/app-integrations/:id/test-connection
        (dataset-backend api/server.go:811). The bare ``/:id/test`` suffix is
        not registered.
        """
        return self._req(
            "POST",
            f"{self._BASE}/{target_id}/test-connection",
            json_payload=payload or {},
        )

    def probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Test connection without saving (dry run).

        Node-SDK parity: same request as ``TargetsClient.testConnection`` in
        disseqt-node-sdk. Names diverge historically — kept as-is to avoid
        a breaking rename with no correctness upside.
        """
        return self._req("POST", f"{self._BASE}/test-connection", json_payload=payload)

    def parse_curl(self, curl_text: str) -> dict[str, Any]:
        """Parse a curl command (api/app_integration_types.go ``curl_commands``)."""
        return self._req(
            "POST", f"{self._BASE}/parse-curl", json_payload={"curl_commands": curl_text}
        )


class RagTargetsResource(_Base):
    """RAG targets — separate table + endpoints."""

    _BASE = "/api/v1/llm/rag-integrations"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def get(self, target_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{target_id}")

    def update(self, target_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{target_id}", json_payload=payload)

    def delete(self, target_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{target_id}")


class McpTargetsResource(_Base):
    """MCP targets — separate table + endpoints."""

    _BASE = "/api/v1/llm/mcp-integrations"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def get(self, target_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{target_id}")

    def update(self, target_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{target_id}", json_payload=payload)

    def delete(self, target_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{target_id}")


class PacksResource(_Base):
    """Prompt-packs — full CRUD, prompts, publish/unpublish, ratings, reviews.

    Complements the legacy ``generate_prompt_pack`` on
    :class:`DisseqtAPIClient`; those methods stay for back-compat.
    """

    # Service-key mount (api/server.go sdkPromptPackRoutes). /api/v1/prompt-packs is
    # browser-session only and rejects API-key callers with ErrAuthHeaderRequired.
    _BASE = "/api/v1/sdk/prompt-packs"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a pack.

        ``severity``, ``interaction_mode`` and ``task_type`` are ignored by the
        server — it aggregates them from the pack's prompts and blanks whatever
        you send (api/prompt_packs_handlers.go createPromptPack ~:855-870).
        """
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """List the public marketplace catalogue. For your own packs use :meth:`list_mine`."""
        return self._req("GET", self._BASE, params=params)

    def list_mine(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """List packs owned by the caller (``/my-packs``, api/server.go sdkPromptPackRoutes)."""
        return self._req("GET", f"{self._BASE}/my-packs", params=params)

    def get(self, pack_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}")

    def update(self, pack_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Update a pack.

        As with :meth:`create`, ``severity``, ``interaction_mode`` and
        ``task_type`` are ignored — they are aggregated from the pack's prompts.
        """
        return self._req("PATCH", f"{self._BASE}/{pack_id}", json_payload=payload)

    def delete(self, pack_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{pack_id}")

    def prompts(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/prompts", params=params)

    def add_prompts(self, pack_id: str, prompts: builtins.list[dict[str, Any]]) -> dict[str, Any]:
        """Bulk-add prompts to a pack. Uses the /prompts/bulk endpoint."""
        return self._req(
            "POST", f"{self._BASE}/{pack_id}/prompts/bulk", json_payload={"prompts": prompts}
        )

    def attach_entries(self, pack_id: str, entry_ids: builtins.list[str]) -> dict[str, Any]:
        """Attach existing library entries to a pack by id (/prompts/add).

        To create brand-new prompts use :meth:`add_prompts` instead.
        """
        return self._req(
            "POST", f"{self._BASE}/{pack_id}/prompts/add", json_payload={"entry_ids": entry_ids}
        )

    def duplicate(self, pack_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/duplicate", json_payload=payload or {})

    def publish(self, pack_id: str, sharing_scope: str = "PRIVATE") -> dict[str, Any]:
        """Publish a pack. ``sharing_scope`` is required by the backend."""
        scope = sharing_scope.upper()
        if scope not in PACK_SHARING_SCOPES:
            raise ValueError(
                f"sharing_scope must be one of {', '.join(PACK_SHARING_SCOPES)}; got {sharing_scope!r}"
            )
        return self._req(
            "PATCH", f"{self._BASE}/{pack_id}/publish", json_payload={"sharing_scope": scope}
        )

    def unpublish(self, pack_id: str) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{pack_id}/unpublish")

    def download(self, pack_id: str) -> bytes:
        """Download pack as CSV. Returns raw bytes (CSV content)."""
        return self._req_bytes("GET", f"{self._BASE}/{pack_id}/download")

    def rate(self, pack_id: str, rating: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/ratings", json_payload=rating)

    def review(self, pack_id: str, review: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/reviews", json_payload=review)

    def list_reviews(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/reviews", params=params)


class RunsResource(_Base):
    """Prompt-pack runs (nested under packs)."""

    _BASE = "/api/v1/sdk/prompt-packs"

    def create(
        self,
        pack_id: str,
        payload: dict[str, Any] | None = None,
        *,
        application_id: str | None = None,
        app_integration_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict[str, Any]:
        """api/prompt_pack_runs_handlers.go PromptPackRunRequest; target ids are top-level."""
        target = _one_target(
            application_id=application_id,
            app_integration_id=app_integration_id,
            endpoint_url=endpoint_url,
        )
        body = _with_fields(payload, **dict([target])) if target else dict(payload or {})
        return self._req("POST", f"{self._BASE}/{pack_id}/runs", json_payload=body)

    def list(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/runs", params=params)

    def get(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}", params=params)

    def delete(self, run_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/runs/{run_id}")

    def stats(self, pack_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/runs/stats")

    def compare(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/runs/compare", params=params)

    def outputs(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/outputs", params=params)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Backend agent adds this endpoint; call may 404 on older servers."""
        return self._req("POST", f"{self._BASE}/runs/{run_id}/cancel")

    def trace(self, run_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/trace")

    def report(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/report", params=params)

    def add_to_pack(
        self,
        run_id: str,
        entry_ids: builtins.list[str],
        pack_id: str,
    ) -> dict[str, Any]:
        """Copy entries (outputs' ``prompt_pack_entry_id``) into ``pack_id``; ``run_id`` is unused."""
        return self._req(
            "POST", f"{self._BASE}/{pack_id}/prompts/add", json_payload={"entry_ids": entry_ids}
        )


class OutputValidationsResource(_Base):
    """Output-validations on a prompt-pack run."""

    _BASE = "/api/v1/sdk/prompt-packs"

    def create(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req(
            "POST", f"{self._BASE}/runs/{run_id}/validate-outputs", json_payload=payload
        )

    def list_for_pack(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/output-validations", params=params)

    def get(self, validation_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/output-validations/{validation_id}")

    def summary(self, validation_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/output-validations/{validation_id}/summary")

    def rca_status(self, validation_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/output-validations/{validation_id}/rca-status")

    def results_csv(self, validation_id: str) -> bytes:
        """Raw CSV bytes."""
        return self._req_bytes(
            "GET", f"{self._BASE}/output-validations/{validation_id}/results/csv"
        )

    def compare(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/validations/compare", params=params)

    def delete(self, validation_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/output-validations/{validation_id}")

    def cancel(self, validation_id: str) -> dict[str, Any]:
        """Backend agent adds this endpoint; call may 404 on older servers."""
        return self._req("POST", f"{self._BASE}/output-validations/{validation_id}/cancel")


class SessionsResource(_Base):
    """Red-team / testing sessions."""

    _BASE = "/api/v1/testing"

    def create(
        self,
        payload: dict[str, Any],
        *,
        application_id: str | None = None,
        app_integration_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict[str, Any]:
        """Target kwargs go into ``target_config`` (api/testing_handlers.go)."""
        target = _one_target(
            application_id=application_id,
            app_integration_id=app_integration_id,
            endpoint_url=endpoint_url,
        )
        body = dict(payload)
        if target:
            body["target_config"] = _with_fields(body.get("target_config"), **dict([target]))
        return self._req("POST", f"{self._BASE}/sessions", json_payload=body)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/sessions", params=params)

    def get(self, session_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/sessions/{session_id}")

    def delete(self, session_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/sessions/{session_id}")

    def create_run(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/sessions/{session_id}/runs", json_payload=payload)

    def list_runs(self, session_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/sessions/{session_id}/runs", params=params)

    def report_csv(self, session_id: str) -> bytes:
        """Raw CSV bytes."""
        return self._req_bytes("GET", f"{self._BASE}/sessions/{session_id}/report/csv")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}")

    def run_status(self, run_id: str) -> TestingRunStatus:
        """Typed run status incl. ``stop_reason``; ``is_terminal`` covers ``stopped``."""
        return TestingRunStatus.from_dict(self.get_run(run_id))

    def validate(
        self,
        input: str,
        validators: builtins.list[str],
        *,
        output: str = "",
        input_context: str = "",
        threshold: float | None = None,
    ) -> TestingValidateResult:
        """POST /testing/validate (api/testing_validate_handler.go TestingValidateRequest)."""
        body: dict[str, Any] = {
            "input": input,
            "output": output,
            "validators": validators,
            "input_context": input_context,
        }
        if threshold is not None:
            body["threshold"] = threshold
        return TestingValidateResult.from_dict(
            self._req("POST", f"{self._BASE}/validate", json_payload=body)
        )

    def run_results(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/results", params=params)

    def run_breaches(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/results/breaches", params=params)

    def get_breach(self, run_id: str, breach_id: str) -> dict[str, Any]:
        """Fetch one breach ("finding") from a run.

        No ``/findings/:id`` route exists, so walk the paginated breach list
        (api/testing_types.go ListTestingResultsResponse ``{items, pagination}``).
        """
        page, seen = 1, 0
        while True:
            payload = self.run_breaches(run_id, params={"page": page, "page-size": 100})
            rows = payload.get("items") or []
            for row in rows:
                if isinstance(row, dict) and str(row.get("id", "")) == breach_id:
                    return row
            seen += len(rows)
            if not has_more_pages(payload, page, seen):
                raise KeyError(f"breach {breach_id} not found in run {run_id}")
            page += 1

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/runs/{run_id}/cancel")


class ByovValidatorsResource(_Base):
    """Bring-your-own-validator custom validators."""

    _BASE = "/api/v1/llm/custom-validators"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """``project_id`` (required query) defaults to the client's project."""
        return self._req(
            "GET", self._BASE, params={"project_id": self._client.project_id, **(params or {})}
        )

    def get(self, validator_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{validator_id}")

    def update(self, validator_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{validator_id}", json_payload=payload)

    def delete(self, validator_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{validator_id}")

    def test(self, validator_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{validator_id}/test", json_payload=payload or {})

    def probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Test connection without saving."""
        return self._req("POST", f"{self._BASE}/test-connection", json_payload=payload)


class VulnerabilitiesResource(_Base):
    """Read-only vulnerabilities catalog + a poll-test endpoint."""

    _BASE = "/api/v1/vulnerabilities"

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def get(self, vuln_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{vuln_id}")

    def test(
        self,
        vuln_id: str,
        payload: dict[str, Any] | None = None,
        *,
        application_id: str | None = None,
        app_integration_id: str | None = None,
        project_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        """api/vulnerability_types.go VulnerabilityTestRequest; app integrations only.
        project/organization ids default to the client's, then ``whoami()``."""
        target = _one_target(application_id=application_id, app_integration_id=app_integration_id)
        body = _with_fields(payload, **dict([target])) if target else dict(payload or {})
        project_id = project_id or self._client.project_id
        if not (project_id and organization_id):
            who = self._client.whoami()
            project_id = project_id or who.get("project_id")
            organization_id = organization_id or who.get("organization_id")
        params = {"project_id": project_id, "organization_id": organization_id}
        return self._req("POST", f"{self._BASE}/{vuln_id}/test", json_payload=body, params=params)


def plan_run_target(
    application_id: str | None, app_integration_id: str | None, endpoint_url: str | None
) -> dict[str, str] | None:
    """api/test_plan_runs_create_types.go TestPlanRunTargetConfig for one target id."""
    target = _one_target(
        application_id=application_id,
        app_integration_id=app_integration_id,
        endpoint_url=endpoint_url,
    )
    if target is None:
        return None
    mode = "direct" if target[0] == "endpoint_url" else "app_integration"
    return {"execution_mode": mode, target[0]: target[1]}


class TestPlansResource(_Base):
    """Test Plans (T3) — versioned, shareable red-team plan templates.

    Gated by ``ff_test-plans_enabled_global`` server-side; a caller without
    the flag will receive the same rejection any other flag-gated route
    returns. Endpoint surface tracked verbatim against
    ``api/test_plans_routes.go`` on stage.
    """

    _BASE = "/api/v1/test-plans"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def gallery(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/gallery", params=params)

    def list_deleted(self) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/deleted")

    def options(self, ref: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/options/{ref}")

    def get(self, plan_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{plan_id}")

    def summary(self, plan_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{plan_id}/summary")

    def update(self, plan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{plan_id}", json_payload=payload)

    def delete(self, plan_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{plan_id}")

    def restore(self, plan_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{plan_id}/restore")

    def copy(self, plan_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{plan_id}/copy", json_payload=payload or {})

    def list_versions(self, plan_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{plan_id}/versions")

    def create_version(self, plan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{plan_id}/versions", json_payload=payload)

    def publish(self, plan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{plan_id}/publish", json_payload=payload)

    def generate_inputs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Kick off description-based prompt generation (TP-2244)."""
        return self._req("POST", f"{self._BASE}/generate-inputs", json_payload=payload)

    def get_generate_inputs_job(self, job_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/generate-inputs/{job_id}")


class TestPlanRunsResource(_Base):
    """Test Plan Runs (T6) — execution + report/trace/prompts.

    Creation is addressed by PLAN id (``POST /test-plans/{plan_id}/runs``);
    every other operation lives under ``/test-plan-runs/{run_id}``. Gated
    by ``ff_test-plans_enabled_global`` server-side.
    """

    _BASE = "/api/v1/test-plan-runs"
    _PLAN_BASE = "/api/v1/test-plans"

    def create(
        self,
        plan_id: str,
        payload: dict[str, Any] | None = None,
        *,
        application_id: str | None = None,
        app_integration_id: str | None = None,
        endpoint_url: str | None = None,
    ) -> dict[str, Any]:
        """Target kwargs build ``target`` (api/test_plan_runs_create_types.go)."""
        target = plan_run_target(application_id, app_integration_id, endpoint_url)
        body = _with_fields(payload, target=target) if target else dict(payload or {})
        return self._req("POST", f"{self._PLAN_BASE}/{plan_id}/runs", json_payload=body)

    def list_for_plan(self, plan_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._PLAN_BASE}/{plan_id}/runs", params=params)

    def list_deleted(self) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/deleted")

    def get(self, run_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{run_id}")

    def get_stage(self, run_id: str, stage_key: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{run_id}/stages/{stage_key}")

    def trace(self, run_id: str, prompt_ref: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{run_id}/trace", params={"prompt_ref": prompt_ref})

    def report(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{run_id}/report", params=params)

    def prompts(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{run_id}/prompts", params=params)

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{run_id}/cancel")

    def delete(self, run_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{run_id}")

    def restore(self, run_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{run_id}/restore")

    def reveal(self, run_id: str, result_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{run_id}/results/{result_id}/reveal")


class ApplicationsResource(_Base):
    """AI Systems: list via policy-management, link/unlink via dataset-backend."""

    _LINKS = "/api/v1/llm/application-links"

    def list(self) -> dict[str, Any]:
        """``{applications, count}`` (policy-management handlers_applications.go)."""
        return self._client._request_abs(
            "GET", "/api/v1/applications", base=self._client.policies_base_url
        )

    def links(self, application_id: str) -> dict[str, Any]:
        """Bucketed links (api/application_links.go ListApplicationLinks)."""
        return self._req("GET", self._LINKS, params={"application_id": application_id})

    def link(
        self, application_id: str, app_integration_id: str, role: str | None = None
    ) -> dict[str, Any]:
        """api/application_links.go createApplicationLinkRequest (app integrations only)."""
        body: dict[str, Any] = {
            "application_id": application_id,
            "integration_type": "app_integration",
            "integration_id": app_integration_id,
        }
        if role:
            body["role"] = role
        return self._req("POST", self._LINKS, json_payload=body)

    def unlink(self, application_id: str) -> builtins.list[str]:
        """Delete every link of the AI System; returns the deleted link ids."""
        listing = self.links(application_id)
        ids = [
            str(row["id"])
            for bucket in (
                "app_integrations",
                "llm_integrations",
                "rag_integrations",
                "mcp_integrations",
            )
            for row in listing.get(bucket) or []
        ]
        if not ids:
            raise KeyError(f"AI System {application_id} has no linked integration")
        for link_id in ids:
            self._req("DELETE", f"{self._LINKS}/{link_id}")
        return ids


# Re-exported; defined there to keep this module small.
from .jailbreak_resources import JailbreakResource, MrResource  # noqa: E402

__all__ = [
    "ApplicationsResource",
    "ByovValidatorsResource",
    "JailbreakResource",
    "McpTargetsResource",
    "MrResource",
    "OutputValidationsResource",
    "PacksResource",
    "RagTargetsResource",
    "RunsResource",
    "SessionsResource",
    "TargetsResource",
    "TestPlanRunsResource",
    "TestPlansResource",
    "VulnerabilitiesResource",
]
