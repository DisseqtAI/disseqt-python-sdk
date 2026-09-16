"""Resource clients for the Disseqt dataset-backend REST surface.

Additive to :class:`~disseqt_sdk.api_client.DisseqtAPIClient`: mounted as
lazy attribute-style resources (``client.targets``, ``client.packs``,
``client.runs``, ``client.output_validations``, ``client.rag_validations``,
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

if TYPE_CHECKING:
    from .api_client import DisseqtAPIClient


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
        """Parse a curl command into a target-integration preview."""
        return self._req("POST", f"{self._BASE}/parse-curl", json_payload={"curl": curl_text})


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

    _BASE = "/api/v1/prompt-packs"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

    def get(self, pack_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}")

    def update(self, pack_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/{pack_id}", json_payload=payload)

    def delete(self, pack_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/{pack_id}")

    def restore(self, pack_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/restore")

    def prompts(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/prompts", params=params)

    def add_prompts(self, pack_id: str, prompts: builtins.list[dict[str, Any]]) -> dict[str, Any]:
        """Bulk-add prompts to a pack. Uses the /prompts/bulk endpoint."""
        return self._req(
            "POST", f"{self._BASE}/{pack_id}/prompts/bulk", json_payload={"prompts": prompts}
        )

    def add_prompt(self, pack_id: str, prompt: dict[str, Any]) -> dict[str, Any]:
        """Add a single prompt (uses /prompts/add)."""
        return self._req("POST", f"{self._BASE}/{pack_id}/prompts/add", json_payload=prompt)

    def duplicate(self, pack_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/duplicate", json_payload=payload or {})

    def publish(self, pack_id: str) -> dict[str, Any]:
        # Backend PATCHes at /api/v1/prompt-packs/:id/publish (server.go:2237)
        # and /api/v1/sdk/prompt-packs/:id/publish (server.go:2460).
        return self._req("PATCH", f"{self._BASE}/{pack_id}/publish")

    def unpublish(self, pack_id: str) -> dict[str, Any]:
        # See publish(); PATCH at server.go:2238 and :2461.
        return self._req("PATCH", f"{self._BASE}/{pack_id}/unpublish")

    def import_status(self, pack_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/import-status")

    def download(self, pack_id: str) -> bytes:
        """Download pack as CSV. Returns raw bytes (CSV content)."""
        return self._req_bytes("GET", f"{self._BASE}/{pack_id}/download")

    def rate(self, pack_id: str, rating: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/ratings", json_payload=rating)

    def review(self, pack_id: str, review: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/reviews", json_payload=review)

    def list_reviews(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/reviews", params=params)

    def upload_session_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Start a chunked upload session (POST /upload/sessions)."""
        return self._req("POST", f"{self._BASE}/upload/sessions", json_payload=payload)

    def upload_session_chunk(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req(
            "POST", f"{self._BASE}/upload/sessions/{session_id}/chunks", json_payload=payload
        )

    def upload_session_complete(self, session_id: str) -> dict[str, Any]:
        # Backend route: POST /api/v1/prompt-packs/upload/sessions/:sid/complete
        # (server.go:2196, handler completeChunkedUpload). The prior /finish
        # suffix was never registered.
        return self._req("POST", f"{self._BASE}/upload/sessions/{session_id}/complete")


class RunsResource(_Base):
    """Prompt-pack runs (nested under packs)."""

    _BASE = "/api/v1/prompt-packs"

    def create(self, pack_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/runs", json_payload=payload)

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

    def retrieval_traces(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/retrieval-traces", params=params)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Backend agent adds this endpoint; call may 404 on older servers."""
        return self._req("POST", f"{self._BASE}/runs/{run_id}/cancel")

    def trace(self, run_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/trace")

    def report(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/report", params=params)

    # NOTE: reveal_output removed — no backend route exists for prompt-pack
    # runs. The only reveal endpoint is on test-plan runs
    # (test_plan_runs_routes.go:80), already exposed via
    # TestPlanRunsResource.reveal(). If prompt-pack run outputs also need a
    # reveal path, a new backend endpoint must land first.

    def add_to_pack(
        self,
        run_id: str,
        output_ids: builtins.list[str],
        pack_id: str,
    ) -> dict[str, Any]:
        """Add selected run outputs to a prompt pack.

        POST /prompt-packs/{pack_id}/prompts/add with body
        {"run_id": ..., "output_ids": [...]}.
        """
        return self._req(
            "POST",
            f"{self._BASE}/{pack_id}/prompts/add",
            json_payload={"run_id": run_id, "output_ids": output_ids},
        )


class OutputValidationsResource(_Base):
    """Output-validations on a prompt-pack run."""

    _BASE = "/api/v1/prompt-packs"

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

    def results_csv(self, validation_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/output-validations/{validation_id}/results/csv")

    def compare(self, pack_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/{pack_id}/validations/compare", params=params)

    def delete(self, validation_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/output-validations/{validation_id}")

    def cancel(self, validation_id: str) -> dict[str, Any]:
        """Backend agent adds this endpoint; call may 404 on older servers."""
        return self._req("POST", f"{self._BASE}/output-validations/{validation_id}/cancel")


class RagValidationsResource(_Base):
    """RAG validations on a prompt-pack run."""

    _BASE = "/api/v1/prompt-packs"

    def create(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/runs/{run_id}/rag-validate", json_payload=payload)

    def list_for_run(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/rag-validations", params=params)

    def get(self, validation_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/rag-validations/{validation_id}")

    def cancel(self, validation_id: str) -> dict[str, Any]:
        """Backend agent adds this endpoint; call may 404 on older servers."""
        return self._req("POST", f"{self._BASE}/rag-validations/{validation_id}/cancel")


class SessionsResource(_Base):
    """Red-team / testing sessions."""

    _BASE = "/api/v1/testing"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/sessions", json_payload=payload)

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

    def report_csv(self, session_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/sessions/{session_id}/report/csv")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}")

    def run_results(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/results", params=params)

    def run_breaches(self, run_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/runs/{run_id}/results/breaches", params=params)

    def get_breach(self, run_id: str, breach_id: str) -> dict[str, Any]:
        """Fetch one breach ("finding") from a run.

        The dataset-backend has no standalone ``/findings/:id`` endpoint —
        it returns the whole breach list on ``/runs/:id/results/breaches``.
        We client-side filter by id (matching the field the backend uses:
        one of ``id`` / ``breach_id`` / ``finding_id``).
        """
        payload: Any = self.run_breaches(run_id)
        if isinstance(payload, builtins.list):
            rows: builtins.list[Any] = payload
        elif isinstance(payload, dict):
            data = payload.get("data") or payload.get("breaches") or []
            rows = data if isinstance(data, builtins.list) else []
        else:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("id", "breach_id", "finding_id"):
                if str(row.get(key, "")) == breach_id:
                    return row
        raise KeyError(f"breach {breach_id} not found in run {run_id}")

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/runs/{run_id}/cancel")


class JailbreakResource(_Base):
    """Jailbreak technique + prompt CRUD and generated-prompt controls.

    Routes registered under ``/api/v1/jailbreak/*`` — see
    ``dataset-backend api/jailbreak_routes.go`` on stage.
    """

    _BASE = "/api/v1/jailbreak"

    # Techniques
    def create_technique(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/techniques", json_payload=payload)

    def get_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques/{technique_id}")

    def list_techniques(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/techniques", params=params)

    def update_technique(self, technique_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/techniques/{technique_id}", json_payload=payload)

    def delete_technique(self, technique_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/techniques/{technique_id}")

    # Prompts
    def create_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/prompts", json_payload=payload)

    def get_prompt(self, prompt_id: str) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/prompts/{prompt_id}")

    def list_prompts(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", f"{self._BASE}/prompts", params=params)

    def update_prompt(self, prompt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("PATCH", f"{self._BASE}/prompts/{prompt_id}", json_payload=payload)

    def delete_prompt(self, prompt_id: str) -> dict[str, Any]:
        return self._req("DELETE", f"{self._BASE}/prompts/{prompt_id}")

    # Generated prompts
    def mark_generated_prompt_successful(self, generated_prompt_id: str) -> dict[str, Any]:
        """PATCH /generated-prompts/:id/success — mark a generated prompt successful."""
        return self._req("PATCH", f"{self._BASE}/generated-prompts/{generated_prompt_id}/success")


class ByovValidatorsResource(_Base):
    """Bring-your-own-validator custom validators."""

    _BASE = "/api/v1/llm/custom-validators"

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", self._BASE, json_payload=payload)

    def list(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._req("GET", self._BASE, params=params)

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

    def test(self, vuln_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{vuln_id}/test", json_payload=payload)


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

    def create(self, plan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._req("POST", f"{self._PLAN_BASE}/{plan_id}/runs", json_payload=payload)

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


__all__ = [
    "ByovValidatorsResource",
    "JailbreakResource",
    "McpTargetsResource",
    "OutputValidationsResource",
    "PacksResource",
    "RagTargetsResource",
    "RagValidationsResource",
    "RunsResource",
    "SessionsResource",
    "TargetsResource",
    "TestPlanRunsResource",
    "TestPlansResource",
    "VulnerabilitiesResource",
]
