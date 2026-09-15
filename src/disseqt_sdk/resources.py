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
        """Test a saved integration by id."""
        return self._req("POST", f"{self._BASE}/{target_id}/test", json_payload=payload or {})

    def probe(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Test connection without saving (dry run)."""
        return self._req("POST", f"{self._BASE}/test-connection", json_payload=payload)

    def parse_curl(self, curl_text: str) -> dict[str, Any]:
        """Parse a curl command into a target-integration preview."""
        return self._req("POST", f"{self._BASE}/parse-curl", json_payload={"curl": curl_text})


class RagTargetsResource(_Base):
    """RAG targets — separate table + endpoints."""

    _BASE = "/api/v1/rag-integrations"

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

    _BASE = "/api/v1/mcp-integrations"

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
        return self._req("POST", f"{self._BASE}/{pack_id}/publish")

    def unpublish(self, pack_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/{pack_id}/unpublish")

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

    def upload_session_finish(self, session_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/upload/sessions/{session_id}/finish")


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

    def reveal_output(self, run_id: str, output_id: str) -> dict[str, Any]:
        """Reveal a masked run output (POST /runs/{run_id}/results/{output_id}/reveal)."""
        return self._req("POST", f"{self._BASE}/runs/{run_id}/results/{output_id}/reveal")

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

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._req("POST", f"{self._BASE}/runs/{run_id}/cancel")


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


__all__ = [
    "ByovValidatorsResource",
    "McpTargetsResource",
    "OutputValidationsResource",
    "PacksResource",
    "RagTargetsResource",
    "RagValidationsResource",
    "RunsResource",
    "SessionsResource",
    "TargetsResource",
    "VulnerabilitiesResource",
]
