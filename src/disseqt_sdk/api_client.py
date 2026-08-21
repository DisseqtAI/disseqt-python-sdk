"""Disseqt API client for Prompt Packs and other REST endpoints."""

from __future__ import annotations

import json
from typing import Any, cast

import requests

from ._version import check_version_notice, sdk_identity_headers
from .client import HTTPError, _version_blocked_error
from .models.prompt_packs import (
    CreateRunRequest,
    GeneratePromptPackRequest,
    PaginationParams,
    PromptPackOutputValidationRequest,
)

# Kong route prefix + service path
_PROMPT_PACKS_BASE = "/sdk/prompt-packs/api/v1/sdk/prompt-packs"


class DisseqtAPIClient:
    """Disseqt API client for Prompt Packs management.

    This client provides REST methods for the full prompt-pack lifecycle:
    generation, runs, outputs, and output validations.

    Example::

        from disseqt_sdk import DisseqtAPIClient
        from disseqt_sdk.models.prompt_packs import (
            GeneratePromptPackRequest,
            PromptPackCategory,
        )

        client = DisseqtAPIClient(
            project_id="your-project-id",
            api_key="your-api-key",
            base_url="http://localhost:8000",
        )

        pack = client.generate_prompt_pack(
            GeneratePromptPackRequest(
                pack_name="Security Pack",
                pack_short_desc="AI-generated prompts for security testing",
                author="AI Generator",
                domain="Security",
                generation_type="AI",
                categories=[
                    PromptPackCategory(
                        main_category="reliability_and_safety",
                        subcategory="hate_speech",
                        prompts_count=5,
                    ),
                ],
            )
        )
    """

    def __init__(
        self,
        project_id: str,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
    ) -> None:
        """Initialize the Disseqt API client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Kong gateway base URL
            timeout: Request timeout in seconds
        """
        self.project_id = project_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests."""
        return {
            "X-API-Key": self.api_key,
            "X-Project-Id": self.project_id,
            "Content-Type": "application/json",
            **sdk_identity_headers(),
        }

    def _url(self, path: str) -> str:
        """Build a full URL from a relative path under the prompt-packs prefix."""
        return f"{self.base_url}{_PROMPT_PACKS_BASE}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request and return the parsed JSON body.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: Path relative to prompt-packs base
            json_payload: Optional JSON body for POST requests
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            HTTPError: If the API returns a non-2xx status
            ValueError: If JSON decoding fails
        """
        url = self._url(path)
        headers = self._build_headers()

        try:
            response = requests.request(
                method,
                url,
                json=json_payload,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            check_version_notice(response.headers)

            if not response.ok:
                body = response.text[:512] if response.text else ""
                blocked = _version_blocked_error(
                    response.status_code, response.headers, response.text
                )
                if blocked is not None:
                    raise blocked
                raise HTTPError(
                    status_code=response.status_code,
                    message="API request failed",
                    response_body=body,
                )

            # DELETE with 204 No Content
            if response.status_code == 204:
                return {"status": "deleted"}

            try:
                raw = response.json()
                if raw is None:
                    raise ValueError("Server returned null/empty JSON response")
                return cast(dict[str, Any], raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to decode JSON response: {e}. " f"Response text: {response.text[:200]}"
                ) from e

        except requests.RequestException as e:
            raise HTTPError(
                status_code=0,
                message=f"Network error: {e}",
                response_body="",
            ) from e

    def _get_raw(
        self, path: str, *, params: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], str]:
        """Perform a GET and return status, headers, and body text (for CSV etc.)."""
        url = self._url(path)
        headers = dict(self._build_headers())
        headers.pop("Content-Type", None)
        try:
            response = requests.request(
                "GET",
                url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            check_version_notice(response.headers)
            return response.status_code, dict(response.headers), response.text or ""
        except requests.RequestException as e:
            raise HTTPError(
                status_code=0,
                message=f"Network error: {e}",
                response_body="",
            ) from e

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_prompt_pack(self, request: GeneratePromptPackRequest) -> dict[str, Any]:
        """Generate a new prompt pack.

        Args:
            request: Prompt pack generation parameters

        Returns:
            Created prompt pack data (includes pack_id / id)
        """
        return self._request("POST", "/generate", json_payload=request.to_payload())

    def download_pack_csv(self, pack_id: str) -> str | dict[str, Any]:
        """Export prompts of a prompt pack as CSV (by pack ID).

        Calls GET /api/v1/sdk/prompt-packs/{pack_id}/download. If the server
        returns CSV (text/csv), the CSV string is returned. If it returns JSON
        (e.g. a download URL), the parsed dict is returned.

        Args:
            pack_id: ID of the prompt pack to export

        Returns:
            CSV string, or a dict if the API returns JSON (e.g. download_url)

        Raises:
            HTTPError: If the API returns a non-2xx status
        """
        status, headers, text = self._get_raw(f"/{pack_id}/download")
        if status != 200:
            blocked = _version_blocked_error(status, headers, text)
            if blocked is not None:
                raise blocked
            raise HTTPError(
                status_code=status,
                message="Download pack CSV failed",
                response_body=text[:512] if text else "",
            )
        content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if "csv" in content_type or not text.strip().startswith("{"):
            return text
        try:
            return cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            return text

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(self, pack_id: str, request: CreateRunRequest) -> dict[str, Any]:
        """Create a new run for a prompt pack.

        Args:
            pack_id: ID of the prompt pack
            request: Run creation parameters

        Returns:
            Created run data (includes run_id / id)
        """
        return self._request("POST", f"/{pack_id}/runs", json_payload=request.to_payload())

    def list_runs(
        self,
        pack_id: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """List runs for a prompt pack.

        Args:
            pack_id: ID of the prompt pack
            pagination: Optional pagination parameters

        Returns:
            List of runs
        """
        params = (pagination or PaginationParams()).to_query_params()
        return self._request("GET", f"/{pack_id}/runs", params=params)

    def get_run(
        self,
        run_id: str,
        *,
        include_outputs: bool = True,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """Get details of a specific run.

        Args:
            run_id: ID of the run
            include_outputs: Whether to include output data
            pagination: Optional pagination for outputs

        Returns:
            Run details
        """
        params = (pagination or PaginationParams()).to_query_params()
        params["include_outputs"] = str(include_outputs).lower()
        return self._request("GET", f"/runs/{run_id}", params=params)

    def delete_run(self, run_id: str) -> dict[str, Any]:
        """Delete a run.

        Args:
            run_id: ID of the run to delete

        Returns:
            Deletion confirmation
        """
        return self._request("DELETE", f"/runs/{run_id}")

    def get_run_outputs(
        self,
        run_id: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """Get outputs for a specific run.

        Args:
            run_id: ID of the run
            pagination: Optional pagination parameters

        Returns:
            Run outputs
        """
        params = (pagination or PaginationParams()).to_query_params()
        return self._request("GET", f"/runs/{run_id}/outputs", params=params)

    def get_run_outputs_csv(self, run_id: str) -> dict[str, Any]:
        """Get run outputs in CSV format.

        Args:
            run_id: ID of the run

        Returns:
            CSV data (or download URL)
        """
        return self._request("GET", f"/runs/{run_id}/outputs/csv")

    # ------------------------------------------------------------------
    # Output Validations
    # ------------------------------------------------------------------

    def create_output_validation(
        self,
        run_id: str,
        request: PromptPackOutputValidationRequest,
    ) -> dict[str, Any]:
        """Create an output validation on a run.

        Args:
            run_id: ID of the run to validate
            request: Validation run name and metric_evaluations (metric_name + category)

        Returns:
            Created validation data (includes validation_id / id)
        """
        return self._request(
            "POST",
            f"/runs/{run_id}/validate-outputs",
            json_payload=request.to_payload(),
        )

    def list_run_output_validations(self, run_id: str) -> dict[str, Any]:
        """List output validations for a specific run.

        Args:
            run_id: ID of the run

        Returns:
            List of output validations
        """
        return self._request("GET", f"/runs/{run_id}/output-validations")

    def get_output_validation(self, validation_id: str) -> dict[str, Any]:
        """Get details of a specific output validation.

        Args:
            validation_id: ID of the output validation

        Returns:
            Output validation details
        """
        return self._request("GET", f"/output-validations/{validation_id}")

    def get_output_validation_summary(self, validation_id: str) -> dict[str, Any]:
        """Get summary of an output validation.

        Args:
            validation_id: ID of the output validation

        Returns:
            Validation summary
        """
        return self._request("GET", f"/output-validations/{validation_id}/summary")

    def get_output_validation_results(
        self,
        validation_id: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """Get detailed results of an output validation.

        Args:
            validation_id: ID of the output validation
            pagination: Optional pagination parameters

        Returns:
            Validation results
        """
        params = (pagination or PaginationParams()).to_query_params()
        return self._request(
            "GET",
            f"/output-validations/{validation_id}/results",
            params=params,
        )

    def get_output_validation_grouped_outputs(
        self,
        validation_id: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """Get grouped outputs of an output validation.

        Args:
            validation_id: ID of the output validation
            pagination: Optional pagination parameters

        Returns:
            Grouped output data
        """
        params = (pagination or PaginationParams()).to_query_params()
        return self._request(
            "GET",
            f"/output-validations/{validation_id}/outputs/grouped",
            params=params,
        )

    def get_output_validation_results_csv(self, validation_id: str) -> dict[str, Any]:
        """Get output validation results in CSV format.

        Args:
            validation_id: ID of the output validation

        Returns:
            CSV data (or download URL)
        """
        return self._request("GET", f"/output-validations/{validation_id}/results/csv")

    def delete_output_validation(self, validation_id: str) -> dict[str, Any]:
        """Delete an output validation.

        Args:
            validation_id: ID of the output validation to delete

        Returns:
            Deletion confirmation
        """
        return self._request("DELETE", f"/output-validations/{validation_id}")

    def list_pack_output_validations(
        self,
        pack_id: str,
        pagination: PaginationParams | None = None,
    ) -> dict[str, Any]:
        """List all output validations for a prompt pack.

        Args:
            pack_id: ID of the prompt pack
            pagination: Optional pagination parameters

        Returns:
            List of output validations for the pack
        """
        params = (pagination or PaginationParams()).to_query_params()
        return self._request(
            "GET",
            f"/{pack_id}/output-validations",
            params=params,
        )
