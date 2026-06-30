"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import requests

from disseqt_logging import digest, get_logger

from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


class HTTPError(Exception):
    """HTTP error from the Disseqt API."""

    def __init__(self, status_code: int, message: str, response_body: str) -> None:
        """Initialize HTTP error.

        Args:
            status_code: HTTP status code
            message: Error message
            response_body: Truncated response body
        """
        self.status_code = status_code
        self.message = message
        self.response_body = response_body
        super().__init__(f"HTTP {status_code}: {message}")


class Client:
    """Disseqt SDK client for validator API calls."""

    def __init__(
        self,
        project_id: str,
        api_key: str,
        base_url: str = "https://api.disseqt.ai/realtime-validations",
        timeout: int = 30,
    ) -> None:
        """Initialize the Disseqt SDK client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Base URL for the Dataset API
            timeout: Request timeout in seconds
        """
        self.project_id = project_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "X-API-Key": self.api_key,
            "X-Project-Id": self.project_id,
            "Content-Type": "application/json",
        }

    def validate(
        self, request: BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator
    ) -> dict[str, Any]:
        """Validate using the specified validator.

        Args:
            request: Validator request instance

        Returns:
            Normalized validation response

        Raises:
            HTTPError: If the API request fails
            ValueError: If JSON decoding fails
        """
        # Build the URL
        url = build_validator_url(
            self.base_url,
            request.domain,
            request.slug,
            request._path_template,
        )

        # Stable, secrets-free correlation fields for every log line in this call.
        domain = getattr(request.domain, "value", str(request.domain))
        slug = request.slug

        # Get validator metadata from registry
        try:
            metadata = get_validator_metadata(request.domain, request.slug)
            request_handler = metadata.get("request_handler")
            response_handler = metadata.get("response_handler")
        except KeyError:
            # Validator not registered, use default behavior
            request_handler = None
            response_handler = None

        # Prepare the payload using custom handler or default
        if request_handler:
            payload = request_handler(request)
        else:
            payload = request.to_payload()

        # Build headers (auth headers are never logged)
        headers = self._build_headers()

        # Log the outgoing request. The payload may contain user prompts, so we
        # emit only a content-free digest of it, never the body itself.
        logger.debug(
            "validation.request",
            domain=domain,
            slug=slug,
            url=url,
            payload_digest=digest(json.dumps(payload, sort_keys=True, default=str)),
            timeout_s=self.timeout,
        )

        started = time.monotonic()
        try:
            # Make the API request
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            latency_ms = round((time.monotonic() - started) * 1000, 1)
            logger.error(
                "validation.network_error",
                domain=domain,
                slug=slug,
                latency_ms=latency_ms,
                exc_info=True,
            )
            raise HTTPError(
                status_code=0,
                message=f"Network error: {e}",
                response_body="",
            ) from e

        latency_ms = round((time.monotonic() - started) * 1000, 1)

        # Check for HTTP errors
        if not response.ok:
            # Truncate response body for error message
            body = response.text[:512] if response.text else ""
            logger.error(
                "validation.http_error",
                domain=domain,
                slug=slug,
                status=response.status_code,
                latency_ms=latency_ms,
                response_body_digest=digest(response.text or ""),
            )
            raise HTTPError(
                status_code=response.status_code,
                message="API request failed",
                response_body=body,
            )

        # Parse JSON response
        try:
            server_response_raw = response.json()
            if server_response_raw is None:
                raise ValueError("Server returned null/empty JSON response")
            server_response = cast(dict[str, Any], server_response_raw)
        except json.JSONDecodeError as e:
            logger.error(
                "validation.decode_error",
                domain=domain,
                slug=slug,
                status=response.status_code,
                latency_ms=latency_ms,
                exc_info=True,
            )
            raise ValueError(
                f"Failed to decode JSON response: {e}. Response text: {response.text[:200]}"
            ) from e

        logger.info(
            "validation.response",
            domain=domain,
            slug=slug,
            status=response.status_code,
            latency_ms=latency_ms,
        )

        # Use custom response handler or default
        if response_handler:
            result = response_handler(server_response)
            return cast(dict[str, Any], result)
        else:
            # Use default response handling (no forced normalization)
            return server_response
