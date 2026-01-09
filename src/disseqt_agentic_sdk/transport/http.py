"""
HTTP transport for sending traces/spans to the backend API.
"""

import json
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger()


class HTTPTransport:
    """
    HTTP transport for sending spans to the backend API.

    Handles:
    - HTTP POST requests
    - Retry logic
    - Error handling
    - Request formatting
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """
        Initialize HTTP transport.

        Args:
            endpoint: Backend API endpoint URL (e.g., "http://localhost:8080/v1/traces")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            verify_ssl: Whether to verify SSL certificates
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def send_spans(self, spans: list[EnrichedSpan]) -> bool:
        """
        Send spans to the backend API in Custom Format.

        Formats spans as Custom Format payload with resource and traces.

        Args:
            spans: List of EnrichedSpan objects to send

        Returns:
            bool: True if successful, False otherwise
        """
        if not spans:
            return True

        # Group spans by trace_id
        traces_dict: dict[str, list[dict[str, Any]]] = {}
        resource_attrs: dict[str, Any] = {}

        for span in spans:
            trace_id_str = (
                str(span.trace_id) if hasattr(span.trace_id, "__str__") else str(span.trace_id)
            )

            if trace_id_str not in traces_dict:
                traces_dict[trace_id_str] = []

            # Convert EnrichedSpan to Custom Format span
            span_dict = span.to_dict()

            # Convert to Custom Format span structure
            custom_span = {
                "traceId": span_dict["trace_id"],
                "spanId": span_dict["span_id"],
                "parentSpanId": span_dict.get("parent_span_id") or "",
                "name": span_dict["name"],
                "spanKind": span_dict["kind"],
                "startTimeMs": span_dict["start_time_unix_nano"] // 1_000_000,  # Convert ns to ms
                "endTimeMs": span_dict["end_time_unix_nano"] // 1_000_000,  # Convert ns to ms
                "status": span_dict["status_code"],
            }

            # Parse attributes_json and organize into genAi/agentic sections
            attributes = json.loads(span_dict.get("attributes_json", "{}"))

            # Extract agentic attributes
            agentic_attrs = {}
            gen_ai_attrs = {}

            for key, value in attributes.items():
                if key.startswith("agentic."):
                    agentic_attrs[key] = value
                elif key.startswith("gen_ai."):
                    gen_ai_attrs[key] = value

            if agentic_attrs:
                custom_span["agentic"] = agentic_attrs
            if gen_ai_attrs:
                custom_span["genAi"] = gen_ai_attrs

            traces_dict[trace_id_str].append(custom_span)

            # Extract resource attributes from first span
            if not resource_attrs:
                resource_attrs = {
                    "service.name": span_dict.get("service_name", ""),
                    "service.version": span_dict.get("service_version", ""),
                    "deployment.environment": span_dict.get("environment", ""),
                    "org.id": span_dict.get("org_id", ""),
                    "project.id": span_dict.get("project_id", ""),
                    "ingestion_url": self.endpoint,  # Add endpoint URL to resource attributes
                }
                # Add API key to resource attributes if provided
                if self.api_key:
                    resource_attrs["api.key"] = self.api_key

        # Build Custom Format payload
        traces = []
        for trace_id, span_list in traces_dict.items():
            traces.append(
                {
                    "traceId": trace_id,
                    "spans": span_list,
                }
            )

        payload = {
            "resource": {"attributes": resource_attrs},
            "traces": traces,
        }

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()
            logger.info(
                "Successfully sent spans to backend",
                extra={
                    "endpoint": self.endpoint,
                    "span_count": len(spans),
                    "trace_count": len(traces),
                },
            )
            return True
        except requests.exceptions.RequestException as e:
            logger.error(
                "Failed to send spans to backend",
                extra={
                    "endpoint": self.endpoint,
                    "span_count": len(spans),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            return False

    def send_trace(self, trace_spans: list[EnrichedSpan]) -> bool:
        """
        Send trace spans (alias for send_spans for compatibility).

        Args:
            trace_spans: List of EnrichedSpan objects from a trace

        Returns:
            bool: True if successful, False otherwise
        """
        return self.send_spans(trace_spans)
