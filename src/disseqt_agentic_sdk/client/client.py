"""
DisseqtAgenticClient - Main client for the SDK.

Manages configuration, transport, and buffering.
"""

import atexit

from disseqt_agentic_sdk.buffer import TraceBuffer
from disseqt_agentic_sdk.trace import DisseqtTrace
from disseqt_agentic_sdk.transport import HTTPTransport
from disseqt_agentic_sdk.utils.logging import get_logger

logger = get_logger()


class DisseqtAgenticClient:
    """
    Main SDK client - manages configuration, transport, and buffering.

    Responsibilities:
    - Store SDK configuration (org_id, project_id, endpoint, etc.)
    - Initialize transport layer
    - Manage buffering for efficient ingestion
    - Provide resource metadata
    """

    SDK_NAME = "disseqt-agentic-sdk"
    SDK_VERSION = "0.1.0"

    def __init__(
        self,
        api_key: str,
        org_id: str,
        project_id: str,
        service_name: str,
        endpoint: str = "http://localhost:8080/v1/traces",
        service_version: str = "1.0.0",
        environment: str = "production",
        user_id: str = "",
        max_batch_size: int = 100,
        flush_interval: float = 1.0,
        timeout: float = 10.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        """
        Initialize SDK client.

        Args:
            api_key: API key for authentication (required)
            org_id: Organization ID (required)
            project_id: Project ID (required)
            service_name: Service name (required)
            endpoint: Backend API endpoint URL (required, default: http://localhost:8080/v1/traces)
            service_version: Service version
            environment: Environment (required, default: production)
            user_id: Default user ID
            max_batch_size: Maximum spans per batch
            flush_interval: Flush interval in seconds
            timeout: HTTP request timeout
            max_retries: Maximum retry attempts
            verify_ssl: Whether to verify SSL certificates

        Raises:
            ValueError: If any required field is missing or empty
        """
        # Validate required fields
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required and cannot be empty")

        if not org_id or not org_id.strip():
            raise ValueError("org_id is required and cannot be empty")

        if not project_id or not project_id.strip():
            raise ValueError("project_id is required and cannot be empty")

        if not service_name or not service_name.strip():
            raise ValueError("service_name is required and cannot be empty")

        if not endpoint or not endpoint.strip():
            raise ValueError("endpoint is required and cannot be empty")

        if not environment or not environment.strip():
            raise ValueError("environment is required and cannot be empty")

        # Configuration
        self.api_key = api_key
        self.org_id = org_id
        self.project_id = project_id
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.user_id = user_id

        # Initialize transport
        self.transport = HTTPTransport(
            endpoint=endpoint,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )

        # Initialize buffer
        self.buffer = TraceBuffer(
            transport=self.transport,
            max_batch_size=max_batch_size,
            flush_interval=flush_interval,
        )

        # Register cleanup on exit
        atexit.register(self.shutdown)

        logger.info(
            "DisseqtAgenticClient initialized",
            extra={
                "service_name": self.service_name,
                "endpoint": endpoint,
                "org_id": self.org_id,
                "project_id": self.project_id,
                "max_batch_size": max_batch_size,
                "flush_interval": flush_interval,
            },
        )

    def send_trace(self, trace: DisseqtTrace) -> None:
        """
        Send a trace to the backend (buffered).

        Args:
            trace: DisseqtTrace instance
        """
        # Convert trace spans to EnrichedSpan models
        enriched_spans = trace.to_enriched_spans()

        logger.debug(
            "Sending trace to buffer",
            extra={
                "trace_id": trace.trace_id,
                "trace_name": trace.name,
                "span_count": len(enriched_spans),
            },
        )

        # Add to buffer
        self.buffer.add_spans(enriched_spans)

    def flush(self) -> None:
        """
        Flush all buffered spans to backend immediately.
        """
        logger.debug("Flushing buffered spans")
        self.buffer.flush()

    def shutdown(self) -> None:
        """
        Shutdown client - flush all buffered spans.
        """
        logger.info("Shutting down DisseqtAgenticClient")
        self.flush()
        logger.info("DisseqtAgenticClient shutdown complete")
