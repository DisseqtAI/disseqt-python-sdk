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
    - Store SDK configuration (project_id, endpoint, etc.)
    - Initialize transport layer
    - Manage buffering for efficient ingestion
    - Provide resource metadata

    """

    SDK_NAME = "disseqt-agentic-sdk"
    SDK_VERSION = "0.1.0"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        service_name: str,
        endpoint: str = "https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
        service_version: str = "1.0.0",
        environment: str = "production",
        max_batch_size: int = 100,
        flush_interval: float = 1.0,
        max_retries: int = 3,
    ):
        """
        Initialize SDK client.

        Args:
            api_key: API key for authentication (required)
            project_id: Project ID (required)
            service_name: Service name (required)
            endpoint: Backend API endpoint URL (required, default: https://api.disseqt.ai/agentic-monitoring/api/v1/traces)
            service_version: Service version
            environment: Environment (required, default: production)
            max_batch_size: Maximum spans per batch
            flush_interval: Flush interval in seconds
            max_retries: Maximum retry attempts

        Raises:
            ValueError: If any required field is missing or empty

        """
        # Validate required fields
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required and cannot be empty")

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
        self.project_id = project_id
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment

        # Initialize transport
        self.transport = HTTPTransport(
            endpoint=endpoint,
            api_key=api_key,
            max_retries=max_retries,
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
        Shutdown client - flush all buffered spans and stop background threads.
        """
        logger.info("Shutting down DisseqtAgenticClient")
        # Stop buffer (will flush remaining spans and stop flush thread)
        self.buffer.stop()
        logger.info("DisseqtAgenticClient shutdown complete")
