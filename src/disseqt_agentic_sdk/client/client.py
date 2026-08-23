"""
DisseqtAgenticClient - Main client for the SDK.

Manages configuration, transport, and buffering.
"""

import atexit

from disseqt_agentic_sdk._version import SDK_VERSION as _SDK_VERSION
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

    Realtime policies are opt-in
    ----------------------------
    The agentic SDK works fine **without** a realtime policy. Three modes:

    1. **No policy anywhere** — don't pass ``realtime_policy_id`` to the
       client or to ``start_trace``. Spans are sent with no
       ``policy.id`` resource attribute, and llm-monitoring runs them
       through its legacy ``feature_settings`` validator path (the same
       behaviour as before realtime policies existed). Use this when you
       just want observability without policy enforcement.

    2. **Client default** — pass ``realtime_policy_id=...`` to the
       constructor. Every trace this client produces carries that
       policy unless a trace overrides it. Use when one application
       runs against one policy.

    3. **Per-trace override** — pass ``realtime_policy_id=...`` on
       :func:`start_trace`. Wins over the client default for that
       trace's spans only. Use when one application runs multiple
       agents under different policies (init the client once, vary the
       policy per trace).

    """

    SDK_NAME = "disseqt-agentic-sdk"
    SDK_VERSION = _SDK_VERSION

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
        realtime_policy_id: str | None = None,
        application_id: str | None = None,
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
            application_id: Optional application UUID. When set, sent as
                the ``X-Application-Id`` request header on every trace POST.
                Kong's traces-auth plugin verifies the header against
                policy-management (checks project + org match); mismatch
                or unknown app is rejected before any span reaches
                llm-monitoring. When unset, spans go through project-only
                scope (existing behaviour) — safe default for callers that
                haven't been assigned an application id yet.
            realtime_policy_id: Optional realtime-policy UUID. When set,
                every span emitted by this client carries it as the
                ``policy.id`` resource attribute, which is the contract
                llm-monitoring's validation consumer reads to route the
                span through policy-driven evaluation (rather than
                feature_settings). Same convention as ``api.key`` — set
                once on the client, propagated to every span
                automatically.

                When ``realtime_policy_id`` is set, ``service_name`` is
                the application identifier that lands in
                ``policy_decisions.application_name`` on the dashboard
                (analogous to ``application_name`` on
                :class:`disseqt_sdk.Client`). ``service_name`` is already
                a required positional arg so this pairing is enforced
                by signature — there is no way to construct a client
                with ``realtime_policy_id`` but without ``service_name``.

        Raises:
            ValueError: If any required field is missing or empty

        """
        # Validate required fields. service_name is required regardless
        # of realtime_policy_id (it populates the OTel resource attribute
        # on every span), which means setting realtime_policy_id
        # automatically requires service_name too — same rule as
        # disseqt_sdk.Client's (realtime_policy_id ⇒ application_name)
        # check.
        if not api_key or not api_key.strip():
            raise ValueError("api_key is required and cannot be empty")

        if not project_id or not project_id.strip():
            raise ValueError("project_id is required and cannot be empty")

        if not service_name or not service_name.strip():
            raise ValueError(
                "service_name is required and cannot be empty "
                "(also identifies the application on the policies "
                "dashboard when realtime_policy_id is set)"
            )

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
        self.realtime_policy_id = realtime_policy_id
        # Normalise empty / whitespace-only to None so the transport can
        # gate on a single "is set?" check. Sending X-Application-Id: ""
        # to Kong would either fail strict verify or short-circuit to
        # project-only scope depending on the plugin's require flag —
        # cleaner to just not send the header at all.
        self.application_id = (
            application_id.strip() if application_id and application_id.strip() else None
        )

        # Nudge callers who haven't wired an application_id yet. One-shot
        # per process; silenced via `DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE=1`
        # or `logging.getLogger("disseqt_agentic_sdk").setLevel(logging.ERROR)`.
        if self.application_id is None:
            from disseqt_agentic_sdk._notices import notify_missing_application_id

            notify_missing_application_id()

        # Initialize transport
        self.transport = HTTPTransport(
            endpoint=endpoint,
            api_key=api_key,
            max_retries=max_retries,
            realtime_policy_id=realtime_policy_id,
            application_id=self.application_id,
        )

        # Initialize buffer
        self.buffer = TraceBuffer(
            transport=self.transport,
            max_batch_size=max_batch_size,
            flush_interval=flush_interval,
        )

        # Register cleanup on exit
        atexit.register(self.shutdown)

        # Defense-in-depth: never log the project_id (a sensitive identifier),
        # even though the logger would redact it. Only non-sensitive fields here.
        logger.info(
            "DisseqtAgenticClient initialized",
            extra={
                "service_name": self.service_name,
                "endpoint": endpoint,
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
