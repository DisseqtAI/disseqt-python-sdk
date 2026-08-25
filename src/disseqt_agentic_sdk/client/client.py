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


# The two things that actually break sending an application_id as an
# HTTP header, checked directly against the real failure modes rather
# than an assumed character blacklist:
#
#   1. Embedded \r / \n — `requests` raises InvalidHeader for these
#      (header-injection risk); everything else in the C0-control /
#      DEL range (tab, NUL, bell, ...) is, in practice, sent over the
#      wire by `requests` without complaint, so it isn't checked here.
#   2. Anything outside the Latin-1 range — `http.client.putheader`
#      encodes header values as Latin-1 and raises an uncaught
#      UnicodeEncodeError for anything outside it (an emoji, most
#      non-Latin scripts, a copy-pasted smart quote). That exception
#      isn't a `requests.exceptions.RequestException`, so nothing
#      downstream catches it — it can kill the background flush thread
#      outright. Checked directly via an encode attempt rather than an
#      enumerated character set, so it can't drift out of sync with
#      what actually breaks. TP-2128 round-3 senior review P1 #1.1
#      (the original C0-control blacklist rejected harmless characters
#      like tab while letting the actual crash-risk class through).
_APPLICATION_ID_DISALLOWED_LINE_BREAKS = frozenset("\r\n")


def _validate_application_id(value: str) -> None:
    """
    Raise ``ValueError`` immediately on a malformed ``application_id``.

    The X-Application-Id header value is opaque to us (Kong's traces-
    auth plugin verifies it against policy-management) but must not
    carry characters that would break sending it as an HTTP header on
    every send. Combined with retain-on-failure (TP-2128 round-2 P1
    #1.2) an un-caught failure here would otherwise retry forever
    against a value that will never succeed. Fail loudly here instead.
    TP-2128 round-2 P2 #2.3, tightened by round-3 P1 #1.1 to match what
    ``requests``/``http.client`` actually reject rather than an assumed
    control-character list. Caller is responsible for the emptiness
    check — ``application_id`` is a required field so ``""`` / whitespace
    is rejected at construction with a different, more actionable
    message than the header-encoding one below.
    """
    if _APPLICATION_ID_DISALLOWED_LINE_BREAKS & set(value):
        raise ValueError(
            "application_id contains a carriage return or newline character, "
            "which requests rejects as a header-injection risk on every send. "
            "Use a plain token (typically a UUID)."
        )
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(
            f"application_id contains a character outside the Latin-1 range "
            f"({exc}), which would crash HTTP header encoding on every send. "
            f"Use a plain ASCII token (typically a UUID)."
        ) from exc


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
        *,
        application_id: str,
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
            application_id: **Required.** Application UUID sent as the
                ``X-Application-Id`` request header on every trace POST.
                Kong's traces-auth plugin verifies the header against
                policy-management (checks project + org match); a
                mismatch, unknown app, or missing header is rejected
                before any span reaches llm-monitoring. Passed as a
                keyword-only argument (``application_id=...``) so a
                caller can't accidentally miss it by position; missing
                / empty / whitespace-only raises ``ValueError``
                immediately at construction rather than silently
                dropping every telemetry POST at flush time.
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

        # application_id is required. Reject missing / empty / whitespace-
        # only with the same message shape as the other required-field
        # checks. Kong's traces-auth plugin drops every POST that arrives
        # without a matching X-Application-Id header, so silently
        # constructing a client that will never be able to deliver spans
        # is worse than failing loudly here.
        if not application_id or not application_id.strip():
            raise ValueError(
                "application_id is required and cannot be empty. "
                "See https://docs.disseqt.ai/docs/disseqt-sdk/agentic-observability/applications-registry "
                "for how to obtain one."
            )

        # Configuration
        self.api_key = api_key
        self.project_id = project_id
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment
        self.realtime_policy_id = realtime_policy_id
        self.application_id = application_id.strip()
        # Fail-fast on a value whose characters would break HTTP header
        # encoding at send time (CRLF injection risk, non-Latin-1 codepoints
        # that http.client.putheader raises on). Combined with retain-on-
        # failure in the buffer, an unvalidated bad value would fail every
        # flush forever without ever reaching the CRITICAL auth-failure
        # banner. TP-2128 round-2 P2 #2.3 + round-3 P1 #1.1.
        _validate_application_id(self.application_id)

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

        # Auto-register as the process-default client so helpers that
        # accept ``client=None`` (e.g. ``@trace_function`` without an
        # explicit client arg) can resolve it via ``get_client()``. Last
        # constructed client wins — matches the common single-client
        # deployment pattern; callers running multiple clients in the
        # same process should pass ``client=...`` explicitly.
        from disseqt_agentic_sdk.api.client import set_client

        set_client(self)

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
