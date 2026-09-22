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
from disseqt_agentic_sdk.utils.validation import validate_header_value as _validate_header_value

logger = get_logger()


# Shared sentinel for every required-string constructor argument so a
# caller who omits it entirely gets our own ``ValueError`` (with an
# actionable message) instead of Python's stock
# ``TypeError: missing N required ... argument``. That TypeError is
# raised by CPython's C-level arg-binder *before* the constructor body
# runs, so a bare ``foo: str`` with no default never reaches our
# validation branch — the customer sees a bare parameter-name message,
# no context on why it matters or what to do next. Using ``object()``
# keeps ``None`` / ``""`` / whitespace-only as
# separately-distinguishable "explicitly-passed but empty" cases.
class _MissingSentinel:
    """Distinct sentinel type per required-string constructor argument.

    A single shared ``object()`` default caused CodeQL's dataflow to
    treat all sentinel-defaulted parameters as one aliased source:
    ``api_key`` is flagged as a password source by name, and every
    other parameter sharing the same default inherited that taint —
    so innocuous fields like ``service_name`` landed in a
    ``py/clear-text-logging-sensitive-data`` alert when logged in
    ``logger.info(...)``. Giving each parameter its own instance
    keeps their flows independent for the taint tracker without
    changing runtime behavior (the ``isinstance`` check below matches
    every instance uniformly).
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return "<missing>"


# One sentinel per required argument. Do NOT share instances — see
# ``_MissingSentinel`` for the reason.
_MISSING_API_KEY: object = _MissingSentinel()
_MISSING_SERVICE_NAME: object = _MissingSentinel()
_MISSING_APPLICATION_ID: object = _MissingSentinel()


def _reject_missing_or_empty(value: object, name: str, extra_hint: str = "") -> None:
    """Raise ValueError if ``value`` is a missing sentinel, None, empty, or whitespace.

    Every required-string constructor argument routes through this
    helper so a call like ``DisseqtAgenticClient()`` — omitting the
    positional/kwonly args entirely — raises the same actionable
    ValueError shape that ``DisseqtAgenticClient(api_key="")`` /
    ``... api_key=None)`` already produced.
    """
    if isinstance(value, _MissingSentinel) or value is None or not str(value).strip():
        msg = f"{name} is required and cannot be empty"
        if extra_hint:
            msg = f"{msg}. {extra_hint}"
        raise ValueError(msg)


class DisseqtAgenticClient:
    """
    Main SDK client - manages configuration, transport, and buffering.

    Responsibilities:
    - Store SDK configuration (api_key, endpoint, etc.)
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
        api_key: str = _MISSING_API_KEY,  # type: ignore[assignment]
        service_name: str = _MISSING_SERVICE_NAME,  # type: ignore[assignment]
        endpoint: str = "https://api.disseqt.ai/agentic-monitoring/api/v1/traces",
        service_version: str = "1.0.0",
        environment: str = "production",
        max_batch_size: int = 100,
        flush_interval: float = 1.0,
        max_retries: int = 3,
        realtime_policy_id: str | None = None,
        *,
        application_id: str = _MISSING_APPLICATION_ID,  # type: ignore[assignment]
    ):
        """
        Initialize SDK client.

        Args:
            api_key: API key for authentication (required). Kong resolves
                the owning project + org + user from this alone via
                auth-svc's validate-api-key endpoint (user_api_keys has
                UNIQUE (api_key_hash)); no project_id needs to be passed.
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
        # Validate required fields. Every check routes through
        # ``_reject_missing_or_empty`` so a call that omits the arg
        # entirely (sentinel default) raises the same actionable
        # ``ValueError`` as an explicit empty / whitespace-only value,
        # instead of Python's stock ``TypeError`` from the arg-binder.
        # service_name is required regardless of realtime_policy_id (it
        # populates the OTel resource attribute on every span), which
        # means setting realtime_policy_id automatically requires
        # service_name too — same rule as disseqt_sdk.Client's
        # (realtime_policy_id ⇒ application_name) check.
        _reject_missing_or_empty(api_key, "api_key")
        _reject_missing_or_empty(
            service_name,
            "service_name",
            "Also identifies the application on the policies "
            "dashboard when realtime_policy_id is set.",
        )
        _reject_missing_or_empty(endpoint, "endpoint")
        _reject_missing_or_empty(environment, "environment")
        # application_id is required. Kong's traces-auth plugin drops
        # every POST that arrives without a matching X-Application-Id
        # header, so silently constructing a client that will never be
        # able to deliver spans is worse than failing loudly here.
        _reject_missing_or_empty(
            application_id,
            "application_id",
            "See https://docs.disseqt.ai/docs/disseqt-sdk/agentic-observability/applications-registry "
            "for how to obtain one.",
        )

        # Configuration
        self.api_key = api_key
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
        #
        # This covers the documented construction path (this client), not
        # every possible path: project_id can also reach a header via a
        # directly-constructed DisseqtTrace/DisseqtSpan/EnrichedSpan
        # (all public classes) bypassing this client entirely, and
        # application_id/api_key can likewise bypass this client via a
        # directly-constructed HTTPTransport. transport/http.py validates
        # both at the one point every value actually passes through
        # before becoming a header, regardless of how it got there — this
        # check here is the fail-fast-and-loud layer for the common path,
        # not the only layer.
        _validate_header_value(self.api_key, "api_key")
        _validate_header_value(self.project_id, "project_id")
        _validate_header_value(self.application_id, "application_id")

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
        # accept ``client=None`` (e.g. ``@disseqt_trace`` without an
        # explicit client arg) can resolve it via ``get_client()``. Last
        # constructed client wins — matches the common single-client
        # deployment pattern; callers running multiple clients in the
        # same process should pass ``client=...`` explicitly.
        from disseqt_agentic_sdk.api.client import set_client

        set_client(self)

        # Non-sensitive fields only. api_key/application_id are secrets or
        # opaque identifiers and don't belong in structured logs.
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
