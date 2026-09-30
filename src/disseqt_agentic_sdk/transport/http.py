"""
HTTP transport for sending traces/spans to the backend API.
"""

import json
import os
import sys
import threading
import time
from typing import Any, NamedTuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from disseqt_agentic_sdk.models.span import EnrichedSpan
from disseqt_agentic_sdk.utils.logging import get_logger
from disseqt_agentic_sdk.utils.validation import validate_header_value

logger = get_logger()

# Auth-failure escalation channel — bypasses the disseqt_logging silent-
# by-default gate so an unconfigured process still gets an operator-
# visible message on 401/403. Set DISSEQT_SDK_SILENCE_AUTH_STDERR=1 to
# opt out (dashboards / structured-logging setups that route the
# CRITICAL log line themselves). TP-2128 round-2 P1 #1.3.
_SILENCE_AUTH_STDERR = os.environ.get("DISSEQT_SDK_SILENCE_AUTH_STDERR", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# Latch state for the stderr escalation. A bad API key fails every flush
# (once a second by default), so an unthrottled write floods stderr. Emit on
# the first failure, then at most once per exponentially growing interval;
# any successful send resets the latch so a later regression is loud again.
_AUTH_STDERR_FIRST_INTERVAL_S = 60.0
_AUTH_STDERR_MAX_INTERVAL_S = 3600.0
_auth_stderr_lock = threading.Lock()
_auth_stderr_next_at: float | None = None  # monotonic time of next allowed emit
_auth_stderr_interval: float = _AUTH_STDERR_FIRST_INTERVAL_S


def _reset_auth_failure_latch() -> None:
    """Re-arm the stderr auth-failure message (called on first success)."""
    global _auth_stderr_next_at, _auth_stderr_interval
    with _auth_stderr_lock:
        _auth_stderr_next_at = None
        _auth_stderr_interval = _AUTH_STDERR_FIRST_INTERVAL_S


def _write_auth_failure_to_stderr(status_code: int, endpoint: str) -> None:
    """
    Direct stderr write for 401/403 responses. Bypasses the logging
    stack entirely so the message survives a fresh unconfigured
    process (``disseqt_logging`` silences every level until
    ``configure()`` / ``DISSEQT_LOG_LEVEL`` is set). The
    logger.critical call still runs alongside this for anyone with
    structured logging configured.

    Latched: written on the first failure, then at most once per
    exponentially growing interval (60s, 120s, ... capped at 1h), and
    re-armed by the next successful send. Thread-safe.
    """
    global _auth_stderr_next_at, _auth_stderr_interval
    if _SILENCE_AUTH_STDERR:
        return
    with _auth_stderr_lock:
        now = time.monotonic()
        if _auth_stderr_next_at is not None and now < _auth_stderr_next_at:
            return
        _auth_stderr_next_at = now + _auth_stderr_interval
        _auth_stderr_interval = min(_auth_stderr_interval * 2, _AUTH_STDERR_MAX_INTERVAL_S)
    try:
        sys.stderr.write(
            f"[disseqt-agentic-sdk] CRITICAL: ingest auth rejected "
            f"({status_code}) at {endpoint} — spans will keep failing "
            f"until credentials are fixed. Check DISSEQT_API_KEY / "
            f"X-Application-Id and the traces-auth policy binding. "
            f"Repeats are throttled. "
            f"Silence this message with DISSEQT_SDK_SILENCE_AUTH_STDERR=1.\n"
        )
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — never let a stderr write crash the caller
        pass


# Per-group send outcomes.
_OK = "ok"
_RETRY = "retry"
_PERMANENT = "permanent"


class SendResult(NamedTuple):
    """Outcome of a send: spans worth retrying vs. spans the server rejected for good."""

    retryable: list[EnrichedSpan]
    permanent: list[EnrichedSpan]


def _is_permanent_status(status_code: int | None) -> bool:
    """4xx other than 408 (timeout) / 429 (rate limit) will not succeed on retry."""
    return status_code is not None and 400 <= status_code < 500 and status_code not in (408, 429)


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
        realtime_policy_id: str | None = None,
        application_id: str | None = None,
        project_id: str | None = None,
    ):
        """
        Initialize HTTP transport.

        Args:
            endpoint: Backend API endpoint URL (e.g., "http://localhost:8080/v1/traces")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            verify_ssl: Whether to verify SSL certificates
            realtime_policy_id: Optional realtime-policy UUID stamped
                onto every payload's ``resource.attributes['policy.id']``
                field. llm-monitoring's span consumer reads this to route
                the span through policy-driven evaluation. Omitted from
                the payload when None.
            application_id: Optional application UUID. When set, sent as
                the ``X-Application-Id`` request header on every POST.
                Kong's traces-auth plugin verifies the header against
                policy-management before forwarding. When None, the
                header is not sent (project-only scope, backwards-compat).
            project_id: The owning client's project. When set, spans whose own
                ``project_id`` is non-empty and different are never sent: a
                client may only deliver its own project's spans. They are
                reported as permanent failures (dropped, not retried) and
                logged once per foreign project id.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.realtime_policy_id = realtime_policy_id
        self.application_id = application_id
        self.project_id = project_id or None
        self._foreign_projects_logged: set[str] = set()
        # api_key/application_id don't vary after construction (unlike
        # project_id, validated per-send below in _send_group), so fail
        # fast here. This is the layer that catches a value that reached
        # HTTPTransport WITHOUT going through DisseqtAgenticClient's own
        # (earlier, friendlier) validation -- e.g. HTTPTransport
        # constructed directly, as this module's own test suite does.
        if self.api_key:
            validate_header_value(self.api_key, "api_key")
        if self.application_id:
            validate_header_value(self.application_id, "application_id")

        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            # urllib3 excludes POST from retries by default, which made the
            # status_forcelist above a no-op for this POST-only transport.
            # Span ingestion is idempotent on span_id server-side, so retrying
            # POST is safe.
            allowed_methods=frozenset({"POST"}) | Retry.DEFAULT_ALLOWED_METHODS,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def send_spans(self, spans: list[EnrichedSpan]) -> bool:
        """
        Send spans to the backend API in Custom Format.

        Backwards-compatible wrapper around ``send_spans_with_failures``
        that collapses per-group outcomes into a single ``all_ok`` bool.
        Prefer ``send_spans_with_failures`` for callers that need to
        distinguish which spans failed (e.g. the retry buffer, which
        must not re-POST spans a partial-failure batch already
        delivered — TP-2128 round-2 P1 #1.2).

        Args:
            spans: List of EnrichedSpan objects to send

        Returns:
            bool: True if every group sent successfully, False otherwise
        """
        result = self.send_spans_classified(spans)
        return not result.retryable and not result.permanent

    def send_spans_with_failures(self, spans: list[EnrichedSpan]) -> list[EnrichedSpan]:
        """
        Send spans and return the ones that failed to deliver *and are worth retrying*.

        Spans the server permanently rejected (4xx other than 408/429, or an
        unsendable header value) are dropped here -- logged once per group --
        and NOT returned, so the retry buffer doesn't re-POST them forever.
        Use ``send_spans_classified`` to see them.

        Spans are grouped by full resource identity (project, service name /
        version, environment and ``realtime_policy_id``, with the client
        default policy as fallback) and each distinct group is sent as its own
        HTTP POST, because the payload carries a single ``resource`` block.
        Single-resource batches still produce a single POST.

        Returns:
            list[EnrichedSpan]: exactly the retryable failures. Empty list on
            full success. The retry buffer uses this so successfully-delivered
            groups aren't re-POSTed on the next flush.
        """
        return self.send_spans_classified(spans).retryable

    def send_spans_classified(self, spans: list[EnrichedSpan]) -> SendResult:
        """Like ``send_spans_with_failures`` but also reports permanently-rejected spans."""
        retryable: list[EnrichedSpan] = []
        permanent: list[EnrichedSpan] = []
        if not spans:
            return SendResult(retryable, permanent)

        # A client may only send its own project's spans. Anything else is a
        # caller bug (or an attempt to write into another tenant's data), so
        # it is refused here rather than stamped with, or rewritten to, some
        # other project. Empty project_id is not "another project".
        if self.project_id:
            own: list[EnrichedSpan] = []
            for span in spans:
                span_project = str(span.to_dict().get("project_id", "") or "")
                if span_project and span_project != self.project_id:
                    permanent.append(span)
                    if span_project not in self._foreign_projects_logged:
                        self._foreign_projects_logged.add(span_project)
                        logger.error(
                            "Refusing to send span(s): project_id does not match this "
                            "client's project. A client can only send its own project's "
                            "spans; these are dropped.",
                            extra={"span_project_id_len": len(span_project)},
                        )
                else:
                    own.append(span)
            spans = own
            if not spans:
                return SendResult(retryable, permanent)

        # Bucket by full resource identity: the payload has ONE resource
        # block (taken from the first span), so grouping by policy alone
        # would stamp the first span's project/service/env on everyone.
        groups: dict[tuple[str, str, str, str, str], list[EnrichedSpan]] = {}
        for span in spans:
            pid = getattr(span, "realtime_policy_id", "") or self.realtime_policy_id or ""
            d = span.to_dict()
            key = (
                str(d.get("project_id", "") or ""),
                str(d.get("service_name", "") or ""),
                str(d.get("service_version", "") or ""),
                str(d.get("environment", "") or ""),
                pid,
            )
            groups.setdefault(key, []).append(span)

        for key, group in groups.items():
            outcome = self._send_group(key[4], group)
            if outcome == _RETRY:
                retryable.extend(group)
            elif outcome == _PERMANENT:
                permanent.extend(group)
        return SendResult(retryable, permanent)

    def close(self) -> None:
        """Release the underlying HTTP session (connection pool). Idempotent."""
        try:
            self.session.close()
        except Exception:  # noqa: BLE001 — closing must never raise into shutdown
            logger.debug("HTTPTransport session close failed", exc_info=True)

    def _send_group(self, policy_id: str, spans: list[EnrichedSpan]) -> str:
        """Send one resource-block's worth of spans; returns _OK / _RETRY / _PERMANENT."""
        # Group by trace_id within the policy bucket so the wire shape
        # stays {resource, traces: [{traceId, spans}]}.
        traces_dict: dict[str, list[dict[str, Any]]] = {}
        resource_attrs: dict[str, Any] = {}

        for span in spans:
            trace_id_str = (
                str(span.trace_id) if hasattr(span.trace_id, "__str__") else str(span.trace_id)
            )

            if trace_id_str not in traces_dict:
                traces_dict[trace_id_str] = []

            span_dict = span.to_dict()

            custom_span = {
                "traceId": span_dict["trace_id"],
                "spanId": span_dict["span_id"],
                "parentSpanId": span_dict.get("parent_span_id") or "",
                "name": span_dict["name"],
                "spanKind": span_dict["kind"],
                "startTimeMs": span_dict["start_time_unix_nano"] // 1_000_000,
                "endTimeMs": span_dict["end_time_unix_nano"] // 1_000_000,
                "status": span_dict["status_code"],
            }

            attributes = json.loads(span_dict.get("attributes_json", "{}"))
            if attributes:
                custom_span["attributes"] = attributes

            traces_dict[trace_id_str].append(custom_span)

            if not resource_attrs:
                resource_attrs = {
                    "service.name": span_dict.get("service_name", ""),
                    "service.version": span_dict.get("service_version", ""),
                    "deployment.environment": span_dict.get("environment", ""),
                    "project.id": span_dict.get("project_id", ""),
                    "ingestion_url": self.endpoint,
                    # The API key is deliberately NOT placed in the body: it
                    # authenticates via the X-Api-Key header only. A body copy
                    # would be stored/logged as trace resource metadata.
                }
                # policy.id is the OTel-style resource attribute
                # llm-monitoring's validation consumer keys on to route
                # the span through policy-driven evaluation. Only emit
                # when set so non-policy callers get bit-for-bit the
                # same payload as before.
                if policy_id:
                    resource_attrs["policy.id"] = policy_id

        traces = [
            {"traceId": trace_id, "spans": span_list} for trace_id, span_list in traces_dict.items()
        ]

        payload = {
            "resource": {"attributes": resource_attrs},
            "traces": traces,
        }
        headers = {"Content-Type": "application/json"}
        # X-Application-Id: verified by Kong's traces-auth plugin against
        # policy-management before the request reaches llm-monitoring.
        # Only set when the client was constructed with a non-empty
        # application_id — never send an empty value.
        if self.application_id:
            headers["X-Application-Id"] = self.application_id
        # X-Api-Key / X-Project-Id: header-first path for Kong's traces-auth
        # plugin. Historically the SDK put both under resource.attributes,
        # so the plugin had to buffer + JSON-parse the full body just to
        # authenticate — that's what put the auth path next to the 8 KB
        # spool-to-disk bug in TP-2314. Sending them as headers lets a
        # header-aware plugin version decide auth before touching a single
        # byte of body. resource.attributes below stay populated so this
        # SDK still works against plugin versions that only look in the
        # body — old-server / new-client is safe.
        #
        # X-Realtime-Policy-Id is deliberately NOT sent as a header: as of
        # this change nothing server-side reads it (traces-auth's header
        # extractor only checks X-Api-Key/X-Project-Id), so shipping it
        # now would only add exposure (see allow_redirects below) with no
        # matching benefit. Add it in the same change that adds its
        # consumer, not ahead of it. resource.attributes["policy.id"]
        # above is unaffected — that's the existing body-side contract.
        project_id = resource_attrs.get("project.id")
        if project_id:
            # Unlike api_key/application_id (validated once in __init__,
            # above), project_id varies per group/span and can reach here
            # via more than one directly-constructible public class
            # (DisseqtTrace, DisseqtSpan, EnrichedSpan) that bypasses
            # DisseqtAgenticClient's own validation entirely. This is the
            # one point every value actually passes through before
            # becoming a header, so validate it here too -- and treat a
            # bad value as this group's send failure (logged, retried
            # like any other failure) rather than letting an uncaught
            # UnicodeEncodeError propagate into the caller's own thread
            # (add_span's synchronous flush path isn't covered by
            # buffer.py's background-thread try/except).
            try:
                validate_header_value(project_id, "project_id")
            except ValueError as exc:
                logger.error(
                    "Dropping span group: project_id is not safe to send " "as an HTTP header (%s)",
                    exc,
                    extra={
                        "endpoint": self.endpoint,
                        "span_count": len(spans),
                        "policy_id": policy_id or None,
                    },
                )
                # Retrying the same bad value can never succeed.
                return _PERMANENT
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        if project_id:
            headers["X-Project-Id"] = project_id
        try:
            response = self.session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
                verify=self.verify_ssl,
                # requests' default (True) strips only `Authorization` on a
                # cross-host redirect -- custom headers like X-Api-Key are
                # NOT stripped, so a compromised/misconfigured endpoint
                # that answers with a 301/302/303 to a different host
                # would otherwise leak the live API key to that host in
                # plaintext (the JSON body is dropped on that same
                # redirect class, but the header is not). This POST is
                # internal machine-to-machine trace ingestion; there's no
                # product reason it should ever need to follow a redirect.
                allow_redirects=False,
            )
            response.raise_for_status()
            # raise_for_status() only raises on 4xx/5xx -- a 3xx would
            # otherwise fall through as "success" below even though
            # allow_redirects=False means it was never followed, so the
            # spans were never actually delivered anywhere.
            if 300 <= response.status_code < 400:
                logger.error(
                    "Trace POST received an unexpected redirect (%s) and "
                    "was not followed (allow_redirects=False) — spans not "
                    "delivered",
                    response.status_code,
                    extra={
                        "endpoint": self.endpoint,
                        "span_count": len(spans),
                        "status_code": response.status_code,
                    },
                )
                return _RETRY
            logger.info(
                "Successfully sent spans to backend",
                extra={
                    "endpoint": self.endpoint,
                    "span_count": len(spans),
                    "trace_count": len(traces),
                    "policy_id": policy_id or None,
                },
            )
            _reset_auth_failure_latch()
            return _OK
        except requests.exceptions.RequestException as e:
            # Auth failures (401/403) are operator-actionable — a bad
            # API key or missing X-Application-Id will drop every span
            # forever until it's fixed. Surface them at CRITICAL with a
            # deploy-visible message, distinct from a transient 5xx or
            # network blip.
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code in (401, 403):
                logger.critical(
                    "Ingest auth rejected (%s) — spans will keep failing "
                    "until credentials are fixed. Check DISSEQT_API_KEY / "
                    "X-Application-Id and the traces-auth policy binding.",
                    status_code,
                    extra={
                        "endpoint": self.endpoint,
                        "span_count": len(spans),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "status_code": status_code,
                    },
                )
                # disseqt_logging is silent-by-default until the app
                # calls configure() — most fresh deployments never do,
                # so the CRITICAL line above would emit nowhere. Auth
                # failures are exactly the kind of thing operators
                # need to see with default settings, so write a copy
                # straight to stderr as well. TP-2128 round-2 P1 #1.3.
                _write_auth_failure_to_stderr(status_code, self.endpoint)
            elif _is_permanent_status(status_code):
                # Logged once per dropped group; the buffer never sees these.
                logger.error(
                    "Server permanently rejected spans (%s) — dropping them, not retrying",
                    status_code,
                    extra={
                        "endpoint": self.endpoint,
                        "span_count": len(spans),
                        "error": str(e),
                        "status_code": status_code,
                    },
                )
            else:
                logger.error(
                    "Failed to send spans to backend",
                    extra={
                        "endpoint": self.endpoint,
                        "span_count": len(spans),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "status_code": status_code,
                    },
                    exc_info=True,
                )
            return _PERMANENT if _is_permanent_status(status_code) else _RETRY

    def send_trace(self, trace_spans: list[EnrichedSpan]) -> bool:
        """
        Send trace spans (alias for send_spans for compatibility).

        Args:
            trace_spans: List of EnrichedSpan objects from a trace

        Returns:
            bool: True if successful, False otherwise
        """
        return self.send_spans(trace_spans)
