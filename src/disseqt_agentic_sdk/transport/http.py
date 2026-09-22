"""
HTTP transport for sending traces/spans to the backend API.
"""

import json
import os
import sys
from typing import Any

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


def _write_auth_failure_to_stderr(status_code: int, endpoint: str) -> None:
    """
    Direct stderr write for 401/403 responses. Bypasses the logging
    stack entirely so the message survives a fresh unconfigured
    process (``disseqt_logging`` silences every level until
    ``configure()`` / ``DISSEQT_LOG_LEVEL`` is set). The
    logger.critical call still runs alongside this for anyone with
    structured logging configured.
    """
    if _SILENCE_AUTH_STDERR:
        return
    try:
        sys.stderr.write(
            f"[disseqt-agentic-sdk] CRITICAL: ingest auth rejected "
            f"({status_code}) at {endpoint} — spans will keep failing "
            f"until credentials are fixed. Check DISSEQT_API_KEY / "
            f"X-Application-Id and the traces-auth policy binding. "
            f"Silence this message with DISSEQT_SDK_SILENCE_AUTH_STDERR=1.\n"
        )
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 — never let a stderr write crash the caller
        pass


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
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.realtime_policy_id = realtime_policy_id
        self.application_id = application_id
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
        return not self.send_spans_with_failures(spans)

    def send_spans_with_failures(self, spans: list[EnrichedSpan]) -> list[EnrichedSpan]:
        """
        Send spans and return the ones that failed to deliver.

        Spans are grouped by their per-trace ``realtime_policy_id`` (with
        the client default as the fallback) and each distinct group is
        sent as its own HTTP POST. This is what makes per-trace policy
        overrides work: traces using different policies in the same
        batch land in separate payloads, each with the right
        ``resource.attributes["policy.id"]``. Single-policy batches still
        produce a single POST so there's no extra HTTP cost in the common
        case.

        Returns:
            list[EnrichedSpan]: exactly the spans that failed to send.
            Empty list on full success. The retry buffer uses this so
            successfully-delivered groups aren't re-POSTed on the next
            flush after a partial-failure batch (TP-2128 round-2 P1
            #1.2).
        """
        if not spans:
            return []

        # Bucket spans by the policy that should be stamped on their
        # outgoing resource block. Empty string means "no per-trace
        # override; use the client default" — which may itself be empty.
        groups: dict[str, list[EnrichedSpan]] = {}
        for span in spans:
            pid = getattr(span, "realtime_policy_id", "") or self.realtime_policy_id or ""
            groups.setdefault(pid, []).append(span)

        failed: list[EnrichedSpan] = []
        for pid, group in groups.items():
            if not self._send_group(pid, group):
                failed.extend(group)
        return failed

    def _send_group(self, policy_id: str, spans: list[EnrichedSpan]) -> bool:
        """Send one resource-block's worth of spans (single policy_id)."""
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
                    "api.key": self.api_key,
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
                return False
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
                # plaintext (the JSON body, and its
                # resource.attributes["api.key"] copy, IS dropped on that
                # same redirect class -- headers were the one channel the
                # body-only design never exposed here). This POST is
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
                return False
            logger.info(
                "Successfully sent spans to backend",
                extra={
                    "endpoint": self.endpoint,
                    "span_count": len(spans),
                    "trace_count": len(traces),
                    "policy_id": policy_id or None,
                },
            )
            return True
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
