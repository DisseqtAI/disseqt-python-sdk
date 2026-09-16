"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, cast

import requests

from disseqt_logging import digest, get_logger

from ._version import check_version_notice, sdk_identity_headers
from .auth import AuthMissingError
from .factories import (
    _AgenticFactory,
    _CompositeFactory,
    _InputValidatorFactory,
    _McpFactory,
    _OutputValidatorFactory,
    _RagFactory,
    _ThemesFactory,
)
from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


def _load_stored_auth() -> dict[str, Any] | None:
    """Try the local config; swallow any error so import-time / construction
    stays dependable. A wider-than-0600 config surfaces later via the CLI's
    ``disseqt login`` path where we can print an actionable message.
    """
    try:
        from .auth import load as _load

        return _load()
    except Exception:
        return None


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


class SDKVersionBlockedError(HTTPError):
    """The server refused this call with HTTP 426 (DSQ-4260): the installed
    disseqt-ai-sdk is below the enforced minimum version — permanently, or
    during a scheduled brownout rehearsal ahead of the announced cutoff
    (carried in :attr:`sunset`).

    Subclasses :class:`HTTPError`, so existing ``except HTTPError`` handlers
    keep working unchanged; catch this type to branch specifically on
    "upgrade required" — page the platform team, apply a deliberate
    fail-open/fail-closed policy, or trigger upgrade automation. The remedy
    is always: ``pip install -U disseqt-ai-sdk``.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        response_body: str,
        *,
        latest: str | None = None,
        notice: str | None = None,
        sunset: str | None = None,
    ) -> None:
        """Initialize with the server's advertised version context.

        Args:
            status_code: HTTP status (426 in practice).
            message: Self-explanatory refusal text (the server envelope's
                ``error.external`` when available).
            response_body: Truncated raw response body.
            latest: ``X-SDK-Latest-Version`` response header, if present.
            notice: ``X-SDK-Notice`` response header, if present.
            sunset: RFC 8594 ``Sunset`` response header (the cutoff date),
                if present.
        """
        super().__init__(status_code, message, response_body)
        self.latest = latest
        self.notice = notice
        self.sunset = sunset


def _version_blocked_error(
    status_code: int, headers: Mapping[str, str], body_text: str | None
) -> SDKVersionBlockedError | None:
    """Build the typed 426 upgrade-required error, or ``None`` for any other
    status.

    Never raises — it runs on an error path that must stay dependable, so an
    unreadable body or headers degrades to a generic self-explanatory
    message. Headers are re-wrapped case-insensitively because one caller
    (``DisseqtAPIClient._get_raw``) passes a plain dict and the gateway may
    canonicalize header casing.
    """
    if status_code != 426:
        return None
    message = ""
    try:
        envelope = json.loads(body_text or "")
        message = str((envelope.get("error") or {}).get("external") or "")
    except Exception:
        message = ""
    try:
        ci_headers = requests.structures.CaseInsensitiveDict(headers)
        latest = ci_headers.get("X-SDK-Latest-Version") or None
        notice = ci_headers.get("X-SDK-Notice") or None
        sunset = ci_headers.get("Sunset") or None
    except Exception:
        latest = notice = sunset = None
    if not message:
        message = notice or (
            "this disseqt-ai-sdk version is no longer supported; "
            "upgrade with: pip install -U disseqt-ai-sdk"
        )
    return SDKVersionBlockedError(
        status_code,
        message,
        (body_text or "")[:512],
        latest=latest,
        notice=notice,
        sunset=sunset,
    )


class Client:
    """Disseqt SDK client for validator API calls.

    There are three ways to evaluate something with this client. Pick by
    what you want the server to do:

    1. **Run one specific validator** — use :meth:`validate`::

           from disseqt_sdk.validators.input import ToxicityValidator
           from disseqt_sdk.models import InputValidationRequest, SDKConfigInput

           client.validate(
               ToxicityValidator(
                   data=InputValidationRequest(prompt="…"),
                   config=SDKConfigInput(threshold=0.5),
               )
           )

       Hits ``/api/v1/sdk/validators/{type}/{name}``. No policy involved.
       Choose this when you know the exact validator + threshold you want.

    2. **Run a fixed bundle of validators** — use
       :class:`disseqt_sdk.validators.composite.CompositeScoreEvaluator`
       passed to :meth:`validate`. Hits
       ``/api/v1/sdk/validators/composite-score``. No policy involved.

    The prior *policies=[...]* shape (server-side realtime-policy
    evaluation) is not exposed in this release — the runtime evaluate
    endpoint it targeted is not currently served by any in-scope backend.
    Class-based validators are unaffected.
    """

    def __init__(
        self,
        project_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        application_name: str | None = None,
    ) -> None:
        """Initialize the Disseqt SDK client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Base URL for the individual-validator API (the
                ``/sdk/validators/...`` endpoints).
            timeout: Request timeout in seconds
            application_name: Logical name of the calling application
                (e.g. ``"checkout-bot"``). Recorded on outbound requests
                for observability. Mirrors ``service_name`` on
                :class:`DisseqtAgenticClient`.

        Raises:
            AuthMissingError: No creds from kwargs or
                ``~/.disseqt/config.json``. Fails at construction
                instead of the first API call.
        """
        # Resolution order: explicit kwargs → ``~/.disseqt/config.json``
        # (written by ``disseqt login``). Env-var fallback stays in the
        # CLI; the SDK constructor only reads the local config so library
        # callers get a predictable single source of truth. Missing creds
        # still surface at first API call as today.
        if not (project_id and api_key):
            stored = _load_stored_auth()
            if stored is not None:
                project_id = project_id or stored.get("project_id")
                api_key = api_key or stored.get("api_key")
                if base_url is None:
                    base_url = stored.get("base_url")
        if base_url is None:
            base_url = "https://api.disseqt.ai/realtime-validations"

        self.project_id = project_id or ""
        self.api_key = api_key or ""
        if not self.project_id or not self.api_key:
            raise AuthMissingError(
                "Missing project_id and/or api_key. Provide them via:\n"
                "  1. Client(project_id=..., api_key=...)\n"
                "  2. `disseqt login` (writes ~/.disseqt/config.json)\n"
                "  3. env vars DISSEQT_PROJECT_ID and DISSEQT_API_KEY"
            )
        self.base_url = base_url
        self.timeout = timeout
        self.application_name = application_name

        # Factory namespaces mirror the Node SDK's ``client.input.*`` /
        # ``client.rag.*`` / etc. ergonomics. Each method builds and returns
        # the appropriate validator instance; the caller then passes it to
        # ``validate()`` — additive over the class-based API.
        self.input = _InputValidatorFactory()
        self.output = _OutputValidatorFactory()
        self.rag = _RagFactory()
        self.agentic = _AgenticFactory()
        self.mcp = _McpFactory()
        self.themes = _ThemesFactory()
        self.composite = _CompositeFactory()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "X-API-Key": self.api_key,
            "X-Project-Id": self.project_id,
            "Content-Type": "application/json",
            **sdk_identity_headers(),
        }

    def validate(
        self,
        request: BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator,
    ) -> dict[str, Any]:
        """Run a single validator (or composite/themes) and return its response.

        The historical ``policies=[...]`` shape targeted an aspirational
        server-side policy-evaluate endpoint that no in-scope backend
        registers, so it was removed. This method now only runs the
        validator classes exposed under :mod:`disseqt_sdk.validators`.

        Args:
            request: A validator instance from :mod:`disseqt_sdk.validators`
                (or a themes-classifier / composite-score evaluator).

        Returns:
            The validator's raw response envelope.

        Raises:
            HTTPError: If the API request fails.
            ValueError: If ``request`` is not a validator instance.
        """
        if not isinstance(
            request, (BaseValidator, ThemesClassifierValidator, CompositeScoreEvaluator)
        ):
            raise ValueError(
                "request must be a validator instance from disseqt_sdk.validators "
                f"(got {type(request).__name__})"
            )
        return self._run_validator(request)

    def validate_sync(
        self,
        request: BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator,
    ) -> dict[str, Any]:
        """Alias for :meth:`validate` retained for API stability.

        Historical block-on-verdict semantics were tied to the removed
        server-side policy-evaluate path and no longer apply. Now a
        one-line delegator so existing callers keep working.
        """
        return self.validate(request)

    def _run_validator(
        self, request: BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator
    ) -> dict[str, Any]:
        """Run a single validator request (the classic validate() body)."""
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
        # Before the ok-check so error responses (e.g. a future 426
        # enforcement tier) still surface the upgrade notice.
        check_version_notice(response.headers)

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
            blocked = _version_blocked_error(response.status_code, response.headers, response.text)
            if blocked is not None:
                raise blocked
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
