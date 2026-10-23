"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

import requests

from disseqt_logging import digest, get_logger

from ._version import check_version_notice, sdk_identity_headers
from .auth import AuthMissingError, resolve
from .factories import (
    _AgenticFactory,
    _CompositeFactory,
    _InputValidatorFactory,
    _McpFactory,
    _OutputValidatorFactory,
    _RagFactory,
    _ThemesFactory,
)
from .models.composite_score import CompositeScoreRequest
from .models.themes_classifier import ThemesClassifierRequest
from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


_POLICY_DEPRECATION = (
    "policies=/realtime_policy_base_url= are deprecated and will be removed; the "
    "policy-evaluate endpoint is not served by any in-scope backend"
)


@runtime_checkable
class SupportsInputData(Protocol):
    """A bare ``disseqt_sdk.models`` request; deprecated ``policies=`` input."""

    def to_input_data(self) -> dict[str, Any]: ...


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

    ``policies=[...]`` (server-side realtime-policy evaluation) is a deprecated
    shim that emits :class:`DeprecationWarning`.
    """

    def __init__(
        self,
        project_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 30,
        application_name: str | None = None,
        realtime_policy_base_url: str | None = None,
        policies: list[str] | None = None,
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
            AuthMissingError: No API key from kwargs, env or
                ``~/.disseqt/config.json``. Fails at construction
                instead of the first API call.
        """
        # Precedence: kwargs, then env, then ~/.disseqt/config.json (env beats config).
        api_key, project_id, _ = resolve(api_key, project_id)
        if not api_key:
            raise AuthMissingError(
                "Missing api_key. Provide it via:\n"
                "  1. Client(api_key=...)\n"
                "  2. env var DISSEQT_API_KEY (DISSEQT_PROJECT_ID optional)\n"
                "  3. `disseqt login` (writes ~/.disseqt/config.json)"
            )
        if realtime_policy_base_url is not None or policies:
            warnings.warn(_POLICY_DEPRECATION, DeprecationWarning, stacklevel=2)
        default_policies = list(policies) if policies else None
        if default_policies is not None:
            if not all(isinstance(p, str) and p.strip() for p in default_policies):
                raise ValueError(
                    f"Client(policies=...) must be a list of policy-id strings (got {default_policies!r})"
                )
            if not (application_name and application_name.strip()):
                raise ValueError("application_name is required when Client(policies=...) is set")
        self.project_id: str | None = project_id
        self.api_key: str = api_key
        self.realtime_policy_base_url = (
            realtime_policy_base_url or "https://api.disseqt.ai/realtime-validations"
        )
        self.policies = default_policies
        self.base_url = base_url or "https://api.disseqt.ai/realtime-validations"
        self.timeout = timeout
        self.application_name = application_name

        # Node-style factories: ``client.input.toxicity(...)`` returns a validator for validate().
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
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            **sdk_identity_headers(),
        }
        if self.project_id:
            headers["X-Project-Id"] = self.project_id
        return headers

    def validate(
        self,
        request: (
            BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator | SupportsInputData
        ),
        policies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a single validator (or composite/themes) and return its response.

        ``policies=[...]`` (and a ``Client(policies=...)`` default) is a deprecated
        shim kept for 0.11 callers; it emits :class:`DeprecationWarning`.

        Raises:
            HTTPError: If the API request fails.
            ValueError: If ``request`` is not a validator instance.
        """
        if policies is not None:
            warnings.warn(_POLICY_DEPRECATION, DeprecationWarning, stacklevel=2)
            return self._validate_with_policies(request, policies)
        if self.policies is not None and not isinstance(
            request,
            (
                ThemesClassifierValidator,
                CompositeScoreEvaluator,
                ThemesClassifierRequest,
                CompositeScoreRequest,
            ),
        ):
            warnings.warn(_POLICY_DEPRECATION, DeprecationWarning, stacklevel=2)
            return self._validate_with_policies(request, self.policies)
        if not isinstance(
            request, (BaseValidator, ThemesClassifierValidator, CompositeScoreEvaluator)
        ):
            raise ValueError(
                "A bare request object needs policies=[...] — pass a validator "
                "instance to run a single validator "
                f"(got {type(request).__name__})"
            )
        return self._run_validator(request)

    def validate_sync(
        self,
        request: (
            BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator | SupportsInputData
        ),
        policies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Alias for :meth:`validate` (Node ``validateSync`` parity)."""
        return self.validate(request, policies=policies)

    def _validate_with_policies(
        self,
        request: (
            BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator | SupportsInputData
        ),
        policies: list[str],
    ) -> dict[str, Any]:
        """Orchestrate shape 2/3 of :meth:`validate` (``policies=[...]``).

        Every client-side rule is checked — and raises ``ValueError`` —
        BEFORE any network call is made.
        """
        # Normalize first: a one-shot iterable (generator) would otherwise
        # be exhausted by validation and silently evaluate zero policies.
        try:
            policy_ids = list(policies)
        except TypeError:
            raise ValueError(
                f"policies must be a list of policy-id strings (got {policies!r})"
            ) from None
        if not policy_ids or not all(isinstance(p, str) and p.strip() for p in policy_ids):
            raise ValueError(
                "policies must be a non-empty list of policy-id strings " f"(got {policy_ids!r})"
            )
        if isinstance(
            request,
            (
                ThemesClassifierValidator,
                CompositeScoreEvaluator,
                ThemesClassifierRequest,
                CompositeScoreRequest,
            ),
        ):
            raise ValueError(
                "policies=[...] is not supported with composite-score or "
                "themes-classifier requests — those endpoints have their own "
                "aggregation and are never policy-evaluated"
            )
        application_name = self.application_name
        if not (application_name and application_name.strip()):
            raise ValueError(
                "application_name is required to evaluate policies — set "
                "Client(application_name=...) so the Decisions ledger can "
                "attribute each decision to your application"
            )

        # Both shapes carry the input on an object that knows its wire
        # form. A validator's payload already contains the renamed
        # input_data; bare models serialize themselves. A validator's
        # config_input (threshold, custom labels, llm_as_a_judge flag, …)
        # is forwarded to the policy evaluation too, so per-validator
        # config reaches the server-side policy engine; bare models carry
        # no config.
        config_input: dict[str, Any] | None = None
        if isinstance(request, BaseValidator):
            payload = request.to_payload()
            input_data = dict(payload.get("input_data") or {})
            config_input = dict(payload.get("config_input") or {}) or None
        elif isinstance(request, SupportsInputData):
            input_data = request.to_input_data()
        else:
            raise ValueError(
                "request must be a validator instance or a disseqt_sdk.models "
                f"request object, got {type(request).__name__}"
            )
        if not input_data:
            raise ValueError(
                "the request carries no input fields — set prompt/context/"
                "response (or agentic fields) so the policies have something "
                "to evaluate"
            )

        # All guards passed — now (and only now) touch the network.
        validation: dict[str, Any] | None = (
            self._run_validator(request) if isinstance(request, BaseValidator) else None
        )
        envelopes = [
            self._post_policy_evaluate(
                policy_id, input_data, application_name, config_input=config_input
            )
            for policy_id in policy_ids
        ]
        logger.info(
            "validation.policies",
            policy_count=len(envelopes),
            with_validator=validation is not None,
        )
        return {"validation": validation, "policies": envelopes}

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

    def _post_policy_evaluate(
        self,
        policy_id: str,
        input_data: dict[str, Any],
        application_name: str,
        config_input: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """POST one policy-evaluation request and decode the envelope.

        Transport for :meth:`validate` (``policies=[...]``).
        Raises :class:`HTTPError` on any non-2xx
        (unknown/unpublished policies answer 404 DSQ-4040) and
        ``ValueError`` on an undecodable body.
        """
        url = (
            f"{self.realtime_policy_base_url.rstrip('/')}"
            f"/api/v1/sdk/policies/{policy_id}/evaluate"
        )
        payload: dict[str, Any] = {
            "input_data": input_data,
            "application_name": application_name,
        }
        if config_input is not None:
            payload["config_input"] = config_input

        headers = self._build_headers()
        if request_id is not None:
            headers["X-Request-Id"] = request_id

        started = time.monotonic()
        try:
            http_resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            logger.error(
                "policy.network_error",
                policy_id=policy_id,
                latency_ms=round((time.monotonic() - started) * 1000, 1),
                exc_info=True,
            )
            raise HTTPError(
                status_code=0,
                message=f"Network error: {e}",
                response_body="",
            ) from e

        latency_ms = round((time.monotonic() - started) * 1000, 1)
        check_version_notice(http_resp.headers)
        if not http_resp.ok:
            logger.error(
                "policy.http_error",
                policy_id=policy_id,
                status=http_resp.status_code,
                latency_ms=latency_ms,
            )
            blocked = _version_blocked_error(
                http_resp.status_code, http_resp.headers, http_resp.text
            )
            if blocked is not None:
                raise blocked
            raise HTTPError(
                status_code=http_resp.status_code,
                message="Policy evaluation failed",
                response_body=http_resp.text[:512] if http_resp.text else "",
            )
        try:
            data = http_resp.json()
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to decode policy response: {e}. Body: {http_resp.text[:200]}"
            ) from e
        logger.info(
            "policy.response",
            policy_id=policy_id,
            status=http_resp.status_code,
            latency_ms=latency_ms,
        )
        return cast(dict[str, Any], data)
