"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol, cast, runtime_checkable

import requests

from disseqt_logging import digest, get_logger

from .models.composite_score import CompositeScoreRequest
from .models.themes_classifier import ThemesClassifierRequest
from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


@runtime_checkable
class SupportsInputData(Protocol):
    """Anything that can serialize itself to the wire-shape ``input_data``
    dict — every ``disseqt_sdk.models`` request object implements this, so
    a bare model (e.g. ``InputValidationRequest``) can be passed straight
    to :meth:`Client.validate` together with ``policies=[...]``."""

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

    3. **Run one or more published realtime policies** — pass
       ``policies=[...]`` to :meth:`validate`, with or without a
       validator::

           from disseqt_sdk import any_blocking

           result = client.validate(
               InputValidationRequest(prompt="user prompt here"),
               policies=["b1f8…"],
           )
           if any_blocking(result):
               ...  # at least one policy said BLOCK

       For each policy id, the server fetches the policy from
       disseqt-realtime-policies-service, runs every validator the policy
       specifies (with the policy's thresholds and decision strategy),
       aggregates a BLOCK/PASS verdict, and publishes the result to
       ``policy.validation.result.v1`` so it shows up on the Decisions
       dashboard. The policy endpoints live on their own base URL
       (``realtime_policy_base_url``) so they can be mocked or pointed at
       a local server during tests without disturbing the validator base
       URL. Requires ``application_name`` on the client.

    """

    def __init__(
        self,
        project_id: str,
        api_key: str,
        base_url: str = "https://api.disseqt.ai/realtime-validations",
        timeout: int = 30,
        application_name: str | None = None,
        realtime_policy_base_url: str = "https://api.disseqt.ai/realtime-validations",
    ) -> None:
        """Initialize the Disseqt SDK client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Base URL for the individual-validator API (the
                ``/sdk/validators/...`` endpoints).
            timeout: Request timeout in seconds
            application_name: Logical name of the calling application
                (e.g. ``"checkout-bot"``). REQUIRED to evaluate policies
                (``validate(..., policies=[...])``) — the
                ``policy.validation.result.v1`` ledger uses this to
                show which application produced each decision. Mirrors
                ``service_name`` on :class:`DisseqtAgenticClient`.
            realtime_policy_base_url: Base URL of the realtime-policy
                evaluate endpoint. Defaults to the
                ``/realtime-validations`` gateway — the evaluate
                endpoint is served by production-monitoring, the same
                service that hosts the validators (the
                ``/realtime-policies`` gateway is the policy CRUD
                dashboard and has no SDK routes). Kept separate from
                ``base_url`` so the two endpoints can be mocked /
                routed independently — override for local testing
                (e.g. ``http://localhost:9010``) without disturbing
                ``base_url`` callers.

        """
        self.project_id = project_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.application_name = application_name
        self.realtime_policy_base_url = realtime_policy_base_url

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for API requests.

        Returns:
            Dictionary of HTTP headers
        """
        return {
            "X-API-Key": self.api_key,
            "X-Project-Id": self.project_id,
            "Content-Type": "application/json",
        }

    def validate(
        self,
        request: (
            BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator | SupportsInputData
        ),
        policies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a validator, one or more realtime policies, or both.

        Three call shapes, chosen by what you pass:

        1. **Validator only** (unchanged classic behavior)::

               client.validate(ToxicityValidator(data=..., config=...))

           Runs that one validator; returns its validation response.

        2. **Validator + policies** — the validator runs as usual AND the
           same input is evaluated against each policy id, server-side,
           with each policy's own rulesets, thresholds, and decision
           strategy::

               result = client.validate(
                   ToxicityValidator(data=InputValidationRequest(prompt=p)),
                   policies=["994ad00e-…", "1268faa4-…"],
               )

        3. **Policies only** — pass a bare request object (any
           ``disseqt_sdk.models`` request, no validator, no config); the
           policies decide everything::

               result = client.validate(
                   InputValidationRequest(prompt=p, response=r),
                   policies=["994ad00e-…"],
               )

        When ``policies`` is passed the return value is a stable envelope::

            {
              "validation": {...} | None,   # per-validator result, None in shape 3
              "policies":  [{...}, ...],    # one policy envelope per id, in order
            }

        Use :func:`disseqt_sdk.any_blocking` to gate on it. Each policy is
        one server-side evaluation (billed per executed validator, one
        Decisions-ledger entry each); policies are evaluated sequentially
        in the order given. Inputs a policy's validator doesn't receive
        skip neutrally with ``missing_input:<fields>`` — supply the union
        of fields the policies need (see the policy detail endpoint's
        ``required_input_fields``).

        Without ``policies``, behavior is exactly as before.

        Args:
            request: Validator instance, or a bare request object when
                ``policies`` is given.
            policies: Optional list of published policy ids to evaluate
                the input against. Composite-score and themes-classifier
                requests cannot be combined with ``policies``.

        Returns:
            The validation response — or the ``{"validation", "policies"}``
            envelope when ``policies`` is passed.

        Raises:
            HTTPError: If any API request fails (unknown/unpublished
                policy answers 404 DSQ-4040).
            ValueError: On invalid combinations (bare request without
                ``policies``, empty ``policies`` list, missing
                ``application_name``, composite/themes with ``policies``)
                or an undecodable response body.
        """
        if policies is not None:
            return self._validate_with_policies(request, policies)
        if not isinstance(
            request, (BaseValidator, ThemesClassifierValidator, CompositeScoreEvaluator)
        ):
            raise ValueError(
                "A bare request object needs policies=[...] — pass a validator "
                "instance to run a single validator, or add policies=[...] to "
                "evaluate this input against realtime policies"
            )
        return self._run_validator(request)

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
        # input_data; bare models serialize themselves.
        if isinstance(request, BaseValidator):
            input_data = dict(request.to_payload().get("input_data") or {})
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
            self._post_policy_evaluate(policy_id, input_data, application_name)
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
        if not http_resp.ok:
            logger.error(
                "policy.http_error",
                policy_id=policy_id,
                status=http_resp.status_code,
                latency_ms=latency_ms,
            )
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
