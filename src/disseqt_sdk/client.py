"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import requests

from disseqt_logging import digest, get_logger

from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


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

    3. **Run a published realtime policy** — use :meth:`evaluate_policy`::

           result = client.evaluate_policy(
               realtime_policy_id="b1f8…",
               prompt="user prompt here",
               application_name="checkout-bot",
           )
           if result["decision"] == "BLOCK":
               ...  # don't pass the LLM output downstream

       The server fetches the policy from
       disseqt-realtime-policies-service, runs every validator the policy
       specifies (with the policy's thresholds), aggregates a BLOCK/PASS
       verdict, and publishes the result to
       ``policy.validation.result.v1`` so it shows up on the policies
       dashboard. The endpoint lives on its own base URL
       (``realtime_policy_base_url``) so it can be mocked or pointed at a
       local server during tests without disturbing the validator base URL.

    If you call :meth:`evaluate_policy` without a ``realtime_policy_id``
    (and the Client has no default), it will raise — with an error that
    points you back at options 1 and 2.
    """

    def __init__(
        self,
        project_id: str,
        api_key: str,
        base_url: str = "https://api.disseqt.ai/realtime-validations",
        timeout: int = 30,
        realtime_policy_id: str | None = None,
        application_name: str | None = None,
        realtime_policy_base_url: str = "https://api.disseqt.ai/realtime-policies",
    ) -> None:
        """Initialize the Disseqt SDK client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Base URL for the individual-validator API (the
                ``/sdk/validators/...`` endpoints).
            timeout: Request timeout in seconds
            realtime_policy_id: Optional default policy used by
                :meth:`evaluate_policy`. When set, callers can omit the
                ``realtime_policy_id`` argument on each call and this
                default is used. Per-call value always wins.
            application_name: Logical name of the calling application
                (e.g. ``"checkout-bot"``). REQUIRED when
                ``realtime_policy_id`` is set — the
                ``policy.validation.result.v1`` ledger uses this to
                show which application produced each decision. Mirrors
                ``service_name`` on :class:`DisseqtAgenticClient`.
            realtime_policy_base_url: Base URL of the realtime-policy
                evaluate endpoint. Defaults to the managed
                ``/realtime-policies`` gateway. Separate from
                ``base_url`` (which is the validators endpoint on
                production-monitoring) so the two endpoints can be
                mocked / routed independently — override for local
                testing (e.g. ``http://localhost:9010``) without
                disturbing ``base_url`` callers.

        Raises:
            ValueError: When ``realtime_policy_id`` is set without an
                accompanying ``application_name``.
        """
        if realtime_policy_id and not (application_name and application_name.strip()):
            raise ValueError(
                "application_name is required when realtime_policy_id is set "
                "(it identifies the calling application on the policies "
                "dashboard, same as service_name on DisseqtAgenticClient)"
            )
        self.project_id = project_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.realtime_policy_id = realtime_policy_id
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
        self, request: BaseValidator | ThemesClassifierValidator | CompositeScoreEvaluator
    ) -> dict[str, Any]:
        """Validate using the specified validator.

        Args:
            request: Validator request instance

        Returns:
            Normalized validation response

        Raises:
            HTTPError: If the API request fails
            ValueError: If JSON decoding fails
        """
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

    def evaluate_policy(
        self,
        realtime_policy_id: str | None = None,
        *,
        # LLM text fields — match what the typed validators expose.
        # The SDK renames these to the wire shape the validators expect:
        # prompt → llm_input_query, context → llm_input_context,
        # response → llm_output. Same convention as
        # InputValidationRequest / OutputValidationRequest.
        prompt: str | None = None,
        context: str | None = None,
        response: str | None = None,
        # Agentic-behavior fields — for policies that include agentic
        # validators (tool_call_accuracy, topic_adherence, …). Sent 1:1
        # in the wire payload, no rename.
        conversation_history: list[str] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        agent_responses: list[str] | None = None,
        reference_data: dict[str, Any] | None = None,
        # Escape hatch for shapes the typed args don't cover (themes
        # classifier, custom validators). Merged on top of whatever the
        # typed args produced — raw keys win on conflict.
        input_data: dict[str, Any] | None = None,
        config_input: dict[str, Any] | None = None,
        application_name: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Run an input through a server-side realtime policy.

        Hits ``POST /api/v1/sdk/policies/{realtime_policy_id}/evaluate`` on
        the URL configured via ``realtime_policy_base_url``. The server
        fetches the policy from disseqt-realtime-policies-service, runs
        every validator the policy specifies (with the policy's thresholds
        and any custom labels), aggregates a BLOCK/PASS verdict, and
        publishes the result to ``policy.validation.result.v1``.

        A single policy can span multiple validator domains. The same
        ``input_data`` dict is sent to every validator the policy lists,
        and each validator picks the fields it cares about — so a policy
        with both ``factual_consistency`` (needs query + context +
        response) and ``tool_call_accuracy`` (needs tool_calls) requires
        callers to supply the **union** of fields::

            client.evaluate_policy(
                # LLM validators read these
                prompt="What is the capital of France?",
                context="France is a country in Europe.",
                response="The capital of France is Paris.",
                # Agentic validators read these
                tool_calls=[{"name": "lookup_capital", "args": {...}}],
                conversation_history=["..."],
            )

        Use the raw ``input_data`` dict only for shapes the typed args
        don't cover (e.g. ``themes_classifier``).

        Args:
            realtime_policy_id: UUID of the published policy to run.
                Falls back to ``Client.realtime_policy_id`` (set at
                construction). Raises if neither is set.
            prompt: User query. Sent as ``llm_input_query`` on the wire.
            context: Additional context. Sent as ``llm_input_context``.
            response: LLM output to evaluate. Sent as ``llm_output``.
            conversation_history: Prior turns. Sent as
                ``conversation_history`` (no rename).
            tool_calls: Tool invocations the agent made.
            agent_responses: Agent's textual replies.
            reference_data: Ground-truth / reference docs for the
                agentic validators that need them.
            input_data: Escape hatch — a raw dict merged on top of
                whatever the typed args produced. Use only when the
                typed args don't cover what you need.
            config_input: Optional extra config merged into every
                validator's config. The policy's threshold always wins
                on conflict; this only fills in fields the policy
                didn't set.
            application_name: Optional override of the Client's
                ``application_name``. Required (here or on the Client).
            request_id: Optional override; server generates one if absent.

        Returns:
            Decoded JSON response with::

                {
                  "success": true,
                  "policy_id": "...",
                  "policy_name": "...",
                  "policy_version": 3,
                  "decision": "BLOCK" | "PASS",
                  "enforcement": "blocking" | "advisory",
                  "rulesets": [
                    {"ruleset_id": "...", "ruleset_name": "...",
                     "required": true,
                     "rules": [{"validator": "toxicity", "status": "fail",
                                "score": 0.91, "threshold": 0.8,
                                "polarity": "risk", "is_decider": true,
                                ...}]}
                  ],
                  "duration": "...",
                  "request_id": "...",
                  "credit_details": {...}
                }

            ``rulesets[].rules[].validator`` lets you see which validators
            the policy actually ran — useful for confirming the policy is
            configured the way you expect.

            Use :func:`disseqt_sdk.is_blocking` on this dict to short-
            circuit on BLOCK without poking at the shape directly.

        Raises:
            HTTPError: If the server returns a non-2xx, including 404
                when the policy is unknown or unpublished.
            ValueError: If no input fields were supplied, no policy_id
                is set anywhere, or no application_name is set anywhere.
        """
        effective_policy_id = realtime_policy_id or self.realtime_policy_id
        if not effective_policy_id:
            raise ValueError(
                "evaluate_policy() needs a realtime_policy_id. Options:\n"
                "  • Pass realtime_policy_id=... on the call\n"
                "  • Set Client(realtime_policy_id=...) at construction\n"
                "If you don't have a published policy and just want to run\n"
                "validators directly, use client.validate(...) instead:\n"
                "  • One validator → client.validate(ToxicityValidator(...))\n"
                "  • Composite bundle → client.validate(CompositeScoreEvaluator(...))"
            )
        effective_application_name = application_name or self.application_name
        if not effective_application_name:
            raise ValueError(
                "application_name is required when calling evaluate_policy() "
                "— either pass it per call or set Client(application_name=...) "
                "at construction"
            )

        # Build the wire-shape input_data dict from the typed args.
        # Rename llm fields to match what ML services expect; agentic
        # fields go through 1:1.
        built: dict[str, Any] = {}
        if prompt is not None:
            built["llm_input_query"] = prompt
        if context is not None:
            built["llm_input_context"] = context
        if response is not None:
            built["llm_output"] = response
        if conversation_history is not None:
            built["conversation_history"] = conversation_history
        if tool_calls is not None:
            built["tool_calls"] = tool_calls
        if agent_responses is not None:
            built["agent_responses"] = agent_responses
        if reference_data is not None:
            built["reference_data"] = reference_data
        # Raw escape-hatch dict wins on key conflict so callers can
        # override the typed-arg rename when they need to.
        if input_data is not None:
            built.update(input_data)
        if not built:
            raise ValueError(
                "evaluate_policy() needs at least one input field — pass "
                "prompt/context/response for LLM validators, "
                "conversation_history/tool_calls/agent_responses/reference_data "
                "for agentic validators, or input_data=... as a raw dict"
            )

        url = (
            f"{self.realtime_policy_base_url.rstrip('/')}"
            f"/api/v1/sdk/policies/{effective_policy_id}/evaluate"
        )
        payload: dict[str, Any] = {
            "input_data": built,
            "application_name": effective_application_name,
        }
        if config_input is not None:
            payload["config_input"] = config_input

        headers = self._build_headers()
        if request_id is not None:
            headers["X-Request-Id"] = request_id

        try:
            # NB: local is `http_resp`, not `response`, to avoid shadowing
            # the function's `response: str | None` parameter (the LLM
            # output the caller wants validated) — mypy strict mode
            # refuses to narrow a parameter type to requests.Response.
            http_resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if not http_resp.ok:
                body = http_resp.text[:512] if http_resp.text else ""
                raise HTTPError(
                    status_code=http_resp.status_code,
                    message="Policy evaluation failed",
                    response_body=body,
                )
            try:
                data = http_resp.json()
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to decode policy response: {e}. Body: {http_resp.text[:200]}"
                ) from e
            return cast(dict[str, Any], data)
        except requests.RequestException as e:
            raise HTTPError(
                status_code=0,
                message=f"Network error: {e}",
                response_body="",
            ) from e
