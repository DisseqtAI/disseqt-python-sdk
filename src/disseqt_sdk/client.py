"""Disseqt SDK client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Protocol, cast, runtime_checkable

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import MaxRetryError, ResponseError
from urllib3.util.retry import Retry

from disseqt_logging import digest, get_logger
from disseqt_logging.header_validation import validate_header_value

from ._version import check_version_notice, sdk_identity_headers
from .models.composite_score import CompositeScoreRequest
from .models.themes_classifier import ThemesClassifierRequest
from .registry import get_validator_metadata
from .routes import build_validator_url
from .validators.base import BaseValidator, ThemesClassifierValidator
from .validators.composite.evaluate import CompositeScoreEvaluator

logger = get_logger(__name__)


# Longest server-requested ``Retry-After`` (seconds) we will sit out. validate()
# runs on the caller's inference path, so a long wait is worse than surfacing
# the 429 immediately and letting the caller decide.
_MAX_RETRY_AFTER_S = 5.0


class _RateLimitRetry(Retry):
    """Retry policy for the validator client: HTTP 429 only, bounded wait.

    Connection errors and read timeouts are never retried (a request that may
    have reached the server could be billed twice), and a ``Retry-After``
    longer than ``_MAX_RETRY_AFTER_S`` is not waited out -- the 429 is
    returned to the caller instead.
    """

    def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):  # type: ignore[no-untyped-def]
        if response is not None and response.status == 429:
            header = response.headers.get("Retry-After")
            wait = self.parse_retry_after(header) if header else None
            if wait is not None and wait > _MAX_RETRY_AFTER_S:
                raise MaxRetryError(_pool, url, ResponseError("Retry-After exceeds cap"))
        return super().increment(method, url, response, error, _pool, _stacktrace)


def _build_session(max_retries: int) -> requests.Session:
    """One pooled session per Client; retries only HTTP 429 (see _RateLimitRetry)."""
    session = requests.Session()
    retry = _RateLimitRetry(
        total=max_retries,
        connect=0,
        read=0,
        status=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429],
        allowed_methods=frozenset({"POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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


class ResponseDecodeError(HTTPError, ValueError):
    """The server answered 2xx but the body was not a usable JSON object.

    Subclasses both :class:`HTTPError` (so ``except HTTPError`` catches every
    failed call) and ``ValueError`` (what this path raised before, so existing
    handlers keep working). The message names the problem and, where useful,
    the JSON *type* received -- never the body, which may echo user content.
    """


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


def _policy_error_entry(policy_id: str, exc: Exception) -> dict[str, Any]:
    """Per-policy error entry placed in ``policies`` when one evaluation fails."""
    error: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        error["status_code"] = status_code
    return {"policy_id": policy_id, "status": "error", "sdk_error": True, "error": error}


def _is_error_entry(envelope: dict[str, Any]) -> bool:
    return envelope.get("sdk_error") is True


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
        policies: list[str] | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize the Disseqt SDK client.

        Args:
            project_id: Project ID for the Disseqt API
            api_key: API key for authentication
            base_url: Base URL for the individual-validator API (the
                ``/sdk/validators/...`` endpoints).
            timeout: Request timeout in seconds
            max_retries: How many times an HTTP 429 (rate limited) response is
                retried, honouring ``Retry-After`` (waits longer than a few
                seconds are not retried). Connection errors, timeouts and 5xx
                are never retried, so a validator run is never billed twice.
                ``0`` disables retrying.
            application_name: Logical name of the calling application
                (e.g. ``"checkout-bot"``). REQUIRED to evaluate policies
                (a client-level ``policies`` default or per-call
                ``validate(..., policies=[...])``) — the
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
            policies: Optional default list of published policy ids.
                When set, EVERY ``validate()`` call evaluates these
                policies unless the call passes its own ``policies=``
                (per-call always wins; there is no per-call opt-out —
                use a second Client for ungoverned paths). Composite-
                score and themes-classifier requests are incompatible
                with policies and run classically, without the default.
                An empty list means "no default", so env-driven config
                degrades naturally::

                    ids = [p for p in os.environ.get("DISSEQT_POLICIES", "").split(",") if p]
                    client = Client(..., application_name="checkout-bot", policies=ids)

                The list is copied defensively; later mutation of the
                caller's list does not affect the client.

        Raises:
            ValueError: When ``project_id`` or ``api_key`` is empty or not
                safe to send as an HTTP header, when ``policies`` is set
                without an ``application_name``, or when it contains a blank /
                non-string entry.
        """
        default_policies: list[str] | None = None
        if policies:
            default_policies = list(policies)
            if not all(isinstance(p, str) and p.strip() for p in default_policies):
                raise ValueError(
                    "Client(policies=...) must be a list of policy-id strings "
                    f"(got {default_policies!r})"
                )
            if not (application_name and application_name.strip()):
                raise ValueError(
                    "application_name is required when Client(policies=...) is "
                    "set — the Decisions ledger attributes each decision to "
                    "the calling application"
                )
        for _name, _value in (("project_id", project_id), ("api_key", api_key)):
            if not isinstance(_value, str) or not _value.strip():
                raise ValueError(f"Client requires a non-empty {_name} (got {_value!r})")
            # Both travel as HTTP headers. Characters outside Latin-1 make
            # http.client raise UnicodeEncodeError on every call; newlines are
            # a header-injection risk. Fail at construction, not at first send.
            validate_header_value(_value, _name)
        self.project_id = project_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError(f"max_retries must be a non-negative int (got {max_retries!r})")
        self._session = _build_session(max_retries)
        self.application_name = application_name
        self.realtime_policy_base_url = realtime_policy_base_url
        self.policies = default_policies

    def close(self) -> None:
        """Release the pooled HTTP connections. Safe to call more than once."""
        self._session.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

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

        **Client-level default.** A client constructed with
        ``Client(policies=[...])`` applies that list to every ``validate()``
        call that doesn't pass its own ``policies=`` — the per-call value
        always overrides the client default. Composite-score and
        themes-classifier requests are incompatible with policies; they run
        classically and the client default steps aside (logged). Passing
        ``policies=[]`` explicitly is always an error — an accidentally
        empty list must fail loudly rather than silently ungate the call.

        Without ``policies`` anywhere, behavior is exactly as before.

        Args:
            request: Validator instance, or a bare request object when
                policies apply (per-call or client default).
            policies: Optional list of published policy ids to evaluate
                the input against; overrides the client-level default.
                Composite-score and themes-classifier requests cannot be
                combined with ``policies``.

        Returns:
            The validation response — or the ``{"validation", "policies"}``
            envelope when policies apply.

        Raises:
            HTTPError: If any API request fails (unknown/unpublished
                policy answers 404 DSQ-4040).
            ValueError: On invalid combinations (bare request without
                policies anywhere, empty ``policies`` list, missing
                ``application_name``, explicit ``policies`` with
                composite/themes) or an undecodable response body.
        """
        if policies is not None:
            return self._validate_with_policies(request, policies)
        if self.policies is not None:
            # Composite/themes can't be policy-evaluated. An explicit
            # per-call combination raises (caller error), but a client-wide
            # default must not make those endpoints unusable — it steps
            # aside for them, visibly in the logs.
            if isinstance(
                request,
                (
                    ThemesClassifierValidator,
                    CompositeScoreEvaluator,
                    ThemesClassifierRequest,
                    CompositeScoreRequest,
                ),
            ):
                logger.info(
                    "validation.policies.default_skipped",
                    reason="composite/themes requests are never policy-evaluated",
                    request_type=type(request).__name__,
                )
            else:
                return self._validate_with_policies(request, self.policies)
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

        Partial failure: if one policy's evaluation fails after others
        succeeded, the earlier envelopes are NOT discarded. The result keeps
        the stable ``{"validation", "policies"}`` shape and the failed policy
        appears in ``"policies"`` (in order) as an error entry::

            {"policy_id": "...", "status": "error", "sdk_error": True,
             "error": {"type": "HTTPError", "status_code": 500, "message": "..."}}

        :func:`disseqt_sdk.policy.is_blocking` / ``any_blocking`` treat an
        error entry as blocking (fail closed) and ``parse`` returns ``None``
        for it; use :func:`disseqt_sdk.policy.is_error` to tell it apart from a
        real BLOCK. If *every* policy fails the original exception is raised
        (with the ``{"validation", "policies"}`` dict on ``.partial_result``),
        and :class:`SDKVersionBlockedError` (HTTP 426) always raises.
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
        envelopes: list[dict[str, Any]] = []
        first_error: Exception | None = None
        for policy_id in policy_ids:
            try:
                envelopes.append(
                    self._post_policy_evaluate(
                        policy_id, input_data, application_name, config_input=config_input
                    )
                )
            except SDKVersionBlockedError:
                raise  # upgrade-required is never a per-policy condition
            except (HTTPError, ValueError) as exc:
                first_error = first_error or exc
                envelopes.append(_policy_error_entry(policy_id, exc))
        if first_error is not None and all(_is_error_entry(e) for e in envelopes):
            # Nothing succeeded, so there is no decision to preserve: keep the
            # historical behaviour of raising. The validator result (if any)
            # rides along so it is not lost either.
            first_error.partial_result = {  # type: ignore[attr-defined]
                "validation": validation,
                "policies": envelopes,
            }
            raise first_error
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
            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except (requests.RequestException, UnicodeEncodeError) as e:
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
        except json.JSONDecodeError as e:
            logger.error(
                "validation.decode_error",
                domain=domain,
                slug=slug,
                status=response.status_code,
                latency_ms=latency_ms,
                exc_info=True,
            )
            # The body may echo user prompts / PII: log and report only a digest.
            raise ResponseDecodeError(
                status_code=response.status_code,
                message=(
                    f"Failed to decode JSON response: {e}. "
                    f"Response body: {digest(response.text or '')}"
                ),
                response_body="",
            ) from e
        if not isinstance(server_response_raw, dict):
            raise ResponseDecodeError(
                status_code=response.status_code,
                message=(
                    "Server returned an unexpected JSON "
                    f"{type(server_response_raw).__name__} response; expected an object"
                ),
                response_body="",
            )
        server_response = cast(dict[str, Any], server_response_raw)

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
            http_resp = self._session.post(url, json=payload, headers=headers, timeout=self.timeout)
        except (requests.RequestException, UnicodeEncodeError) as e:
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
            raise ResponseDecodeError(
                status_code=http_resp.status_code,
                message=(
                    f"Failed to decode policy response: {e}. "
                    f"Response body: {digest(http_resp.text or '')}"
                ),
                response_body="",
            ) from e
        if not isinstance(data, dict):
            raise ResponseDecodeError(
                status_code=http_resp.status_code,
                message=(
                    "Server returned an unexpected JSON "
                    f"{type(data).__name__} policy response; expected an object"
                ),
                response_body="",
            )
        logger.info(
            "policy.response",
            policy_id=policy_id,
            status=http_resp.status_code,
            latency_ms=latency_ms,
        )
        return cast(dict[str, Any], data)
