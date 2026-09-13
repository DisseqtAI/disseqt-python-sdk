"""Production-runtime guardrails orchestrator (Phase 2a).

Fans out per-request policy evaluations across a list of "input guards" and
"output guards" — where each guard is a published realtime policy id (string)
or an object exposing a ``policy_id`` attribute (:class:`BaseGuard`).

Reuses :meth:`Client.validate` for the actual HTTP work — this class is a
composition layer, not a re-implementation. Fan-out is parallel via
``asyncio.gather`` in the async variants; the sync variants are a simple
sequential loop (the network is the tall pole, and the sync path is meant
for straight-line CI/gates where thread-safety wins over microseconds).

Example::

    from disseqt_sdk import Client, Guardrails
    from disseqt_sdk.models import InputValidationRequest

    client = Client(project_id="p", api_key="k", application_name="app")
    guard = Guardrails(client, input_guards=["policy-id-1"])
    result = guard.guard_input(InputValidationRequest(prompt="hi"))
    if result.blocked:
        return "refusal"
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .policy import any_blocking, is_blocking

if TYPE_CHECKING:
    from .client import Client, SupportsInputData
    from .extensions import BaseGuard
    from .models.input_validation import InputValidationRequest
    from .models.output_validation import OutputValidationRequest


@dataclass(frozen=True, slots=True)
class GuardResult:
    """Aggregated verdict from :meth:`Guardrails.guard_input`/`guard_output`."""

    blocked: bool
    #: Individual policy envelopes, one per guard, in the order supplied.
    policy_envelopes: list[dict[str, Any]] = field(default_factory=list)
    #: The raw ``{"validation", "policies"}`` result from the underlying
    #: :meth:`Client.validate` call — kept so callers can drill in.
    raw: dict[str, Any] | None = None


def _resolve_policy_id(guard: Any) -> str:
    """Extract the policy id from a string or a :class:`BaseGuard`."""
    if isinstance(guard, str):
        return guard
    pid = getattr(guard, "policy_id", None)
    if isinstance(pid, str) and pid.strip():
        return pid
    raise ValueError(
        f"guard must be a policy-id string or expose .policy_id " f"(got {type(guard).__name__})"
    )


class Guardrails:
    """Composable input/output guardrails around a :class:`Client`.

    Args:
        client: An already-configured :class:`Client`. Must have
            ``application_name`` set (policy evaluation requires it).
        input_guards: Policy ids (or :class:`BaseGuard` objects) evaluated
            in :meth:`guard_input` / :meth:`a_guard_input`.
        output_guards: Policy ids (or :class:`BaseGuard` objects) evaluated
            in :meth:`guard_output` / :meth:`a_guard_output`.
        evaluation_model: Reserved — currently unused by the client-side
            fan-out; kept for wire-forward parity with DeepTeam.
        sample_rate: Reserved — server-side sampling is enforced by the
            RuntimePolicy (see Phase 2c). Kept for parity.
    """

    def __init__(
        self,
        client: Client,
        input_guards: list[str | BaseGuard] | None = None,
        output_guards: list[str | BaseGuard] | None = None,
        evaluation_model: str = "gpt-4.1",
        sample_rate: float = 1.0,
    ) -> None:
        self.client = client
        self.input_guards: list[str] = [_resolve_policy_id(g) for g in (input_guards or [])]
        self.output_guards: list[str] = [_resolve_policy_id(g) for g in (output_guards or [])]
        self.evaluation_model = evaluation_model
        self.sample_rate = sample_rate

    # -- sync ------------------------------------------------------------
    def guard_input(self, request: InputValidationRequest | SupportsInputData) -> GuardResult:
        """Evaluate ``input_guards`` against the caller's input.

        Returns a :class:`GuardResult`; callers gate on ``result.blocked``.
        """
        return self._guard(request, self.input_guards)

    def guard_output(
        self,
        request: OutputValidationRequest | SupportsInputData,
    ) -> GuardResult:
        """Evaluate ``output_guards`` against the caller's model output."""
        return self._guard(request, self.output_guards)

    def _guard(self, request: Any, policies: list[str]) -> GuardResult:
        if not policies:
            return GuardResult(blocked=False)
        raw = self.client.validate(request, policies=policies)
        envelopes = raw.get("policies", []) if isinstance(raw, dict) else []
        return GuardResult(
            blocked=any_blocking(raw),
            policy_envelopes=[e for e in envelopes if isinstance(e, dict)],
            raw=raw if isinstance(raw, dict) else None,
        )

    # -- async -----------------------------------------------------------
    async def a_guard_input(
        self, request: InputValidationRequest | SupportsInputData
    ) -> GuardResult:
        """Async variant of :meth:`guard_input` — fans out per-guard in parallel."""
        return await self._a_guard(request, self.input_guards)

    async def a_guard_output(
        self,
        request: OutputValidationRequest | SupportsInputData,
    ) -> GuardResult:
        """Async variant of :meth:`guard_output`."""
        return await self._a_guard(request, self.output_guards)

    async def _a_guard(self, request: Any, policies: list[str]) -> GuardResult:
        if not policies:
            return GuardResult(blocked=False)
        # ponytail: the requests library is sync; fan-out here parks each
        # call in a default thread executor. Fine for small guard counts.
        # Swap for httpx.AsyncClient if fan-out fanout grows > ~10 guards.
        loop = asyncio.get_running_loop()
        envelopes = await asyncio.gather(
            *(loop.run_in_executor(None, self.client.validate, request, [pid]) for pid in policies)
        )
        merged: list[dict[str, Any]] = []
        blocked = False
        for env in envelopes:
            pols = env.get("policies") or []
            for p in pols:
                if isinstance(p, dict):
                    merged.append(p)
                    if is_blocking(p):
                        blocked = True
        return GuardResult(blocked=blocked, policy_envelopes=merged)
