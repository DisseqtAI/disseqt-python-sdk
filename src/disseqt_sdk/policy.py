"""Helpers for the realtime-policy evaluation response.

The server-side endpoint POST /api/v1/sdk/policies/{policy_id}/evaluate
returns a structured response with a per-rule breakdown. These helpers
parse it into typed dataclasses and answer the common "should I block?"
and "is this sync or async?" checks without callers poking at dict keys.

Response fields you actually care about
---------------------------------------
* ``decision`` — ``"BLOCK"`` | ``"PASS"`` — the policy's verdict.
* ``enforcement`` — ``"sync"`` | ``"async"`` — mirrors the policy's
  ``strategy.executionMode``. Tells you whether the decision in this
  response is final, or whether the result will land later on the
  realtime-validations dashboard.

For sync policies you usually only check ``is_blocking()``. For async
policies the server will return without a final ``decision`` and the
caller just records the request id and moves on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# decision values
DECISION_BLOCK = "BLOCK"
DECISION_PASS = "PASS"

# enforcement values — these mirror the policy's strategy.executionMode
ENFORCEMENT_SYNC = "sync"
ENFORCEMENT_ASYNC = "async"


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One validator's outcome within a policy."""

    validator: str
    validator_type: str
    status: str  # pass | fail | skipped | error
    score: float | None = None
    threshold: float | None = None
    polarity: str = ""  # quality | risk
    is_decider: bool = False
    skipped_reason: str = ""


@dataclass(frozen=True, slots=True)
class PolicyRuleset:
    """One ruleset (named group of validators) inside the policy."""

    ruleset_id: str
    ruleset_name: str
    required: bool = False
    rules: list[PolicyRule] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Aggregated verdict from a /policies/:id/evaluate call."""

    policy_id: str
    policy_name: str
    policy_version: int
    decision: str  # BLOCK | PASS
    enforcement: str  # sync | async
    rulesets: list[PolicyRuleset] = field(default_factory=list)


def parse(response: dict[str, Any]) -> PolicyDecision | None:
    """Turn a /policies/:id/evaluate response into a PolicyDecision.

    Returns None when the response doesn't carry a policy verdict (e.g.
    server returned an error envelope with success=false and no policy_id).
    """
    if not response.get("policy_id"):
        return None
    rulesets: list[PolicyRuleset] = []
    for rs in response.get("rulesets", []) or []:
        rules: list[PolicyRule] = []
        for r in rs.get("rules", []) or []:
            rules.append(
                PolicyRule(
                    validator=str(r.get("validator", "")),
                    validator_type=str(r.get("validator_type", "")),
                    status=str(r.get("status", "")),
                    score=_maybe_float(r.get("score")) if r.get("has_score") else None,
                    threshold=_maybe_float(r.get("threshold")),
                    polarity=str(r.get("polarity", "")),
                    is_decider=bool(r.get("is_decider", False)),
                    skipped_reason=str(r.get("skipped_reason", "")),
                )
            )
        rulesets.append(
            PolicyRuleset(
                ruleset_id=str(rs.get("ruleset_id", "")),
                ruleset_name=str(rs.get("ruleset_name", "")),
                required=bool(rs.get("required", False)),
                rules=rules,
            )
        )
    return PolicyDecision(
        policy_id=str(response.get("policy_id", "")),
        policy_name=str(response.get("policy_name", "")),
        policy_version=int(response.get("policy_version", 0)),
        decision=str(response.get("decision", "")),
        enforcement=str(response.get("enforcement", "")),
        rulesets=rulesets,
    )


def is_blocking(response: dict[str, Any]) -> bool:
    """Return True when the policy verdict is BLOCK.

    Convenience for the common "do not pass this output downstream" check.
    Reads ``decision``, which is the actual verdict — independent of
    sync/async.
    """
    return str(response.get("decision", "")) == DECISION_BLOCK


def is_async(response: dict[str, Any]) -> bool:
    """Return True when the policy ran in async mode.

    In async mode the decision in this response is not yet final — the
    real result will land on the realtime-validations dashboard once
    background processing completes. Callers usually log and move on
    instead of acting on ``is_blocking``.
    """
    return str(response.get("enforcement", "")) == ENFORCEMENT_ASYNC


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
