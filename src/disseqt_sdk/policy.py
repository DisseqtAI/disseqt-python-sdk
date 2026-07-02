"""Helpers for the realtime-policy evaluation response.

The server-side endpoint POST /api/v1/sdk/policies/{policy_id}/evaluate
returns a structured response with a per-rule breakdown. These helpers
parse it into typed dataclasses and answer the common "should I block?"
and "is this sync or async?" checks without callers poking at dict keys.

Wire shape
----------
prod-monitoring wraps the verdict in the standard Disseqt DSQ envelope::

    {
      "status":        "success",
      "data":          { "policy_id": "...", "decision": "BLOCK", ... },
      "messages":      [],
      "code":          "DSQ-2000",
      "standard_code": "OK",
      "request_id":    "req_abc",
      "timestamp":     "2026-06-27T15:30:18Z"
    }

The helpers in this module accept either the full envelope OR the
``data`` payload directly — _unwrap_envelope() handles both, so caller
code is the same whether you pass ``client.evaluate_policy(...)``'s
return value or ``response["data"]``.

Response fields inside ``data``
-------------------------------
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


def _unwrap_envelope(response: dict[str, Any]) -> dict[str, Any]:
    """Return ``response["data"]`` when the response is a DSQ envelope,
    else ``response`` itself.

    A DSQ envelope is recognised by ``status == "success"`` together with
    a dict-valued ``data`` field — that's the shape prod-monitoring's
    /policies/:id/evaluate returns. Anything else (a raw payload from a
    legacy or non-prod-monitoring caller, or an error envelope where
    ``data`` is missing/None) falls through unchanged, so existing
    callers that already pass the unwrapped dict keep working.
    """
    if response.get("status") == "success":
        data = response.get("data")
        if isinstance(data, dict):
            return data
    return response


def parse(response: dict[str, Any]) -> PolicyDecision | None:
    """Turn a /policies/:id/evaluate response into a PolicyDecision.

    Accepts either the full DSQ envelope or the unwrapped ``data`` dict.
    Returns None when the response doesn't carry a policy verdict (e.g.
    server returned an error envelope with no policy_id).
    """
    payload = _unwrap_envelope(response)
    if not payload.get("policy_id"):
        return None
    rulesets: list[PolicyRuleset] = []
    for rs in payload.get("rulesets", []) or []:
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
        policy_id=str(payload.get("policy_id", "")),
        policy_name=str(payload.get("policy_name", "")),
        policy_version=int(payload.get("policy_version", 0)),
        decision=str(payload.get("decision", "")),
        enforcement=str(payload.get("enforcement", "")),
        rulesets=rulesets,
    )


def is_blocking(response: dict[str, Any]) -> bool:
    """Return True when the policy verdict is BLOCK.

    Accepts either the full DSQ envelope or the unwrapped ``data`` dict.
    Convenience for the common "do not pass this output downstream"
    check. Reads ``decision``, which is the actual verdict — independent
    of sync/async.
    """
    payload = _unwrap_envelope(response)
    return str(payload.get("decision", "")) == DECISION_BLOCK


def is_async(response: dict[str, Any]) -> bool:
    """Return True when the policy ran in async mode.

    Accepts either the full DSQ envelope or the unwrapped ``data`` dict.
    In async mode the decision in this response is not yet final — the
    real result will land on the realtime-validations dashboard once
    background processing completes. Callers usually log and move on
    instead of acting on ``is_blocking``.
    """
    payload = _unwrap_envelope(response)
    return str(payload.get("enforcement", "")) == ENFORCEMENT_ASYNC


def any_blocking(result: Any) -> bool:
    """Return True when any policy decision in ``result`` is BLOCK.

    Accepts, in order of preference:

    - the dict returned by ``client.validate(..., policies=[...])``
      (reads its ``"policies"`` list),
    - a plain list of policy envelopes,
    - a single policy envelope (falls back to :func:`is_blocking`).

    Anything else — including a classic validator response — returns
    False, so it is always safe to gate on::

        result = client.validate(req, policies=[...])
        if any_blocking(result):
            ...  # at least one policy said BLOCK
    """
    if isinstance(result, dict) and isinstance(result.get("policies"), list):
        return any(is_blocking(p) for p in result["policies"] if isinstance(p, dict))
    if isinstance(result, list):
        return any(is_blocking(p) for p in result if isinstance(p, dict))
    if isinstance(result, dict):
        return is_blocking(result)
    return False


def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
