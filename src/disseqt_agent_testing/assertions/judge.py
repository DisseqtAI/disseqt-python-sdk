"""Judge interfaces (implemented in plan 04). Only the contract lives here."""

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..models import FrozenDict
from ..models._frozen import JSONObject, freeze_mapping

UNSURE = "unsure"
NOT_APPLICABLE = "not_applicable"


@dataclass(slots=True, frozen=True)
class JudgeTask:
    """One question for a judge. ``kind`` is an open string; ``prompt_version == ""`` means the
    engine default."""

    kind: str
    prompt_version: str = ""
    inputs: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", freeze_mapping(self.inputs, "JudgeTask.inputs"))


@dataclass(slots=True, frozen=True)
class JudgeVerdict:
    status: Literal["ok", "error"]
    label: str | None = None
    _: KW_ONLY
    value: float | None = None
    reason: str = ""
    model: str = ""
    prompt_hash: str = ""
    samples: int = 1
    agreement: float | None = None
    cached: bool = False
    tokens: int = 0
    cost: float = 0.0
    raw: JSONObject = field(default_factory=FrozenDict)
    error_code: str | None = None
    gating: str = "gate"  # "gate" or "advisory"
    identity_hash: str = ""


@runtime_checkable
class Judge(Protocol):
    def judge(self, task: JudgeTask) -> JudgeVerdict: ...

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict: ...


@dataclass(slots=True, frozen=True)
class Interp:
    status: str
    passed: bool | None
    reason_code: str | None
    details: dict[str, Any] = field(default_factory=dict)


def describe_unsure(v: JudgeVerdict | None) -> str:
    """``agreement 0.33 over 3 samples; orders disagree`` (the text behind ``judge_unsure``)."""
    if v is None or v.agreement is None:
        return "the judge was unsure"
    consistent = v.raw.get("order_consistent") if isinstance(v.raw, Mapping) else None
    tail = "; orders disagree" if consistent is False else ""
    return f"agreement {v.agreement:.2f} over {v.samples} samples{tail}"


def call_record(v: JudgeVerdict, metric: str = "", kind: str = "") -> dict[str, Any]:
    """One judge call as kept in ``DecisionReport.judge_calls`` (works for any ``Judge``)."""
    raw = v.raw if isinstance(v.raw, Mapping) else {}
    return {
        "metric": metric, "kind": str(raw.get("kind") or kind), "identity_hash": v.identity_hash,
        "model": v.model, "prompt_version": str(raw.get("prompt_version", "")), "status": v.status,
        "label": v.label, "agreement": v.agreement, "samples": v.samples, "cached": v.cached,
        "tokens": v.tokens, "cost": v.cost, "error_code": v.error_code, "gating": v.gating,
    }  # fmt: skip


def interpret_verdict(
    v: JudgeVerdict | None,
    positive: tuple[str, ...] | frozenset[str],
    negative: tuple[str, ...] | frozenset[str],
    *,
    na_benign: bool = False,
) -> Interp:
    """The only place a judge verdict becomes pass/fail. Order matters: errors never inspect a
    label, ``unsure`` and advisory verdicts never gate, an unknown label is an instrument error.
    ``passed=True`` is reachable only with ``status="ok"``, ``gating="gate"`` and a positive label.
    """
    if v is None:
        return Interp("skipped", None, "judge_unavailable")
    if v.status == "error":
        return Interp(
            "error", None, "judge_failed", {"error_code": v.error_code, "reason": v.reason}
        )
    if v.label == UNSURE:
        return Interp("skipped", None, "judge_unsure", {"reason": describe_unsure(v)})
    if v.label == NOT_APPLICABLE:
        if na_benign:
            return Interp("skipped", None, "not_applicable")
        return Interp(
            "skipped",
            None,
            "judge_unsure",
            {"label": NOT_APPLICABLE, "reason": "the judge found the check not applicable"},
        )
    if v.gating == "advisory":
        return Interp(
            "skipped",
            None,
            "judge_uncalibrated",
            {"advisory_label": v.label, "reason": "advisory label: " + str(v.label)},
        )
    if v.label in positive:
        return Interp("ok", True, None)
    if v.label in negative:
        return Interp("ok", False, None)
    return Interp("error", None, "judge_failed", {"error_code": "invalid_verdict"})
