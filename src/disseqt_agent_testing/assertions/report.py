"""``DecisionReport``: the one result object. ``to_dict()`` is the single source of truth;
renderers and report writers consume only the dict."""

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar

from ..models import FrozenDict
from ..models._frozen import JSONObject, freeze_json, freeze_mapping
from .canonical import normalize, round6
from .metrics.base import MetricResult

OUTCOMES = ("passed", "failed", "incomplete")


@dataclass(slots=True, frozen=True)
class TriageResult:
    """Advisory root-cause hint for a failed case. Never gates and never changes the outcome."""

    label: str
    _: KW_ONLY
    call_id: str | None = None
    evidence: tuple[str, ...] = ()
    reasoning: str = ""
    agreement: float | None = None
    identity_hash: str = ""
    judged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "call_id": self.call_id, "evidence": list(self.evidence),
            "reasoning": self.reasoning, "agreement": self.agreement,
            "identity_hash": self.identity_hash, "judged": self.judged,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TriageResult":
        return cls(
            str(data["label"]), call_id=data.get("call_id"), evidence=tuple(data.get("evidence", ())),
            reasoning=str(data.get("reasoning", "")), agreement=data.get("agreement"),
            identity_hash=str(data.get("identity_hash", "")), judged=bool(data.get("judged", True)),
        )  # fmt: skip


@dataclass(slots=True, frozen=True)
class DecisionReport:
    SCHEMA_VERSION: ClassVar[str] = "1"

    case_id: str
    outcome: str
    _: KW_ONLY
    harness: JSONObject = field(default_factory=FrozenDict)
    decisions: tuple[JSONObject, ...] = ()
    summary: JSONObject = field(default_factory=FrozenDict)
    scores: JSONObject = field(default_factory=FrozenDict)
    checks: tuple[MetricResult, ...] = ()
    judge_calls: tuple[JSONObject, ...] = ()
    warnings: tuple[str, ...] = ()
    triage: tuple[TriageResult, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        for name in ("harness", "summary", "scores"):
            object.__setattr__(self, name, freeze_json(normalize(getattr(self, name))))
        object.__setattr__(
            self, "decisions", tuple(freeze_json(normalize(d)) for d in self.decisions)
        )
        object.__setattr__(
            self, "judge_calls", tuple(freeze_json(normalize(d)) for d in self.judge_calls)
        )
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "DecisionReport.extra"))

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def __bool__(self) -> bool:
        raise TypeError("DecisionReport has no truth value; use report.passed")

    def check(self, name: str) -> MetricResult | None:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        out = self._base_dict()
        if self.triage:  # additive: absent unless triage ran, so older goldens stay byte-stable
            out["triage"] = [t.to_dict() for t in self.triage]
        return out

    def _base_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "case_id": self.case_id,
            "outcome": self.outcome,
            "harness": normalize(self.harness),
            "decisions": [normalize(d) for d in self.decisions],
            "summary": normalize(self.summary),
            "scores": normalize(self.scores),
            "checks": [c.to_dict() for c in self.checks],
            "judge_calls": [normalize(j) for j in self.judge_calls],
            "warnings": list(self.warnings),
            "extra": normalize(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionReport":
        """Unknown keys are ignored (a newer report loads in an older tool)."""
        return cls(
            data["case_id"],
            data["outcome"],
            harness=data.get("harness", {}),
            decisions=tuple(data.get("decisions", ())),
            summary=data.get("summary", {}),
            scores=data.get("scores", {}),
            checks=tuple(MetricResult.from_dict(c) for c in data.get("checks", ())),
            judge_calls=tuple(data.get("judge_calls", ())),
            warnings=tuple(data.get("warnings", ())),
            triage=tuple(TriageResult.from_dict(t) for t in data.get("triage", ())),
            extra=data.get("extra", {}),
        )

    def render(self, max_lines: int | None = None, color: bool = False) -> str:
        from .render import render

        return render(self.to_dict(), max_lines=max_lines, color=color)


@dataclass(slots=True, frozen=True)
class RepeatReport:
    """Schema only: plan 05 builds it from several ``DecisionReport``s."""

    n: int
    _: KW_ONLY
    n_passed: int = 0
    n_failed: int = 0
    n_incomplete: int = 0
    k: int = 1
    threshold: float = 1.0
    pass_at_1: float | None = None
    pass_hat_k: float | None = None  # None if fewer than k decided runs; never 1.0 by default
    pass_at_k: float | None = None
    wilson_95: tuple[float, float] | None = None
    decision_histogram: JSONObject = field(default_factory=FrozenDict)
    failure_codes: JSONObject = field(default_factory=FrozenDict)
    outcome: str = "incomplete"
    worst: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "n_passed": self.n_passed, "n_failed": self.n_failed,
            "n_incomplete": self.n_incomplete, "k": self.k, "threshold": round6(float(self.threshold)),
            "pass_at_1": _r6(self.pass_at_1), "pass_hat_k": _r6(self.pass_hat_k),
            "pass_at_k": _r6(self.pass_at_k),
            "wilson_95": None if self.wilson_95 is None else [_r6(self.wilson_95[0]), _r6(self.wilson_95[1])],
            "decision_histogram": normalize(self.decision_histogram),
            "failure_codes": normalize(self.failure_codes), "outcome": self.outcome, "worst": self.worst,
        }  # fmt: skip

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RepeatReport":
        wilson = data.get("wilson_95")
        return cls(
            int(data["n"]), n_passed=int(data.get("n_passed", 0)), n_failed=int(data.get("n_failed", 0)),
            n_incomplete=int(data.get("n_incomplete", 0)), k=int(data.get("k", 1)),
            threshold=float(data.get("threshold", 1.0)), pass_at_1=data.get("pass_at_1"),
            pass_hat_k=data.get("pass_hat_k"), pass_at_k=data.get("pass_at_k"),
            wilson_95=None if wilson is None else (float(wilson[0]), float(wilson[1])),
            decision_histogram=data.get("decision_histogram", {}), failure_codes=data.get("failure_codes", {}),
            outcome=str(data.get("outcome", "incomplete")), worst=data.get("worst"),
        )  # fmt: skip


def report_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of ``DecisionReport.to_dict()``. Changes inside schema
    version 1 are additive: new optional properties only, never removed or retyped ones."""
    nullable_str = {"type": ["string", "null"]}
    metric = {
        "type": "object",
        "required": [
            "name", "version", "params_hash", "scope", "status", "value", "passed", "vetoed",
            "required", "weight", "reason_code", "reason", "details", "components", "judged",
        ],  # fmt: skip
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "params_hash": {"type": "string"},
            "scope": {"enum": ["run", "turn"]},
            "turn": {"type": ["integer", "null"]},
            "status": {"enum": ["ok", "skipped", "error"]},
            "value": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "passed": {"type": ["boolean", "null"]},
            "threshold": {"type": ["number", "null"]},
            "vetoed": {"type": "boolean"},
            "required": {"type": "boolean"},
            "weight": {"type": "number"},
            "reason_code": nullable_str,
            "reason": {"type": "string"},
            "details": {"type": "object"},
            "components": {"type": "array", "items": {"$ref": "#/$defs/metric_result"}},
            "judged": {"type": "boolean"},
            "extra": {"type": "object"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DecisionReport",
        "type": "object",
        "required": [
            "schema_version", "case_id", "outcome", "harness", "decisions", "summary", "scores",
            "checks", "judge_calls", "warnings",
        ],  # fmt: skip
        "properties": {
            "schema_version": {"const": DecisionReport.SCHEMA_VERSION},
            "case_id": {"type": "string"},
            "outcome": {"enum": list(OUTCOMES)},
            "harness": {
                "type": "object",
                "required": ["status", "error", "violations", "warnings", "trace_id"],
                "properties": {
                    "status": {"type": "string"},
                    "error": nullable_str,
                    "violations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["code", "message"],
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "fake": nullable_str,
                                "call_id": nullable_str,
                            },
                        },
                    },
                    "warnings": {"type": "array", "items": {"type": "string"}},
                    "trace_id": {"type": "string"},
                },
            },
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["turn", "label", "confidence", "rule"],
                    "properties": {
                        "turn": {"type": "integer"},
                        "label": {"type": "string"},
                        "confidence": {"type": "number"},
                        "rule": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "judged": {"type": "boolean"},
                    },
                },
            },
            "summary": {
                "type": "object",
                "required": [
                    "n_checks",
                    "n_passed",
                    "n_failed",
                    "n_skipped",
                    "n_error",
                    "n_incomplete",
                ],
                "properties": {
                    k: {"type": "integer", "minimum": 0}
                    for k in (
                        "n_checks",
                        "n_passed",
                        "n_failed",
                        "n_skipped",
                        "n_error",
                        "n_incomplete",
                    )
                },
            },
            "scores": {"type": "object", "additionalProperties": {"type": "number"}},
            "checks": {"type": "array", "items": {"$ref": "#/$defs/metric_result"}},
            "judge_calls": {"type": "array", "items": {"type": "object"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "triage": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["label"],
                    "properties": {
                        "label": {"type": "string"},
                        "call_id": nullable_str,
                        "evidence": {"type": "array", "items": {"type": "string"}},
                        "reasoning": {"type": "string"},
                        "agreement": {"type": ["number", "null"]},
                        "identity_hash": {"type": "string"},
                        "judged": {"type": "boolean"},
                    },
                },
            },
            "extra": {"type": "object"},
        },
        "$defs": {"metric_result": metric},
    }


def _r6(value: float | None) -> float | None:
    return None if value is None else round6(float(value))
