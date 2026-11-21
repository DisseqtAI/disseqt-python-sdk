"""``decision_match``: did each expected turn go ``call | ask | refuse | answer``?"""

from typing import Any

from ...models import Expect
from ..classify import turn_count
from ..config import EvalConfig
from .base import BaseMetric, EvalContext, MetricResult, first_reason, rollup
from .registry import register_metric


@register_metric("decision_match", "1")
class DecisionMatch(BaseMetric):
    scope = "turn"
    rank = 10

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        turns = {str(t): [d.value for d in ds] for t, ds in sorted(expect.turn_decisions.items())}
        return {
            "turns": turns,
            "decision_source": config.decision_source,
            "lang": config.lang,
            "judge": config.use_judge,
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return bool(expect.turn_decisions)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        comps = [
            self._turn(ctx, turn, [d.value for d in labels])
            for turn, labels in sorted(ctx.expect.turn_decisions.items())
        ]
        status, value, passed, code = rollup(comps)
        judged = any(c.judged for c in comps)
        if status == "ok":
            return self.ok(ctx, value, bool(passed), components=tuple(comps), judged=judged)
        if status == "error":
            return self.error(
                ctx, code or "judge_failed", first_reason(comps), components=tuple(comps)
            )
        return self.skip(
            ctx, code or "not_applicable", first_reason(comps), components=tuple(comps)
        )

    def _turn(self, ctx: EvalContext, turn: int, expected: list[str]) -> MetricResult:
        kw: dict[str, Any] = {"turn": turn, "scope": "turn"}
        if turn >= turn_count(ctx.run):
            return self.ok(
                ctx,
                0.0,
                False,
                reason_code="decision.turn_missing",
                reason=f"expected a decision on turn {turn}, the run has {turn_count(ctx.run)}",
                details={"expected": expected, "actual": None},
                **kw,
            )
        cls = ctx.classification(turn)
        details = {
            "expected": expected,
            "actual": cls.label,
            "confidence": cls.confidence,
            "rule": cls.rule,
            "evidence": list(cls.evidence),
            "low_confidence": cls.low_confidence,
        }
        if cls.advisory_label is not None:  # only when an uncalibrated judge disagreed in spirit
            details["advisory_label"] = cls.advisory_label
        undecided = self.undecided(ctx, cls, details=details, **kw)
        if undecided is not None:
            return undecided
        if cls.label == "empty":
            return self.ok(
                ctx,
                0.0,
                False,
                reason_code="decision.empty",
                reason="the agent produced no output",
                details=details,
                judged=cls.judged,
                **kw,
            )
        if cls.label in expected:
            return self.ok(ctx, 1.0, True, details=details, judged=cls.judged, **kw)
        return self.ok(
            ctx,
            0.0,
            False,
            reason_code="decision.mismatch",
            reason=f"expected {' or '.join(expected)}, got {cls.label} ({cls.rule})",
            details=details,
            judged=cls.judged,
            **kw,
        )
