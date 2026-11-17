"""``plan``: named constraints over the whole run, plus an informational convergence score."""

import dataclasses
from typing import Any

from ...models import Expect, PlanExpect, thaw, to_dict
from ..config import EvalConfig
from ..matching.registry import MatchResult
from ..predicates import lookup_predicate
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


def canonical_args(args: Any) -> str:
    import json

    return json.dumps(thaw(args), sort_keys=True, default=str)[:200]


def _verdict(out: Any) -> tuple[bool, str] | None:
    if isinstance(out, bool):
        return out, ""
    if isinstance(out, MatchResult):
        return bool(out.ok), out.reason
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], bool):
        return out[0], str(out[1])
    return None


@register_metric("plan", "2")
class Plan(BaseMetric):
    rank = 140
    sub_scores = ("micro", "macro", "convergence")
    thresholdable = True

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        p = expect.plan
        assert p is None or isinstance(p, PlanExpect)
        return {"constraints": list(p.constraints) if p else [],
                "optimal": p.optimal_tool_calls if p else None,
                "judge_quality": p.judge_quality if p else False}  # fmt: skip

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.plan is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        result = self._constraints(ctx)
        spec = ctx.expect.plan
        assert isinstance(spec, PlanExpect)
        if not spec.judge_quality or result.status != "ok" or not result.passed:
            return result  # a failed or errored constraint is already decisive
        return self._judge_plan(ctx, result)

    def _constraints(self, ctx: EvalContext) -> MetricResult:
        spec = ctx.expect.plan
        assert isinstance(spec, PlanExpect)
        convergence = (
            min(1.0, spec.optimal_tool_calls / max(len(ctx.calls()), 1))
            if spec.optimal_tool_calls is not None
            else None
        )
        rows: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for name in spec.constraints:
            fn = lookup_predicate(name, ctx.config.predicates)
            try:
                got = _verdict(fn(to_dict(ctx.run)))
            except Exception as exc:
                errors.append({"name": name, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if got is None:
                errors.append(
                    {
                        "name": name,
                        "reason": "returned something other than bool, (bool, reason) or MatchResult",
                    }
                )
                continue
            rows.append({"name": name, "passed": got[0], "reason": got[1]})
        total = len(spec.constraints)
        n_pass = sum(1 for r in rows if r["passed"])
        failing = [r for r in rows if not r["passed"]]
        micro = n_pass / total if total else 1.0
        scores = {"micro": micro, "macro": 1.0 if total and n_pass == total else (1.0 if not total else 0.0),
                  "convergence": convergence}  # fmt: skip
        details = {"scores": scores, "constraints": rows, "errors": errors}
        if failing:
            # vetoed: a convergence threshold may relax the score, never clear a failed rule
            return self.ok(ctx, micro, False, vetoed=True, reason_code="plan.constraint_failed",
                           reason=f"constraint {failing[0]['name']} failed {failing[0]['reason']}".strip(),
                           details=details)  # fmt: skip
        if errors:
            return self.error(
                ctx,
                "predicate_error",
                f"constraint {errors[0]['name']}: {errors[0]['reason']}",
                details=details,
            )
        return self.ok(ctx, micro, True, details=details)

    def _judge_plan(self, ctx: EvalContext, base: MetricResult) -> MetricResult:
        from ..judge import JudgeTask
        from ..judge_inputs import conversation
        from .quality import judged_outcome

        steps = [f"{c.name}({canonical_args(c.args)})" for c in ctx.calls()]
        goal = " ".join(m["text"] for m in conversation(ctx.run, 0) if m["role"] == "user")
        verdict = ctx.request_judge(
            JudgeTask("plan_quality", inputs={
                "goal": goal, "plan": steps,
                "available_tools": [t.name for t in ctx.run.available_tools],
            }),
            "plan",
        )  # fmt: skip
        early, passed = judged_outcome(self, ctx, verdict, ("sound",), ("flawed",))
        if early is not None:
            return early
        derived = verdict.raw.get("derived") if verdict is not None else None
        flaws = [dict(f) for f in derived["flaws"]] if isinstance(derived, dict) else []
        details = {**base.details, "flaws": flaws}
        if passed:
            return dataclasses.replace(base, judged=True, details=details)
        return dataclasses.replace(
            base, value=0.0, passed=False, vetoed=True, judged=True, details=details,
            reason_code="plan_quality.flawed",
            reason=f"the plan judge found {len(flaws)} flaw(s)"
            + (f", first: {flaws[0]['code']} at step {flaws[0]['step']}" if flaws else ""),
        )  # fmt: skip
