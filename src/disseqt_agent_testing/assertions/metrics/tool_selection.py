"""``tool_selection``: were the right tools called, no more and no fewer?"""

from fnmatch import fnmatchcase
from typing import Any

from ...models import Expect, Step
from ..assignment import name_ok
from ..config import EvalConfig
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


def describe_call(step: Step) -> dict[str, Any]:
    return {"call_id": step.call_id, "name": step.name, "effect": step.effect, "turn": step.turn}


@register_metric("tool_selection", "1")
class ToolSelection(BaseMetric):
    rank = 20
    sub_scores = ("precision", "recall", "f1")
    thresholdable = True

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {
            "tools": [
                {"name": t.name, "count": t.count, "turn": t.turn, "id": t.id} for t in expect.tools
            ],
            "allowed_tools": expect.allowed_tools,
            "extra_calls": expect.extra_calls,
            "backend": "any",  # every backend returns the same assignment value
            "min_pair_score": config.min_pair_score,
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return (
            bool(expect.tools) or expect.allowed_tools is not None or expect.extra_calls == "forbid"
        )

    def failure_within(self, result: MetricResult, sub_scores: frozenset[str]) -> bool:
        """Recall measures missing calls and precision measures unexpected ones; F1 measures both.
        (Hallucinated, disallowed and over-count calls are vetoes, which no threshold clears.)"""
        details = result.details
        if details.get("missing") and not sub_scores & {"recall", "f1"}:
            return False
        if details.get("unexpected") and not sub_scores & {"precision", "f1"}:
            return False
        return True

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        a = ctx.assignment()
        run, expect = ctx.run, ctx.expect
        offered = {t.name for t in run.available_tools}
        hallucinated = [s for s in a.calls if s.name not in offered] if offered else []
        allowed = expect.allowed_tools
        not_allowed = (
            [s for s in a.calls if not any(fnmatchcase(s.name, g) for g in allowed)]
            if allowed is not None
            else []
        )
        capped = [t for t in expect.tools if isinstance(t.count, tuple) and t.count[1] is not None]
        wrong_count = [s for s in a.unmatched_calls if any(name_ok(t, s) for t in capped)]
        wc_ids = {s.call_id for s in wrong_count}
        policy = expect.extra_calls
        if policy == "forbid":
            unexpected = list(a.unmatched_calls)
        elif policy == "allow_reads":
            unexpected = [s for s in a.unmatched_calls if s.effect != "read"]
        else:
            unexpected = []
        required = a.n_required
        matched_required = sum(1 for p in a.pairs if p.slot.required)
        matched = len(a.pairs)
        recall = matched_required / required if required else None
        denominator = matched + len(unexpected)
        precision = matched / denominator if denominator else None
        if recall is not None and precision is not None:
            f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
        else:
            f1 = None
        vetoed = bool(
            hallucinated or not_allowed or wrong_count or (policy == "forbid" and unexpected)
        )
        passed = (recall in (None, 1.0)) and not (
            unexpected or hallucinated or not_allowed or wrong_count
        )
        value = (
            f1
            if f1 is not None
            else (recall if recall is not None else (precision if precision is not None else 1.0))
        )
        code, reason = None, ""
        if not passed:
            if hallucinated:
                code, reason = (
                    "tool.hallucinated",
                    f"called tools that were not offered: {sorted({s.name for s in hallucinated})}",
                )
            elif not_allowed:
                code, reason = (
                    "tool.not_allowed",
                    f"called tools outside allowed_tools: {sorted({s.name for s in not_allowed})}",
                )
            elif wrong_count:
                code, reason = (
                    "wrong_count",
                    f"more calls than allowed: {sorted({s.name for s in wrong_count})}",
                )
            elif a.missing_required:
                code, reason = (
                    "tool.missing",
                    f"missing required calls: {[s.expect_id for s in a.missing_required]}",
                )
            else:
                code, reason = (
                    "tool.unexpected",
                    f"unexpected calls: {[s.name for s in unexpected]}",
                )
        details = {
            "scores": {"precision": precision, "recall": recall, "f1": f1},
            "matched": [{"expect": p.slot.expect_id, "call": p.step.call_id} for p in a.pairs],
            "missing": [s.expect_id for s in a.missing_required],
            "unexpected": [describe_call(s) for s in unexpected],
            "hallucinated": [describe_call(s) for s in hallucinated],
            "not_allowed": [describe_call(s) for s in not_allowed],
            "wrong_count": [describe_call(s) for s in wrong_count],
            "required": required,
            "extra_calls": policy,
            "allowed_reads": [
                describe_call(s)
                for s in a.unmatched_calls
                if s.effect == "read" and s.call_id not in wc_ids
            ],
        }
        return self.ok(
            ctx, value, passed, vetoed=vetoed, reason_code=code, reason=reason, details=details
        )
