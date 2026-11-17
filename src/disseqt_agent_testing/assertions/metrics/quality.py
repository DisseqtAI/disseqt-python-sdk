"""``answer_quality`` and ``groundedness``: judged checks of the final answer.

Both read a judge verdict whose label the judge engine derived in code from structured evidence
(per-criterion booleans, per-claim verdicts); with no judge the check is skipped, never passed.
"""

from fnmatch import fnmatchcase
from typing import Any

from ...models import AnswerQualityExpect, Expect, GroundednessExpect
from ..config import EvalConfig
from ..judge import JudgeTask, JudgeVerdict, interpret_verdict
from ..judge_inputs import conversation, tool_calls
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric

REFERENCE_CRITERION = "The answer states the same facts and conclusion as the reference answer: {}"


def judged_outcome(
    metric: BaseMetric,
    ctx: EvalContext,
    verdict: JudgeVerdict | None,
    positive: tuple[str, ...],
    negative: tuple[str, ...],
    *,
    na_benign: bool = False,
) -> "tuple[MetricResult | None, bool | None]":
    """``(early_result, passed)``: an early result for error/skip verdicts, else the pass flag."""
    interp = interpret_verdict(verdict, positive, negative, na_benign=na_benign)
    if interp.status == "error":
        return metric.error(
            ctx, interp.reason_code or "judge_failed",
            str(interp.details.get("reason") or "the judge failed"),
        ), None  # fmt: skip
    if interp.status == "skipped":
        return metric.skip(
            ctx, interp.reason_code or "judge_unavailable",
            str(interp.details.get("reason") or "needs a judge"),
        ), None  # fmt: skip
    return None, bool(interp.passed)


@register_metric("answer_quality", "1")
class AnswerQuality(BaseMetric):
    rank = 75

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        a = expect.answer_quality
        assert a is None or isinstance(a, AnswerQualityExpect)
        return {"reference": a.reference if a else None, "accept": a.accept if a else None,
                "criteria": list(a.criteria) if a else []}  # fmt: skip

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.answer_quality is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        spec = ctx.expect.answer_quality
        assert isinstance(spec, AnswerQualityExpect)
        text = ctx.run.final_text
        if text is None:
            return self.ok(ctx, 0.0, False, reason_code="answer_quality.no_output",
                           reason="the agent produced no answer")  # fmt: skip
        criteria = [REFERENCE_CRITERION.format(spec.reference), *spec.criteria]
        verdict = ctx.request_judge(
            JudgeTask("answer_quality", inputs={"conversation": conversation(ctx.run, 0),
                                                "agent_text": text, "criteria": criteria}),
            "answer_quality",
        )  # fmt: skip
        early, passed = judged_outcome(self, ctx, verdict, ("pass",), ("fail",))
        if early is not None:
            return early
        met: list[dict[str, Any]] = []
        if verdict is not None:
            raw = verdict.raw.get("derived")
            met = [dict(c) for c in raw["criteria"]] if isinstance(raw, dict) else []
        share = (sum(1 for c in met if c["met"]) / len(met)) if met else (1.0 if passed else 0.0)
        reference_met = bool(met and met[0]["met"])
        # "partially_correct" accepts an answer that matches the reference but misses extras
        accepted = bool(passed) or (spec.accept == "partially_correct" and reference_met)
        details = {"criteria_met": met, "accept": spec.accept}
        if accepted:
            return self.ok(ctx, share, True, judged=True, details=details)
        return self.ok(ctx, share, False, judged=True, reason_code="answer_quality.below_bar",
                       reason=f"{sum(1 for c in met if c['met'])} of {len(met)} criteria met",
                       details=details)  # fmt: skip


@register_metric("groundedness", "1")
class Groundedness(BaseMetric):
    rank = 80

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        g = expect.grounded
        assert g is None or isinstance(g, GroundednessExpect)
        return {"sources": list(g.sources) if g else [],
                "user": g.include_user_statements if g else None,
                "allow_unsupported": g.allow_unsupported if g else None}  # fmt: skip

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return isinstance(expect.grounded, GroundednessExpect)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        spec = ctx.expect.grounded
        assert isinstance(spec, GroundednessExpect)
        text = ctx.run.final_text
        if text is None:
            return self.ok(ctx, 0.0, False, reason_code="groundedness.no_output",
                           reason="the agent produced no answer")  # fmt: skip
        last = max(len(ctx.run.turn_outputs) - 1, 0)
        sources = [
            c for t in range(last + 1) for c in tool_calls(ctx.run, t)
            if any(fnmatchcase(str(c["name"]), pat) for pat in spec.sources)
        ]  # fmt: skip
        context = (
            "\n".join(m["text"] for m in conversation(ctx.run, last) if m["role"] == "user")
            if spec.include_user_statements else ""
        )  # fmt: skip
        verdict = ctx.request_judge(
            JudgeTask("groundedness", inputs={"agent_text": text, "tool_calls": sources,
                                              "context": context}),
            "groundedness",
        )  # fmt: skip
        early, passed = judged_outcome(self, ctx, verdict, ("grounded",), ("ungrounded",),
                                       na_benign=True)  # fmt: skip
        if early is not None:
            return early
        derived = verdict.raw.get("derived") if verdict is not None else None
        counts = dict(derived["counts"]) if isinstance(derived, dict) else {}
        details = {"counts": counts}
        if passed:
            return self.ok(ctx, 1.0, True, judged=True, details=details)
        if spec.allow_unsupported and not counts.get("contradicted"):
            return self.ok(ctx, 1.0, True, judged=True, details={**details, "allowed": True})
        code = (
            "groundedness.contradicted"
            if counts.get("contradicted")
            else "groundedness.unsupported"
        )
        return self.ok(ctx, 0.0, False, judged=True, reason_code=code,
                       reason="the answer makes claims its sources do not support",
                       details=details)  # fmt: skip
