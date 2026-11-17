"""``tool_args``: for each matched call, do the arguments match what was expected?"""

from typing import Any

from ...models import Expect
from ..canonical import path_str
from ..config import EvalConfig
from ..matching import describe
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


@register_metric("tool_args", "1")
class ToolArgs(BaseMetric):
    rank = 30
    sub_scores = ("arg_score", "arg_exact")
    thresholdable = True

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        from ..matching import describe as _d

        return {
            "tools": [
                {
                    "name": t.name,
                    "args": {k: _d(v) for k, v in sorted(t.args.items())},
                    "mode": t.args_mode,
                    "ignore": list(t.ignore),
                    "defaults": {k: _d(v) for k, v in sorted(t.defaults.items())},
                }
                for t in expect.tools
            ],
            "min_pair_score": config.min_pair_score,
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return any(t.args or t.args_mode == "exact" for t in expect.tools)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        a = ctx.assignment()
        if not a.pairs:
            return self.skip(
                ctx, "not_applicable", "no call was matched, so there are no arguments to compare"
            )
        diffs: list[dict[str, Any]] = []
        n_failed = n_skipped = n_error = 0
        skip_code = "judge_unavailable"
        err_code = "predicate_error"
        exact = 0
        scores = []
        judged = False
        for pair in a.pairs:
            scores.append(pair.eval.score)
            bad = False
            for leaf in pair.eval.leaves:
                judged = judged or leaf.judged
                if leaf.status == "ok" and leaf.ok is False:
                    bad = True
                    n_failed += 1
                elif leaf.status == "skipped":
                    if n_skipped == 0:
                        skip_code = leaf.code or skip_code
                    n_skipped += 1
                elif leaf.status == "error":
                    if n_error == 0:
                        err_code = leaf.code or err_code
                    n_error += 1
                else:
                    continue
                diffs.append(
                    {
                        "call_id": pair.step.call_id,
                        "expect": pair.slot.expect_id,
                        "path": list(leaf.path),
                        "path_str": path_str(leaf.path),
                        "code": leaf.code,
                        "expected": leaf.expected,
                        "actual": leaf.actual,
                        "reason": leaf.reason,
                        "status": leaf.status,
                        **({"ref": dict(leaf.details)} if leaf.details else {}),
                    }
                )
            if not bad and not pair.eval.incomplete and not pair.eval.errored:
                exact += 1
        diffs.sort(key=lambda d: (d["call_id"], d["path_str"]))
        arg_score = sum(scores) / len(scores)
        details = {
            "scores": {"arg_score": arg_score, "arg_exact": exact / len(a.pairs)},
            "pairs": diffs,
        }
        if n_failed:
            first = next(d for d in diffs if d["status"] == "ok")
            return self.ok(
                ctx, arg_score, False, reason_code=first["code"], judged=judged,
                reason=f"{n_failed} argument(s) wrong", details=details,
            )  # fmt: skip
        if n_error:
            return self.error(ctx, err_code, "an argument check could not run", details=details)
        if n_skipped:
            return self.skip(
                ctx, skip_code, "some arguments need a judge",
                details={**details, "partial_value": arg_score},
            )  # fmt: skip
        return self.ok(ctx, arg_score, True, judged=judged, details=details)


__all__ = ["ToolArgs", "describe"]
