"""``forbidden_calls``: a hard veto. One matching call fails the case whatever the scores."""

from fnmatch import fnmatchcase
from typing import Any

from ...models import Expect, ForbiddenCall, Step
from ..assignment import expect_id, name_ok, pair_eval
from ..config import EvalConfig
from ..matching import MatchContext, describe, match_dict
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


@register_metric("forbidden_calls", "1")
class ForbiddenCalls(BaseMetric):
    rank = 50
    run_on_unfinished = True  # a crashed run can still have made a forbidden call

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {
            "forbidden": [
                {
                    "name": f.name,
                    "args": (
                        None
                        if f.args is None
                        else {k: describe(v) for k, v in sorted(f.args.items())}
                    ),
                    "unless_after": f.unless_after,
                }
                for f in expect.forbidden
            ]
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return bool(expect.forbidden)

    def _exempt(self, ctx: EvalContext, fc: ForbiddenCall, hit: Step) -> bool:
        """Exempt only by an earlier prerequisite that succeeded (``error is None``) and did not
        run concurrently with the hit (same non-None ``parallel_group``). ``unless_after`` is a
        ``ToolExpect.id`` (the model validates this), so there is no name-glob form. The
        prerequisite is any call that fully satisfies that expectation, not only the one the
        assignment bound to it: with a repeated prerequisite the assignment may bind a later
        call."""
        if fc.unless_after is None:
            return False
        bound = ctx.assignment().bound(fc.unless_after)
        candidates = [
            s
            for s in ctx.calls()
            if s.seq < hit.seq
            and s.error is None
            and not (s.parallel_group is not None and s.parallel_group == hit.parallel_group)
        ]
        if any(s in bound for s in candidates):
            return True
        prerequisite = next(
            (te for i, te in enumerate(ctx.expect.tools) if expect_id(te, i) == fc.unless_after),
            None,
        )
        if prerequisite is None:
            return False
        mctx = MatchContext(run=ctx.run, use_judge=False, predicates=ctx.config.predicates)
        for s in candidates:
            if not name_ok(prerequisite, s):
                continue
            ev = pair_eval(prerequisite, s, mctx, ctx.config.min_pair_score)
            if not (ev.failed_leaves or ev.incomplete or ev.errored):
                return True
        return False

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        mctx = MatchContext(run=ctx.run, use_judge=False, predicates=ctx.config.predicates)
        hits: list[dict[str, Any]] = []
        for fc in ctx.expect.forbidden:
            for step in ctx.calls():
                if not fnmatchcase(step.name, fc.name):
                    continue
                if fc.args is not None:
                    mctx.consumer = step
                    leaves = match_dict(fc.args, step.args, mctx, ())
                    if not all(r.status == "ok" and r.ok for r in leaves):
                        continue
                if self._exempt(ctx, fc, step):
                    continue
                hits.append(
                    {
                        "forbidden": fc.name,
                        "call_id": step.call_id,
                        "name": step.name,
                        "turn": step.turn,
                        "reason": fc.reason,
                    }
                )
        if not hits:
            return self.ok(ctx, 1.0, True, details={"hits": []})
        first = hits[0]
        why = f" ({first['reason']})" if first["reason"] else ""
        return self.ok(
            ctx,
            0.0,
            False,
            vetoed=True,
            reason_code="forbidden.called",
            reason=f"forbidden call {first['name']} ({first['call_id']}){why}",
            details={"hits": hits},
        )
