"""``order``: ordering constraints, checked on the shared assignment (so a wrong argument is
never also counted as an order failure)."""

from typing import Any

from ...models import Expect, Step
from ..assignment import expect_id
from ..config import EvalConfig
from .base import BaseMetric, EvalContext, MetricResult, first_reason, rollup
from .registry import register_metric


def longest_increasing(seq: list[int]) -> int:
    """Length of the longest strictly increasing subsequence (O(n^2); n is tiny)."""
    best = [1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


@register_metric("order", "1")
class Order(BaseMetric):
    rank = 40
    sub_scores = ()
    thresholdable = True

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {
            "order": expect.order,
            "edges": [
                [after, expect_id(t, i)] for i, t in enumerate(expect.tools) for after in t.after
            ],
            "same_step": [
                [expect_id(t, i), t.same_step_as]
                for i, t in enumerate(expect.tools)
                if t.same_step_as
            ],
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.order != "any" or any(t.after or t.same_step_as for t in expect.tools)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        comps: list[MetricResult] = []
        if ctx.expect.order in ("in_order", "exact"):
            comps.append(self._sequence(ctx))
        edges = [
            (after, expect_id(t, i)) for i, t in enumerate(ctx.expect.tools) for after in t.after
        ]
        if edges:
            comps.append(self._edges(ctx, edges))
        same = [
            (expect_id(t, i), t.same_step_as)
            for i, t in enumerate(ctx.expect.tools)
            if t.same_step_as
        ]
        if same:
            comps.append(self._same_step(ctx, same))
        if not comps:  # order="partial" alone: no sequence constraint beyond edges
            return self.ok(ctx, 1.0, True, details={"mode": ctx.expect.order})
        status, value, passed, code = rollup(comps)
        details = {"mode": ctx.expect.order}
        if status == "ok":
            first_bad = next((c for c in comps if c.passed is False), None)
            return self.ok(
                ctx, value, bool(passed), components=tuple(comps), details=details,
                reason_code=first_bad.reason_code if first_bad else None,
                reason=first_bad.reason if first_bad else "",
            )  # fmt: skip
        if status == "error":
            return self.error(
                ctx,
                code or "metric_error",
                first_reason(comps),
                components=tuple(comps),
                details=details,
            )
        return self.skip(
            ctx,
            code or "not_applicable",
            first_reason(comps),
            components=tuple(comps),
            details=details,
        )

    def _component(self, ctx: EvalContext, suffix: str, **fields: Any) -> MetricResult:
        return MetricResult(
            f"order.{suffix}", version=self.version, params_hash=self._hash(ctx), **fields
        )

    def _sequence(self, ctx: EvalContext) -> MetricResult:
        a = ctx.assignment()
        mode = ctx.expect.order
        if mode == "in_order":
            seqs = [p.step.seq for p in a.pairs]  # pairs are in declared (slot) order
            if not seqs:
                return self._component(
                    ctx, "sequence", status="skipped", reason_code="not_applicable",
                    reason="no call was matched",
                )  # fmt: skip
            lis = longest_increasing(seqs)
            ok = lis == len(seqs)
            return self._component(
                ctx, "sequence", status="ok", value=lis / len(seqs), passed=ok,
                reason_code=None if ok else "order.sequence",
                reason="" if ok else f"observed order {[p.step.name for p in sorted(a.pairs, key=lambda p: p.step.seq)]}",
                details={"mode": mode, "lis": lis, "matched": len(seqs), "seqs": seqs},
            )  # fmt: skip
        # exact: every required call, no extras, expectations in declared order. Optional slots
        # that no call filled are not expected, and calls of one expectation are interchangeable,
        # so the matched calls are compared by expectation rather than by slot position.
        by_call = sorted(a.pairs, key=lambda p: p.call_index)
        width = len(by_call) + 1
        positioned = longest_increasing(
            [p.slot.expect_index * width + k for k, p in enumerate(by_call)]
        )
        want = len(a.pairs) + len(a.missing_required)
        got = len(a.calls)
        ok = positioned == len(by_call) and not a.missing_required and not a.unmatched_calls
        value = positioned / max(want, got, 1)
        declared = [te.name for te in ctx.expect.tools]
        return self._component(
            ctx, "sequence", status="ok", value=value, passed=ok,
            reason_code=None if ok else "order.exact",
            reason="" if ok else f"expected exactly {declared} in order, observed {[s.name for s in a.calls]}",
            details={"mode": mode, "expected": want, "observed": got, "positioned": positioned},
        )  # fmt: skip

    def _edges(self, ctx: EvalContext, edges: list[tuple[str, str]]) -> MetricResult:
        a = ctx.assignment()
        required = {s.expect_id for s in a.slots if s.required}
        violated: list[dict[str, Any]] = []
        satisfied: list[dict[str, Any]] = []
        not_checked: list[dict[str, Any]] = []
        for u, v in edges:
            bu, bv = a.bound(u), a.bound(v)
            row = {"before": u, "after": v}
            missing = [x for x, b in ((u, bu), (v, bv)) if not b]
            if missing:
                if any(x in required for x in missing):
                    violated.append({**row, "code": "order.edge_unverifiable", "missing": missing})
                else:  # an optional call that never happened constrains nothing
                    not_checked.append({**row, "missing": missing})
                continue
            first_u, first_v = bu[0], bv[0]  # bound calls are in seq order
            if (
                first_u.parallel_group is not None
                and first_u.parallel_group == first_v.parallel_group
            ):
                violated.append({**row, "code": "order.edge_same_batch"})
                continue
            if first_u.seq < first_v.seq:
                satisfied.append(row)
            else:
                violated.append({**row, "code": "order.edge_violated",
                                 "seqs": [first_u.seq, first_v.seq]})  # fmt: skip
        checked = len(satisfied) + len(violated)
        if not checked:
            return self._component(
                ctx, "edges", status="skipped", reason_code="not_applicable",
                reason="no edge has both of its calls",
                details={"satisfied": [], "violated": [], "not_checked": not_checked},
            )  # fmt: skip
        ok = not violated
        first = violated[0] if violated else None
        reason = ""
        if first is not None:
            names = [s.name for s in a.calls]
            reason = (
                f"expected {first['before']} before {first['after']}; observed {' -> '.join(names)}"
            )
        return self._component(
            ctx, "edges", status="ok", value=len(satisfied) / checked, passed=ok,
            reason_code=first["code"] if first else None, reason=reason,
            details={
                "satisfied": satisfied, "violated": violated,
                **({"not_checked": not_checked} if not_checked else {}),
            },
        )  # fmt: skip

    def _same_step(self, ctx: EvalContext, pairs: list[tuple[str, str]]) -> MetricResult:
        a = ctx.assignment()
        steps: list[Step] = [s for u, v in pairs for s in (*a.bound(u), *a.bound(v))]
        if not steps or all(s.parallel_group is None for s in steps):
            return self._component(
                ctx, "same_step", status="skipped", reason_code="data_unavailable",
                reason="parallel_group not recorded",
            )  # fmt: skip
        bad = []
        for u, v in pairs:
            groups = {s.parallel_group for s in (*a.bound(u), *a.bound(v))}
            if len(groups) != 1 or None in groups:
                bad.append({"a": u, "b": v})
        return self._component(
            ctx, "same_step", status="ok", value=1.0 - len(bad) / len(pairs), passed=not bad,
            reason_code=None if not bad else "order.same_step",
            reason="" if not bad else f"not in the same step: {bad}", details={"violated": bad},
        )  # fmt: skip
