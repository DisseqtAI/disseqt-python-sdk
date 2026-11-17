"""``loops``: repeated identical calls, runs of them, and periodic cycles. Opt-in only (polling
tools legitimately repeat; ``key="name"`` flags pagination by design)."""

from typing import Any

from ...models import Expect, LoopExpect, Step
from ..canonical import args_hash
from ..config import EvalConfig
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


def _key(step: Step, mode: str) -> tuple[str, str]:
    return (step.name, args_hash(step.args) if mode == "name_args" else "")


def find_cycle(
    keys: list[tuple[str, str]], max_period: int, min_repeats: int
) -> tuple[int, int] | None:
    """The first ``(start, period)`` where a block of ``period`` (2..max_period) distinct-looking
    calls repeats ``min_repeats`` times in a row. Period 1 is the consecutive rule's job."""
    n = len(keys)
    for period in range(2, max_period + 1):
        for start in range(0, n - period * min_repeats + 1):
            block = keys[start : start + period]
            if len(set(block)) < 2:
                continue
            if all(
                keys[start + r * period : start + (r + 1) * period] == block
                for r in range(min_repeats)
            ):
                return start, period
    return None


@register_metric("loops", "1")
class Loops(BaseMetric):
    rank = 90
    run_on_unfinished = True

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        lp = expect.loops
        assert lp is None or isinstance(lp, LoopExpect)
        return {} if lp is None else {
            "max_identical_calls": lp.max_identical_calls,
            "max_consecutive_identical": lp.max_consecutive_identical,
            "detect_cycles": lp.detect_cycles, "max_period": lp.max_period,
            "min_repeats": lp.min_repeats, "key": lp.key,
            "max_retries_after_error": lp.max_retries_after_error,
        }  # fmt: skip

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.loops is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        lp = ctx.expect.loops
        assert isinstance(lp, LoopExpect)
        calls = list(ctx.calls())
        # An identical call right after an errored one is a retry, not a loop (up to the cap).
        kept: list[Step] = []
        retries = 0
        for i, step in enumerate(calls):
            prev = calls[i - 1] if i else None
            if (
                prev is not None and prev.error is not None
                and _key(prev, lp.key) == _key(step, lp.key) and retries < lp.max_retries_after_error
            ):  # fmt: skip
                retries += 1
                continue
            if prev is not None and _key(prev, lp.key) != _key(step, lp.key):
                retries = 0
            kept.append(step)
        keys = [_key(s, lp.key) for s in kept]
        findings: list[dict[str, Any]] = []
        if lp.max_identical_calls is not None:
            counts: dict[tuple[str, str], int] = {}
            for k in keys:
                counts[k] = counts.get(k, 0) + 1
            for k in sorted(counts):
                if counts[k] > lp.max_identical_calls:
                    findings.append({"code": "loop.identical", "name": k[0], "count": counts[k],
                                     "limit": lp.max_identical_calls})  # fmt: skip
        if lp.max_consecutive_identical is not None:
            run_len, best = 0, (0, "")
            for i, k in enumerate(keys):
                run_len = run_len + 1 if i and keys[i - 1] == k else 1
                if run_len > best[0]:
                    best = (run_len, k[0])
            if best[0] > lp.max_consecutive_identical:
                findings.append({"code": "loop.consecutive", "name": best[1], "count": best[0],
                                 "limit": lp.max_consecutive_identical})  # fmt: skip
        if lp.detect_cycles:
            cycle = find_cycle(keys, lp.max_period, lp.min_repeats)
            if cycle is not None:
                start, period = cycle
                findings.append({"code": "loop.cycle", "period": period,
                                 "calls": [k[0] for k in keys[start : start + period]],
                                 "repeats": lp.min_repeats})  # fmt: skip
        details = {"calls": len(calls), "considered": len(kept), "findings": findings}
        if not findings:
            return self.ok(ctx, 1.0, True, details=details)
        first = findings[0]
        if first["code"] == "loop.cycle":
            why = f"cycle of {first['calls']} repeated {first['repeats']}x"
        else:
            why = f"{first['name']} x{first['count']} (limit {first['limit']})"
        return self.ok(ctx, 0.0, False, reason_code=first["code"], reason=why, details=details)
