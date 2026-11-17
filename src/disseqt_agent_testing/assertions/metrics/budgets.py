"""``budgets``: step, tool-call, model-call, token and latency limits (inclusive bounds)."""

from fnmatch import fnmatchcase
from typing import Any

from ...models import Expect
from ..config import EvalConfig
from .base import BaseMetric, EvalContext, MetricResult, first_reason, rollup
from .registry import register_metric

FIELDS = ("max_steps", "max_tool_calls", "max_llm_calls", "max_tokens", "max_latency_s")


@register_metric("budgets", "1")
class Budgets(BaseMetric):
    rank = 60

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in FIELDS:
            value = getattr(expect, name)
            if value is not None:
                out[name] = dict(value) if hasattr(value, "items") else value
        return out

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return any(getattr(expect, name) is not None for name in FIELDS)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        comps = []
        for name in FIELDS:
            if getattr(ctx.expect, name) is not None:
                comps.append(getattr(self, f"_{name}")(ctx))
        status, value, passed, code = rollup(comps)
        if status == "ok":
            bad = next((c for c in comps if c.passed is False), None)
            return self.ok(
                ctx, value, bool(passed), components=tuple(comps),
                reason_code=bad.reason_code if bad else None, reason=bad.reason if bad else "",
            )  # fmt: skip
        if status == "error":
            return self.error(
                ctx, code or "metric_error", first_reason(comps), components=tuple(comps)
            )
        return self.skip(
            ctx, code or "not_applicable", first_reason(comps), components=tuple(comps)
        )

    def _comp(self, ctx: EvalContext, field: str, **fields: Any) -> MetricResult:
        return MetricResult(
            f"budgets.{field}", version=self.version, params_hash=self._hash(ctx), **fields
        )

    def _bound(
        self, ctx: EvalContext, field: str, used: float, limit: float, note: str = ""
    ) -> MetricResult:
        ok = used <= limit
        return self._comp(
            ctx, field, status="ok", value=1.0 if ok else 0.0, passed=ok,
            reason_code=None if ok else "budget.exceeded",
            reason="" if ok else f"{field}: used {used:g}, limit {limit:g}{note}",
            details={"used": used, "limit": limit},
        )  # fmt: skip

    def _max_steps(self, ctx: EvalContext) -> MetricResult:
        note = "" if ctx.run.model_steps_captured else " (model calls not captured)"
        return self._bound(ctx, "max_steps", len(ctx.run.steps), ctx.expect.max_steps or 0, note)

    def _max_tool_calls(self, ctx: EvalContext) -> MetricResult:
        limit = ctx.expect.max_tool_calls
        calls = ctx.calls()
        if isinstance(limit, int):
            return self._bound(ctx, "max_tool_calls", len(calls), limit)
        assert limit is not None
        over = {}
        for glob in sorted(limit):
            n = sum(1 for s in calls if fnmatchcase(s.name, glob))
            if n > limit[glob]:
                over[glob] = {"used": n, "limit": limit[glob]}
        ok = not over
        return self._comp(
            ctx, "max_tool_calls", status="ok", value=1.0 if ok else 0.0, passed=ok,
            reason_code=None if ok else "budget.exceeded",
            reason="" if ok else f"max_tool_calls exceeded for {sorted(over)}",
            details={"over": over},
        )  # fmt: skip

    def _max_llm_calls(self, ctx: EvalContext) -> MetricResult:
        if not ctx.run.model_steps_captured:
            return self._comp(
                ctx, "max_llm_calls", status="skipped", reason_code="data_unavailable",
                reason="model calls not captured",
            )  # fmt: skip
        n = sum(1 for s in ctx.run.steps if s.kind == "model")
        return self._bound(ctx, "max_llm_calls", n, ctx.expect.max_llm_calls or 0)

    def _max_tokens(self, ctx: EvalContext) -> MetricResult:
        limit = ctx.expect.max_tokens or 0
        if not ctx.run.model_steps_captured:
            return self._comp(
                ctx, "max_tokens", status="skipped", reason_code="data_unavailable",
                reason="model calls not captured",
            )  # fmt: skip
        model = [s for s in ctx.run.steps if s.kind == "model"]
        known = [s for s in model if s.usage is not None]
        used = sum(s.usage.input_tokens + s.usage.output_tokens for s in known if s.usage)
        missing = len(model) - len(known)
        if not known:
            return self._comp(
                ctx, "max_tokens", status="skipped", reason_code="data_unavailable",
                reason=f"token usage missing for {missing} of {len(model)} LLM calls",
            )  # fmt: skip
        if missing and used <= limit:
            return self._comp(
                ctx, "max_tokens", status="skipped", reason_code="data_unavailable",
                reason=f"token usage missing for {missing} of {len(model)} LLM calls",
                details={"used_lower_bound": used, "limit": limit},
            )  # fmt: skip
        note = " (lower bound: usage missing for some calls)" if missing else ""
        return self._bound(ctx, "max_tokens", used, limit, note)

    def _max_latency_s(self, ctx: EvalContext) -> MetricResult:
        limit = ctx.expect.max_latency_s or 0.0
        if ctx.result is not None:
            seconds = max(ctx.result.duration_ms - ctx.result.setup_ms, 0.0) / 1000.0
            return self._bound(ctx, "max_latency_s", seconds, limit)
        steps = [s for s in ctx.run.steps if s.duration_ms is not None]
        if steps:
            start = min(s.ts for s in steps)
            end = max(s.ts + (s.duration_ms or 0.0) for s in steps)
            lower = (end - start) / 1000.0
            if lower > limit:
                return self._bound(ctx, "max_latency_s", lower, limit, " (lower bound)")
        return self._comp(
            ctx, "max_latency_s", status="skipped", reason_code="data_unavailable",
            reason="latency needs a RunResult",
        )  # fmt: skip
