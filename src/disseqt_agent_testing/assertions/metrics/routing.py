"""``routing``: did the agent hand off to the expected destinations?"""

from typing import Any

from ...models import Expect, RouteExpect
from ..config import EvalConfig
from ..routing import norm_target, offered_targets, route_events, same_target
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


@register_metric("routing", "1")
class Routing(BaseMetric):
    rank = 80

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        routes = expect.routes
        assert routes is None or isinstance(routes, RouteExpect)
        return {
            "path": list(routes.path) if routes else [],
            "mode": routes.mode if routes else None,
            "patterns": list(config.route_patterns),
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.routes is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        routes = ctx.expect.routes
        assert isinstance(routes, RouteExpect)
        offered = offered_targets(ctx.run, ctx.config)
        events = route_events(ctx.run, ctx.config)
        if not offered and not events:
            return self.skip(ctx, "data_unavailable", "no routable target was offered to the agent")
        observed = [e.target for e in events]
        want = [norm_target(p) for p in routes.path]
        details = {"mode": routes.mode, "expected": list(routes.path), "observed": observed,
                   "offered": sorted(offered)}  # fmt: skip
        invalid = [e for e in events if offered and e.target not in offered]
        if invalid:
            return self.ok(
                ctx, 0.0, False, vetoed=True, reason_code="route.invalid_target",
                reason=f"routed to {invalid[0].raw!r}, which was not offered (offered: {sorted(offered)})",
                details=details,
            )  # fmt: skip
        mode = routes.mode
        if mode == "exact":
            same = len(observed) == len(want) and all(map(same_target, observed, want))
            ok, code = same, "route.mismatch"
        elif mode == "in_order":
            it = iter(observed)
            ok, code = all(any(same_target(t, o) for o in it) for t in want), "route.mismatch"
        elif mode == "first":
            ok = bool(observed) and bool(want) and same_target(observed[0], want[0])
            code = "route.mismatch"
        elif mode == "any":
            ok = any(same_target(t, o) for t in want for o in observed)
            code = "route.mismatch"
        else:  # none: none of the listed targets (or no route at all) was taken
            taken = [t for t in observed if not want or any(same_target(t, w) for w in want)]
            ok, code = not taken, "route.unexpected"
        if ok:
            return self.ok(ctx, 1.0, True, details=details)
        return self.ok(
            ctx, 0.0, False, reason_code=code,
            reason=f"expected route {mode} {list(routes.path)}, observed {observed}", details=details,
        )  # fmt: skip
