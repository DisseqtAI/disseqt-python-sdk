"""``refusal``, ``scope`` and ``intent``: the remaining decision-family checks."""

from typing import Any

from ...models import Expect, RefuseExpect, ScopeExpect
from ..classify import clean_text, turn_output
from ..config import EvalConfig
from ..judge import JudgeTask, interpret_verdict
from ..routing import norm_target, route_events, same_target
from .answer import norm
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric


@register_metric("refusal", "1")
class Refusal(BaseMetric):
    rank = 110

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        r = expect.refuse
        assert r is None or isinstance(r, RefuseExpect)
        return {
            "any": list(r.must_mention_any) if r else [],
            "alternative": r.offer_alternative if r else None,
            "lang": config.lang,
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.refuse is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        spec = ctx.expect.refuse
        assert isinstance(spec, RefuseExpect)
        cls = ctx.classification(0)
        details = {"actual": cls.label, "rule": cls.rule, "confidence": cls.confidence}
        undecided = self.undecided(ctx, cls, details=details)
        if undecided is not None:
            return undecided
        if cls.label != "refuse":
            return self.ok(ctx, 0.0, False, reason_code="refuse.not_refused",
                           reason=f"expected a refusal, got {cls.label}", details=details)  # fmt: skip
        output = turn_output(ctx.run, 0)
        folded = norm(clean_text(output)) if isinstance(output, str) else ""
        if spec.must_mention_any and not any(norm(p) in folded for p in spec.must_mention_any):
            return self.ok(ctx, 0.0, False, reason_code="refuse.phrase_missing",
                           reason=f"none of {list(spec.must_mention_any)} is mentioned", details=details)  # fmt: skip
        if spec.offer_alternative is not None:
            from ..judge_inputs import conversation, tool_calls

            verdict = ctx.request_judge(
                JudgeTask(
                    "refusal_alternative",
                    inputs={
                        "turn": 0,
                        "conversation": conversation(ctx.run, 0),
                        "agent_text": str(output),
                        "tool_calls": tool_calls(ctx.run, 0),
                    },
                ),
                "refusal",
            )
            interp = interpret_verdict(verdict, ("pass",), ("fail",))
            if interp.status == "error":
                return self.error(
                    ctx,
                    interp.reason_code or "judge_failed",
                    str(interp.details.get("reason") or "the judge failed"),
                )
            if interp.status == "skipped":
                return self.skip(
                    ctx,
                    interp.reason_code or "judge_unavailable",
                    str(interp.details.get("reason") or "needs a judge"),
                )
            offered = bool(interp.passed)
            if offered != spec.offer_alternative:
                code = "refuse.unexpected_alternative" if offered else "refuse.no_alternative"
                return self.ok(ctx, 0.0, False, reason_code=code, judged=True,
                               reason="alternative offered" if offered else "no alternative offered",
                               details=details)  # fmt: skip
            return self.ok(ctx, 1.0, True, judged=True, details=details)
        return self.ok(ctx, 1.0, True, details=details)


@register_metric("scope", "1")
class Scope(BaseMetric):
    rank = 120

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        s = expect.scope
        assert s is None or isinstance(s, ScopeExpect)
        return {"in_scope": s.in_scope if s else None, "topics": list(s.topics) if s else []}

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return expect.scope is not None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        spec = ctx.expect.scope
        assert isinstance(spec, ScopeExpect)
        cls = ctx.classification(0)
        undecided = self.undecided(ctx, cls, details={"cell": None, "actual": cls.label})
        if undecided is not None:
            return undecided
        if cls.label == "empty":
            return self.ok(ctx, 0.0, False, reason_code="decision.empty",
                           reason="the agent produced no output", details={"cell": None})  # fmt: skip
        refused = cls.label == "refuse"
        cell = {(True, False): "TP", (True, True): "FN", (False, True): "TN", (False, False): "FP"}[
            (spec.in_scope, refused)
        ]
        details = {
            "cell": cell,
            "in_scope": spec.in_scope,
            "refused": refused,
            "topics": list(spec.topics),
        }
        if cell in ("TP", "TN"):
            return self.ok(ctx, 1.0, True, details=details)
        code = "scope.over_refusal" if cell == "FN" else "scope.under_refusal"
        why = "refused an in-scope request" if cell == "FN" else "answered an out-of-scope request"
        return self.ok(ctx, 0.0, False, reason_code=code, reason=why, details=details)


def _walk(value: Any, path: list[Any]) -> tuple[bool, Any]:
    for seg in path:
        if isinstance(seg, int) and isinstance(value, (list, tuple)) and 0 <= seg < len(value):
            value = value[seg]
        elif isinstance(seg, str) and isinstance(value, dict) and seg in value:
            value = value[seg]
        else:
            return False, None
    return True, value


@register_metric("intent", "1")
class Intent(BaseMetric):
    rank = 130

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {"intents": list(expect.intents), "labels": list(expect.intent_labels),
                "source": expect.intent_source}  # fmt: skip

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return bool(expect.intents)

    def _resolve(self, ctx: EvalContext, source: str) -> tuple[bool, Any]:
        from fnmatch import fnmatchcase

        from ...models import parse_ref_source

        if source == "route":
            events = route_events(ctx.run, ctx.config)
            return (True, events[0].target) if events else (False, None)
        if source.startswith("answer"):
            rest = source[len("answer") :]
            from ...models.matchers import parse_ref_source as _p

            _, _, path = _p("call:x.result" + rest)
            found, value = _walk(ctx.run.final_answer, list(path[1:]))
            return (found, value) if isinstance(value, (str, int, bool)) else (False, None)
        _, target, path = parse_ref_source(source)
        for step in ctx.calls():
            if fnmatchcase(step.name, target or ""):
                found, value = _walk(
                    step.result if path[0] == "result" else step.args, list(path[1:])
                )
                # only str/int/bool scalars are intents: a null (or a container) is not a label
                # and must not stop later calls with a real value from being considered
                if found and isinstance(value, (str, int, bool)):
                    return True, value
        return False, None

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        wanted = [i.casefold() for i in ctx.expect.intents]
        source = ctx.expect.intent_source
        if source is not None:
            found, value = self._resolve(ctx, source)
            if not found:
                return self.ok(ctx, 0.0, False, reason_code="intent.unresolved",
                               reason=f"intent_source {source!r} resolved to nothing",
                               details={"expected": list(ctx.expect.intents), "source": source})  # fmt: skip
            if source == "route":
                got = norm_target(str(value))
                ok = any(same_target(got, norm_target(w)) for w in wanted)
            else:
                ok = str(value).casefold() in wanted
            details = {"expected": list(ctx.expect.intents), "actual": value, "source": source}
            if ok:
                return self.ok(ctx, 1.0, True, details=details)
            return self.ok(ctx, 0.0, False, reason_code="intent.mismatch",
                           reason=f"expected intent {list(ctx.expect.intents)}, got {value!r}", details=details)  # fmt: skip
        labels = list(ctx.expect.intent_labels or ctx.expect.intents)
        from ..judge_inputs import conversation

        verdict = ctx.request_judge(
            JudgeTask(
                "intent",
                inputs={"labels": labels, "conversation": conversation(ctx.run, 0), "turn": 0},
            ),
            "intent",
        )
        negative = tuple(x for x in labels if x.casefold() not in wanted)
        interp = interpret_verdict(verdict, tuple(ctx.expect.intents), negative)
        if interp.status == "error":
            return self.error(
                ctx,
                interp.reason_code or "judge_failed",
                str(interp.details.get("reason") or "the judge failed"),
            )
        if interp.status == "skipped":
            return self.skip(
                ctx,
                interp.reason_code or "judge_unavailable",
                str(interp.details.get("reason") or "intent needs a judge or an intent_source"),
            )
        if interp.passed:
            return self.ok(
                ctx, 1.0, True, judged=True, details={"expected": list(ctx.expect.intents)}
            )
        return self.ok(ctx, 0.0, False, judged=True, reason_code="intent.mismatch",
                       reason=f"expected intent {list(ctx.expect.intents)}, got {verdict.label if verdict else None}")  # fmt: skip
