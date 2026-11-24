"""``clarification``: when the agent should ask, did it ask well (and before acting)?"""

from typing import Any

from ...models import ClarifyExpect, Decision, Expect
from ..classify import clean_text, count_questions, turn_calls, turn_count, turn_output
from ..config import EvalConfig
from ..judge import JudgeTask, interpret_verdict
from ..lexicons import get_lexicon
from .answer import norm
from .base import BaseMetric, EvalContext, MetricResult, first_reason, rollup
from .registry import register_metric


def ask_turns(expect: Expect) -> list[int]:
    turns = {t for t, labels in expect.turn_decisions.items() if labels == (Decision.ASK,)}
    if expect.clarify is not None:
        turns.add(0)
    return sorted(turns)


@register_metric("clarification", "1")
class Clarification(BaseMetric):
    rank = 100

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        c = expect.clarify
        assert c is None or isinstance(c, ClarifyExpect)
        return {
            "turns": ask_turns(expect),
            "any": list(c.must_mention_any) if c else [],
            "all": list(c.must_mention_all) if c else [],
            "max_questions": c.max_questions if c else None,
            "reads_ok": c.allow_read_calls_before if c else True,
            "relevance": c.judge_relevance if c else False,
            "lang": config.lang,
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return bool(ask_turns(expect))

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        comps = [self._turn(ctx, t) for t in ask_turns(ctx.expect)]
        status, value, passed, code = rollup(comps)
        if status == "ok":
            bad = next((c for c in comps if c.passed is False), None)
            return self.ok(
                ctx, value, bool(passed), components=tuple(comps),
                reason_code=bad.reason_code if bad else None, reason=bad.reason if bad else "",
                judged=any(c.judged for c in comps),
            )  # fmt: skip
        if status == "error":
            return self.error(
                ctx, code or "judge_failed", first_reason(comps), components=tuple(comps)
            )
        return self.skip(
            ctx, code or "not_applicable", first_reason(comps), components=tuple(comps)
        )

    def _fail(
        self, ctx: EvalContext, turn: int, code: str, reason: str, **details: Any
    ) -> MetricResult:
        return self.ok(
            ctx,
            0.0,
            False,
            turn=turn,
            scope="turn",
            reason_code=code,
            reason=reason,
            details=details,
        )

    def _turn(self, ctx: EvalContext, turn: int) -> MetricResult:
        # ``clarify`` describes the question asked on turn 0; later ask turns are only checked
        # for asking (and not acting first)
        spec = ctx.expect.clarify if turn == 0 else None
        assert spec is None or isinstance(spec, ClarifyExpect)
        if turn >= turn_count(ctx.run):
            return self._fail(ctx, turn, "decision.turn_missing", f"the run has no turn {turn}")
        cls = ctx.classification(turn)
        undecided = self.undecided(ctx, cls, turn=turn, scope="turn", details={"actual": cls.label})
        if undecided is not None:
            return undecided
        if cls.label != "ask":
            return self._fail(
                ctx,
                turn,
                "clarify.not_asked",
                f"expected a clarifying question, got {cls.label}",
                actual=cls.label,
            )
        calls = turn_calls(ctx.run, turn)
        reads_ok = spec.allow_read_calls_before if spec else True
        acted = (
            [s for s in calls if s.effect == "write" or s.kind == "subagent"] if reads_ok else calls
        )
        if acted:
            return self._fail(
                ctx, turn, "clarify.acted_before_asking",
                f"called {acted[0].name} before asking", calls=[s.name for s in acted],
            )  # fmt: skip
        output = turn_output(ctx.run, turn)
        text = clean_text(output) if isinstance(output, str) else ""
        pack = get_lexicon(ctx.config.lang)
        if spec is not None:
            folded = norm(text)
            wanted_any = list(spec.must_mention_any)
            wanted_all = list(spec.must_mention_all)
            if wanted_any and not any(norm(p) in folded for p in wanted_any):
                return self._fail(
                    ctx,
                    turn,
                    "clarify.slot_missing",
                    f"none of {wanted_any} is mentioned",
                    missing=wanted_any,
                )
            missing = [p for p in wanted_all if norm(p) not in folded]
            if missing:
                return self._fail(
                    ctx, turn, "clarify.slot_missing", f"missing {missing}", missing=missing
                )
            if spec.max_questions is not None:
                n = count_questions(text, pack, ctx.config)
                if n > spec.max_questions:
                    return self._fail(
                        ctx,
                        turn,
                        "clarify.too_many_questions",
                        f"{n} questions (limit {spec.max_questions})",
                        questions=n,
                    )
            if spec.judge_relevance:
                from ..judge_inputs import conversation, tool_schemas

                verdict = ctx.request_judge(
                    JudgeTask(
                        "clarification_relevance",
                        inputs={
                            "turn": turn,
                            "conversation": conversation(ctx.run, turn),
                            "agent_text": text,
                            "missing_hints": list(spec.must_mention_any),
                            "tool_schemas": tool_schemas(ctx.run),
                        },
                    ),
                    "clarification",
                )
                interp = interpret_verdict(verdict, ("pass",), ("fail",))
                if interp.status == "error":
                    return self.error(
                        ctx,
                        interp.reason_code or "judge_failed",
                        str(interp.details.get("reason") or "the judge failed"),
                        turn=turn,
                        scope="turn",
                    )
                if interp.status == "skipped":
                    return self.skip(
                        ctx,
                        interp.reason_code or "judge_unavailable",
                        str(interp.details.get("reason") or "relevance needs a judge"),
                        turn=turn,
                        scope="turn",
                    )
                if not interp.passed:
                    return self.ok(ctx, 0.0, False, turn=turn, scope="turn", judged=True,
                                   reason_code="clarify.irrelevant_question",
                                   reason="the question does not target missing information")  # fmt: skip
                return self.ok(
                    ctx, 1.0, True, turn=turn, scope="turn", judged=True, details={"label": "ask"}
                )
        return self.ok(ctx, 1.0, True, turn=turn, scope="turn", details={"label": "ask"})
