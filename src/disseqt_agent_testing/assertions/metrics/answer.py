"""``answer``: required and forbidden text in the final answer."""

import re
import unicodedata
from typing import Any

from ...models import Expect
from ..canonical import clip
from ..config import EvalConfig
from ..matching import MatchContext, MatchResult, describe, match_value
from .base import BaseMetric, EvalContext, MetricResult
from .registry import register_metric

_SPACES = re.compile(r"\s+")


def norm(text: str) -> str:
    """NFKC, casefold, ``_``/``-`` as spaces, collapsed whitespace."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("_", " ").replace("-", " ")
    return _SPACES.sub(" ", text).strip()


def _check(item: Any, text: str, ctx: MatchContext) -> MatchResult:
    """Tri-state: an ``ok`` result is a decided hit/miss; a skipped or errored matcher stays
    undecided, so an undecidable ``answer_excludes`` entry can never read as "not present"."""
    if isinstance(item, str):
        hit = norm(item) in norm(text)
        return MatchResult("ok", hit, 1.0 if hit else 0.0, None, (), item)
    leaves = match_value(item, text, ctx, ())
    for status in ("error", "skipped"):
        found = next((r for r in leaves if r.status == status), None)
        if found is not None:
            return found
    hit = all(r.ok for r in leaves)
    return MatchResult("ok", hit, 1.0 if hit else 0.0, None, (), describe(item))


@register_metric("answer", "1")
class Answer(BaseMetric):
    rank = 70

    def params(self, expect: Expect, config: EvalConfig) -> dict[str, Any]:
        return {
            "contains": [i if isinstance(i, str) else describe(i) for i in expect.answer_contains],
            "excludes": [i if isinstance(i, str) else describe(i) for i in expect.answer_excludes],
        }

    def applicable(self, expect: Expect, config: EvalConfig) -> bool:
        return bool(expect.answer_contains or expect.answer_excludes)

    def evaluate(self, ctx: EvalContext) -> MetricResult:
        text = ctx.run.final_text
        if text is None:
            return self.ok(
                ctx, 0.0, False, reason_code="answer.no_output",
                reason="the agent produced no answer", details={"missing": [], "forbidden": []},
            )  # fmt: skip
        mctx = MatchContext(
            run=ctx.run,
            predicates=ctx.config.predicates,
            judge=ctx.judge,
            use_judge=ctx.config.use_judge,
            judge_log=ctx.judge_log,
        )
        wanted = list(ctx.expect.answer_contains)
        banned = list(ctx.expect.answer_excludes)
        checks = [(i, _check(i, text, mctx)) for i in [*wanted, *banned]]
        wanted_checks, banned_checks = checks[: len(wanted)], checks[len(wanted) :]
        missing = [i for i, r in wanted_checks if r.status == "ok" and not r.ok]
        present = [i for i, r in banned_checks if r.status == "ok" and r.ok]
        undecided = [(i, r) for i, r in checks if r.status != "ok"]
        total = len(wanted) + len(banned)
        satisfied = (len(wanted) - len(missing)) + (len(banned) - len(present))
        details = {
            "missing": [i if isinstance(i, str) else describe(i) for i in missing],
            "forbidden": [i if isinstance(i, str) else describe(i) for i in present],
            "answer": clip(text),
        }
        if present:
            return self.ok(
                ctx, satisfied / total, False, vetoed=False, reason_code="answer.forbidden_text",
                reason=f"the answer contains forbidden text: {details['forbidden']}", details=details,
            )  # fmt: skip
        if missing:
            return self.ok(
                ctx, satisfied / total, False, reason_code="answer.missing_text",
                reason=f"the answer is missing: {details['missing']}", details=details,
            )  # fmt: skip
        if undecided:  # a real failure above dominates; otherwise absent evidence is no pass
            item, res = undecided[0]
            why = f"{describe(item) if not isinstance(item, str) else item}: {res.reason}"
            if any(r.status == "error" for _, r in undecided):
                err = next(r for _, r in undecided if r.status == "error")
                return self.error(ctx, err.code or "predicate_error", why, details=details)
            return self.skip(ctx, res.code or "judge_unavailable", why, details=details)
        return self.ok(ctx, 1.0, True, details=details)
