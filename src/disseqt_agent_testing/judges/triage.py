"""Opt-in advisory root-cause triage: ``enable_triage()`` registers a post-compose hook that asks
the ``root_cause`` judge why a failed case failed and writes ``DecisionReport.triage``.

The hint never changes the outcome, never gates and never raises: a missing or failing judge just
leaves ``triage`` empty.
"""

import dataclasses

from ..assertions.api import register_post_compose, unregister_post_compose
from ..assertions.judge import JudgeTask, call_record
from ..assertions.judge_inputs import conversation, tool_calls
from ..assertions.metrics.base import EvalContext
from ..assertions.report import DecisionReport, TriageResult


def triage_hook(ctx: EvalContext, report: DecisionReport) -> DecisionReport:
    if (
        report.outcome != "failed"
        or report.triage
        or ctx.judge is None
        or not ctx.config.use_judge
        or not ctx.config.triage
    ):  # EvalConfig.triage is the switch: a registered hook alone never spends a judge call
        return report
    failed = [c for c in report.checks if c.status == "ok" and c.passed is False]
    if not failed:
        return report
    last = max(len(ctx.run.turn_outputs) - 1, 0)
    task = JudgeTask("root_cause", inputs={
        "failed_checks": [f"{c.name}: {c.reason_code or ''} {c.reason}".strip() for c in failed],
        "conversation": conversation(ctx.run, last, include_agent_of_turn=True),
        "tool_calls": tool_calls(ctx.run, last),
        "agent_text": ctx.run.final_text or "",
    })  # fmt: skip
    try:
        verdict = ctx.judge.judge(task)
    except Exception:  # advisory: an instrument failure must not hide the real failure
        return report
    record = call_record(verdict, "triage", "root_cause")
    calls = (*report.judge_calls, record)
    if verdict.status != "ok" or verdict.label in (None, "unsure"):
        return dataclasses.replace(report, judge_calls=calls)
    hint = TriageResult(
        str(verdict.label),
        evidence=tuple(c.name for c in failed),
        reasoning=verdict.reason,
        agreement=verdict.agreement,
        identity_hash=verdict.identity_hash,
        judged=True,
    )
    return dataclasses.replace(report, triage=(hint,), judge_calls=calls)


def enable_triage() -> None:
    register_post_compose(triage_hook, override=True)


def disable_triage() -> None:
    unregister_post_compose(triage_hook)
