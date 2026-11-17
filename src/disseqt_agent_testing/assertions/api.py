"""``evaluate`` / ``aevaluate`` / ``assert_agent``: the public entry points."""

import asyncio
import dataclasses
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

from ..models import AgentRun, Expect, FakeSpec, RunResult
from .assignment import check_ref_graph
from .canonical import short_hash
from .classify import classify_turn, turn_count
from .config import EvalConfig, EvalOptions, ExpectConfigError, resolve_config
from .judge import Judge
from .metrics import EvalContext, Memo, MetricResult
from .planner import (
    PlannedCheck,
    apply_threshold,
    flat_scores,
    gate,
    plan_checks,
)
from .report import DecisionReport, TriageResult

# Tool checks that cannot say anything useful when the decision already failed with no calls.
DEPENDENT_ON_DECISION = ("tool_selection", "tool_args", "order", "clarification")


class AgentAssertionError(AssertionError):
    """An expectation failed. ``.report`` is the full ``DecisionReport``."""

    def __init__(self, message: str, report: DecisionReport):
        super().__init__(message)
        self.report = report

    def __reduce__(self) -> tuple[Any, ...]:  # picklable (pytest-xdist ships exceptions)
        return (_rebuild_error, (type(self), str(self), self.report.to_dict()))


class AgentIncompleteError(AgentAssertionError):
    """Nothing failed, but something configured could not be measured (never a silent pass)."""


def _rebuild_error(cls: type, message: str, report: dict[str, Any]) -> AgentAssertionError:
    return cls(message, DecisionReport.from_dict(report))  # type: ignore[no-any-return]


def _split(result: RunResult | AgentRun) -> tuple[RunResult | None, AgentRun]:
    if isinstance(result, RunResult):
        return result, result.run
    if isinstance(result, AgentRun):
        return None, result
    raise TypeError(f"evaluate() needs a RunResult or AgentRun, got {type(result).__name__}")


def _blocked(name: str, code: str, reason: str) -> MetricResult:
    from .metrics import get_metric

    metric = get_metric(name)
    return MetricResult(
        name, version=metric.version, scope=metric.scope, status="skipped",
        reason_code=code, reason=reason,
    )  # fmt: skip


def _crash(ctx: EvalContext, check: PlannedCheck, exc: Exception) -> MetricResult:
    metric = check.metric
    return MetricResult(
        metric.name, version=metric.version, params_hash=check.params_hash, scope=metric.scope,
        status="error", reason_code="metric_error", reason=f"{type(exc).__name__}: {exc}",
    )  # fmt: skip


def _pre_skip(
    ctx: EvalContext, check: PlannedCheck, decision_failed_empty: bool
) -> MetricResult | None:
    metric = check.metric
    if not ctx.finished and not metric.run_on_unfinished:
        status = ctx.result.status if ctx.result else "?"
        return _blocked(metric.name, "blocked_by:run_status", f"the run did not finish ({status})")
    if decision_failed_empty and metric.name in DEPENDENT_ON_DECISION:
        return _blocked(
            metric.name, "blocked_by:decision_match", "the decision failed and no call was made"
        )
    if "judge" in metric.requires and (ctx.judge is None or not ctx.config.use_judge):
        return _blocked(metric.name, "judge_unavailable", "no judge is configured")
    if "model_steps" in metric.requires and not ctx.run.model_steps_captured:
        return _blocked(metric.name, "data_unavailable", "model calls not captured")
    if "run_result" in metric.requires and ctx.result is None:
        return _blocked(metric.name, "data_unavailable", "no RunResult")
    return None


def _decision_blocks(results: list[MetricResult], ctx: EvalContext) -> bool:
    for r in results:
        if r.name == "decision_match" and r.status == "ok" and r.passed is False:
            return not ctx.calls()
    return False


def _prepare(
    result: RunResult | AgentRun,
    expect: Expect,
    case_id: str | None,
    judge: Judge | None,
    config: EvalConfig | None,
    fakes: tuple[FakeSpec, ...],
    options: Mapping[str, Any],
) -> tuple[EvalContext, list[PlannedCheck]]:
    cfg = resolve_config(config, options)
    run_result, run = _split(result)
    if not isinstance(expect, Expect):
        raise TypeError(f"expect must be an Expect, got {type(expect).__name__}")
    check_ref_graph(expect.tools)
    ctx = EvalContext(
        result=run_result,
        run=run,
        expect=expect,
        case_id=case_id or run.case_id or "case",
        config=cfg,
        judge=judge,
        fakes=tuple(fakes),
        memo=Memo(),
    )
    return ctx, plan_checks(expect, cfg)


def _stamp(ctx: EvalContext, check: PlannedCheck, result: MetricResult) -> MetricResult:
    """A judged result compares only like with like: its version names the judge prompts and its
    params hash includes the judge identities (a result from a judge without identity data, such
    as a ``StaticJudge``, is left alone)."""
    recs = [
        r for r in ctx.judge_log if r.get("metric") == check.metric.name and r.get("identity_hash")
    ]
    if not recs:
        return result
    kinds = sorted({(str(r["kind"]), str(r.get("prompt_version", ""))) for r in recs})
    version = f"{result.version}+" + ",".join(f"{k}@{v}" for k, v in kinds)
    ids = sorted({str(r["identity_hash"]) for r in recs})
    details = dict(result.details)
    if result.status == "skipped" and result.reason_code == "judge_uncalibrated":
        last = recs[-1]
        details.setdefault("judge_kind", last["kind"])
        details.setdefault("advisory_label", last["label"])
    return dataclasses.replace(
        result,
        version=version,
        params_hash=short_hash({"params": result.params_hash, "judges": ids}),
        details=details,
    )


def _run_one(ctx: EvalContext, check: PlannedCheck) -> MetricResult:
    try:
        return _stamp(ctx, check, apply_threshold(check.metric.evaluate(ctx), check))
    except ExpectConfigError:
        raise
    except Exception as exc:  # a metric bug must not hide the other results
        return _crash(ctx, check, exc)


def evaluate(
    result: RunResult | AgentRun,
    expect: Expect,
    *,
    case_id: str | None = None,
    judge: Judge | None = None,
    config: EvalConfig | None = None,
    fakes: tuple[FakeSpec, ...] = (),
    **options: Any,
) -> DecisionReport:
    """Score a run against an expectation. Never raises for a failed expectation; raises only
    ``ExpectConfigError`` (a bug in the test) or a ``BaseException``."""
    ctx, planned = _prepare(result, expect, case_id, judge, config, fakes, options)
    results: list[MetricResult] = []
    with _eval_scope(judge):
        for check in planned:
            skipped = _pre_skip(ctx, check, _decision_blocks(results, ctx))
            results.append(skipped if skipped is not None else _run_one(ctx, check))
    return _compose(ctx, results)


async def aevaluate(
    result: RunResult | AgentRun,
    expect: Expect,
    *,
    case_id: str | None = None,
    judge: Judge | None = None,
    config: EvalConfig | None = None,
    fakes: tuple[FakeSpec, ...] = (),
    **options: Any,
) -> DecisionReport:
    """Like ``evaluate``; independent metrics run concurrently (``asyncio.gather``)."""
    ctx, planned = _prepare(result, expect, case_id, judge, config, fakes, options)
    results: list[MetricResult] = []
    head = [c for c in planned if c.metric.name == "decision_match"]
    rest = [c for c in planned if c.metric.name != "decision_match"]

    async def one(check: PlannedCheck, blocked: bool) -> MetricResult:
        skipped = _pre_skip(ctx, check, blocked)
        if skipped is not None:
            return skipped
        try:
            return _stamp(ctx, check, apply_threshold(await check.metric.aevaluate(ctx), check))
        except ExpectConfigError:
            raise
        except Exception as exc:
            return _crash(ctx, check, exc)

    with _eval_scope(judge):
        for check in head:
            results.append(await one(check, False))
        blocked = _decision_blocks(results, ctx)
        results.extend(await asyncio.gather(*(one(c, blocked) for c in rest)))
    order = {id(c): i for i, c in enumerate(planned)}
    by_name = {c.metric.name: order[id(c)] for c in planned}
    results.sort(key=lambda r: by_name.get(r.name, 0))
    return _compose(ctx, results)


def _eval_scope(judge: Any) -> Any:
    """A judge that offers ``eval_scope()`` bounds its calls per evaluation."""
    scope = getattr(judge, "eval_scope", None)
    return scope() if callable(scope) else nullcontext()


PostCompose = Callable[[EvalContext, DecisionReport], DecisionReport]
_POST_COMPOSE: list[PostCompose] = []


def register_post_compose(hook: PostCompose, *, override: bool = False) -> None:
    """Hooks run in registration order on the finished report (for example advisory triage)."""
    if hook in _POST_COMPOSE and not override:
        raise ValueError("this post-compose hook is already registered")
    if hook not in _POST_COMPOSE:
        _POST_COMPOSE.append(hook)


def unregister_post_compose(hook: PostCompose) -> None:
    if hook in _POST_COMPOSE:
        _POST_COMPOSE.remove(hook)


def _advisory_warnings(results: list[MetricResult]) -> list[str]:
    """An uncalibrated judge never decides a label; say so when it disagrees with the rule."""
    out = []
    for r in results:
        for comp in r.components if r.name == "decision_match" else ():
            advisory = comp.details.get("advisory_label")
            actual = comp.details.get("actual")
            if advisory and advisory != "None" and advisory != actual:
                out.append(
                    f"decision turn {comp.turn}: uncalibrated judge says {advisory!r}, "
                    f"heuristic says {actual!r} (heuristic kept)"
                )
    return out


def _compose(ctx: EvalContext, results: list[MetricResult]) -> DecisionReport:
    report = _compose_base(ctx, results)
    for hook in list(_POST_COMPOSE):
        report = hook(ctx, report)
    return report


def _compose_base(ctx: EvalContext, results: list[MetricResult]) -> DecisionReport:
    run_result = ctx.result
    status = run_result.status if run_result else "finished"
    violations = list(run_result.violations) if run_result else []
    harness_failed = status != "finished" or bool(violations)
    verdict = gate(results, harness_failed, ctx.config)
    expected_turns = set(ctx.expect.turn_decisions)
    decisions = []
    for turn in range(turn_count(ctx.run)):
        cls = (
            ctx.classification(turn)
            if turn in expected_turns
            else classify_turn(ctx.run, turn, ctx.config, None)
        )
        decisions.append(
            {
                "turn": turn,
                "label": cls.label,
                "confidence": cls.confidence,
                "rule": cls.rule,
                "evidence": list(cls.evidence),
                "judged": cls.judged,
            }
        )
    harness = {
        "status": status,
        "error": run_result.error.message if run_result and run_result.error else None,
        "violations": [
            {"code": v.code, "message": v.message, "fake": v.fake, "call_id": v.call_id}
            for v in violations
        ],
        "warnings": list(run_result.warnings) if run_result else [],
        "trace_id": run_result.trace_id if run_result else "",
    }
    summary = {
        "n_checks": len(results),
        "n_passed": verdict.n_passed,
        "n_failed": verdict.n_failed + (1 if harness_failed else 0),
        "n_skipped": verdict.n_skipped,
        "n_error": verdict.n_error,
        "n_incomplete": verdict.n_incomplete,
    }
    return DecisionReport(
        ctx.case_id,
        verdict.outcome,
        harness=harness,
        decisions=tuple(decisions),
        summary=summary,
        scores=flat_scores(results),
        checks=tuple(results),
        judge_calls=tuple(ctx.judge_log),
        warnings=(*verdict.warnings, *_advisory_warnings(results)),
    )


check = evaluate  # the Step 5 runner's name for it


def assert_agent(
    result: RunResult | AgentRun,
    expect: Expect | None = None,
    *,
    case_id: str | None = None,
    judge: Judge | None = None,
    config: EvalConfig | None = None,
    fakes: tuple[FakeSpec, ...] = (),
    max_lines: int | None = None,
    **options: Any,
) -> "DecisionReport | Any":
    """Assert an expectation. With ``expect`` it returns the report or raises
    ``AgentAssertionError`` (``AgentIncompleteError`` when nothing failed but something could not
    be measured). Without ``expect`` it returns the fluent builder."""
    __tracebackhide__ = True
    if expect is None:
        from .fluent import AgentAssertions

        return AgentAssertions(
            result, case_id=case_id, judge=judge, config=config, fakes=fakes, options=options
        )
    report = evaluate(
        result, expect, case_id=case_id, judge=judge, config=config, fakes=fakes, **options
    )
    raise_for_report(report, max_lines, config)
    return report


def raise_for_report(
    report: DecisionReport, max_lines: int | None = None, config: EvalConfig | None = None
) -> None:
    __tracebackhide__ = True
    if report.outcome == "passed":
        return
    limit = max_lines if max_lines is not None else (config.max_report_lines if config else None)
    message = report.render(limit)
    if report.outcome == "incomplete":
        raise AgentIncompleteError(message, report)
    raise AgentAssertionError(message, report)


_ = TriageResult
__all__ = [
    "register_post_compose",
    "unregister_post_compose",
    "AgentAssertionError",
    "AgentIncompleteError",
    "EvalOptions",
    "aevaluate",
    "assert_agent",
    "check",
    "dataclasses",
    "evaluate",
    "raise_for_report",
]
