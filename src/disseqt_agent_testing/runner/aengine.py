"""``arun_case`` / ``arun_suite``: the same engine on the caller's event loop.

Units are tasks gated by an ``asyncio.Semaphore(max_workers)``; they call ``Orchestrator.arun``
and ``aevaluate``. Nothing here calls ``asyncio.run``. A unit stopped by Disseqt's own deadline is
``not_run``; cancellation from outside is re-raised after the in-flight tasks are cancelled and
awaited.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from ..assertions import ExpectConfigError, aevaluate
from ..assertions.judge import Judge
from ..harness import HarnessConfigError, Orchestrator
from .config import SuiteConfig
from .engine import (
    EventCallback,
    Plan,
    SuiteState,
    UnitOutcome,
    _abandon_all,
    _finish_if_done,
    _with_collection,
    _zombies,
    build_result,
    classify_failure,
    make_judge,
    not_run,
    outcome_of,
    prepare,
    to_specs,
)
from .events import CaseFinished, CaseStarted, RepeatFinished, SuiteFinished
from .limiter import Limiter
from .models import CaseResult, CaseSpec, SuiteResult
from .seeds import _CURRENT_REPEAT, RepeatContext


async def _unit_body(plan: Plan, r: int) -> UnitOutcome:
    orch = plan.orchestrator_for(r)
    result = await orch.arun(plan.case.input, follow_ups=plan.case.follow_ups)
    report = await aevaluate(
        result, plan.case.expect, case_id=plan.case.id, judge=plan.judge, config=plan.eval_config,
        fakes=orch.fakes,
    )  # fmt: skip
    return outcome_of(r, result, report, plan.config)


async def arun_unit(plan: Plan, r: int, suite: SuiteState) -> UnitOutcome:
    reason = plan.skip_reason(suite)
    if reason is not None:
        return not_run(r, reason)
    plan.mark_started()
    token = _CURRENT_REPEAT.set(RepeatContext(plan.case.id, r, plan.seeds[r], plan.n))
    try:
        limit = plan.config.case_timeout_s
        if limit is None:
            return await _unit_body(plan, r)
        remaining = max(0.001, (plan.started_at or time.monotonic()) + limit - time.monotonic())
        try:
            # wait_for works on 3.10 too; its own deadline is TimeoutError, an outside cancel is
            # CancelledError, so the two are never confused.
            return await asyncio.wait_for(_unit_body(plan, r), timeout=remaining)
        except asyncio.TimeoutError:
            plan.cancel_reason = plan.cancel_reason or "case_timeout"
            return not_run(r, "case_timeout")
    except (HarnessConfigError, ExpectConfigError) as exc:
        classify_failure(plan, exc)
        return not_run(r, "cancelled")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        classify_failure(plan, exc)
        return not_run(r, "cancelled")
    except BaseException as exc:
        suite.abort(exc)
        raise
    finally:
        _CURRENT_REPEAT.reset(token)


async def arun_suite(
    cases: Any, *, agent_factory: Callable[..., Any] | None = None, config: SuiteConfig = SuiteConfig(),
    judge: Judge | None = None, limiter: Limiter | None = None, filter: Any = None,
    on_event: EventCallback | None = None, orchestrator: Orchestrator | None = None,
    suite_deadline: float | None = None,
) -> SuiteResult:  # fmt: skip
    specs, errors = to_specs(cases, filter)
    t0 = time.monotonic()
    suite = SuiteState(config, on_event)
    if suite_deadline is not None:
        suite.deadline = suite_deadline
    wrapped = make_judge(judge, config, limiter)
    results: list[CaseResult] = []
    plans: list[Plan] = []
    for i, spec in enumerate(specs):
        prepared = prepare(spec, i, config, suite, wrapped, agent_factory, orchestrator)
        if isinstance(prepared, CaseResult):
            results.append(prepared)
            suite.emit(CaseFinished(spec.case.id, prepared.status))
        else:
            plans.append(prepared)
    sem = asyncio.Semaphore(config.max_workers)

    async def gated(plan: Plan, r: int) -> UnitOutcome:
        async with sem:
            return await arun_unit(plan, r, suite)

    tasks: dict[asyncio.Task[UnitOutcome], tuple[Plan, int]] = {}
    try:
        for plan in plans:
            suite.emit(CaseStarted(plan.case.id, plan.n))
            plan.pending.update(range(plan.n))
        for plan in plans:
            for r in range(plan.n):
                if config.only_repeat is not None and r != config.only_repeat:
                    plan.record(not_run(r, "only_repeat"))
                    continue
                tasks[asyncio.ensure_future(gated(plan, r))] = (plan, r)
            _finish_if_done(plan, suite, results)
        while tasks:
            done, _ = await asyncio.wait(
                list(tasks), timeout=0.05, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                plan, r = tasks.pop(task)
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None and not isinstance(exc, Exception):
                    raise exc
                out = task.result() if exc is None else not_run(r, "cancelled")
                if plan.record(out):
                    suite.emit(RepeatFinished(plan.case.id, r, out.status))
                _finish_if_done(plan, suite, results)
    except BaseException:
        suite.cancelled = True
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)  # cancelled in their own loop
        raise
    _abandon_all(plans, suite, results)
    result = build_result(
        results, tuple(_with_collection(errors)), config, suite, t0, _zombies(plans)
    )
    suite.emit(SuiteFinished(len(result.cases), result.exit_code))
    return result


async def arun_case(
    case: Any, *, agent_factory: Callable[..., Any] | None = None, orchestrator: Orchestrator | None = None,
    config: SuiteConfig = SuiteConfig(), judge: Judge | None = None, limiter: Limiter | None = None,
    on_event: EventCallback | None = None, suite_deadline: float | None = None,
) -> CaseResult:  # fmt: skip
    spec = case if isinstance(case, CaseSpec) else CaseSpec(case)
    result = await arun_suite(
        [spec], agent_factory=agent_factory, orchestrator=orchestrator, config=config, judge=judge,
        limiter=limiter, on_event=on_event, suite_deadline=suite_deadline,
    )  # fmt: skip
    return result.cases[0]
