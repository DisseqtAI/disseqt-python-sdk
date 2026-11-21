"""``run_case`` / ``run_suite``: schedule ``(case, repeat)`` units on a bounded pool of daemon
threads, score each repeat with ``evaluate`` and decide each case with the bounds rule.

Shared logic (planning, recording, early stopping, result building) lives here and is reused by
the asyncio twins in ``aengine.py``.
"""

import concurrent.futures
import dataclasses
import importlib.util
import platform
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..assertions import ExpectConfigError, evaluate
from ..assertions.judge import Judge
from ..assertions.report import DecisionReport, RepeatReport
from ..harness import FakeLLM, HarnessConfigError, Orchestrator, build_test_orchestrator
from ..models import AgentTestCase, ErrorInfo, Message, from_dict
from .config import SuiteConfig, SuiteConfigError
from .errors import DisseqtWarning, format_code
from .events import CaseFinished, CaseStarted, RepeatFinished, SuiteFinished
from .factory import FactoryImportError, bind_seed, import_factory, takes_seed
from .limiter import JudgeLimiter, LimitedJudge, Limiter
from .loader import LoadedSuite, load_cases
from .models import (
    CUT_OFF_REASONS,
    Aggregate,
    CaseResult,
    CaseSpec,
    CollectionErrorRecord,
    JudgeUsage,
    RepeatRecord,
    SuiteResult,
    Totals,
    case_digest,
    compute_exit_code,
    cut_off_reason,
    not_run_reason_of,
    suite_digest_of,
)
from .seeds import _CURRENT_REPEAT, RepeatContext, derive_seed
from .stats import decide, pass_hat_k
from .summarize import failure_codes_of, summarize_repeats
from .xfail import settle_xfail

GRACE_S = 2.0
EventCallback = Callable[[Any], None]


@dataclass
class UnitOutcome:
    index: int
    status: str  # passed | failed | incomplete | not_run
    not_run_reason: str | None = None
    run_status: str | None = None
    decision: str | None = None
    codes: tuple[str, ...] = ()
    trace_id: str | None = None
    duration_ms: float | None = None
    judge_calls: int = 0
    judge_cached: int = 0
    judge_tokens: int = 0
    judge_cost: float = 0.0
    judge_unknown: int = 0
    judge_error: bool = False
    report: DecisionReport | None = None
    setup_error: ErrorInfo | None = None
    zombie: bool = False


def not_run(index: int, reason: str) -> UnitOutcome:
    return UnitOutcome(index, "not_run", reason)


@dataclass
class CaseFailure:
    code: str  # harness_config | expect_config | agent_import | no_agent_factory | internal | judge_required
    message: str
    error: ErrorInfo | None = None


class SuiteState:
    """Cancellation, deadlines and event emission shared by every plan of one run."""

    def __init__(self, config: SuiteConfig, on_event: EventCallback | None):
        self.config = config
        self.lock = threading.RLock()
        self.cancelled = False
        self.fail_fast_hit = False
        self.interrupted = False
        self.aborted: BaseException | None = None
        self.started = time.monotonic()
        self.deadline = (
            None if config.suite_timeout_s is None else self.started + config.suite_timeout_s
        )
        self._callbacks: list[EventCallback | None] = [on_event] if on_event else []
        self.warnings: list[str] = []

    def past_deadline(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def abort(self, exc: BaseException) -> None:
        with self.lock:
            if self.aborted is None:
                self.aborted = exc
            self.cancelled = True

    def warn(self, text: str) -> None:
        """Never raises: with warnings turned into errors (``-W error``) the text goes to stderr,
        so a warning cannot abort the run or escape the event path."""
        with self.lock:
            self.warnings.append(text)
        try:
            warnings.warn(text, DisseqtWarning, stacklevel=3)
        except Exception:
            print(f"DisseqtWarning: {text}", file=sys.stderr)

    def emit(self, event: Any) -> None:
        """Serialized; a callback that raises is disabled (DSQ-W08), never fatal."""
        with self.lock:
            for i, cb in enumerate(self._callbacks):
                if cb is None:
                    continue
                try:
                    cb(event)
                except Exception as exc:
                    self._callbacks[i] = None
                    self.warn(format_code("DSQ-W08", type=type(exc).__name__, msg=str(exc)))


class Plan:
    """Everything needed to run and then judge one case."""

    def __init__(
        self, spec: CaseSpec, index: int, n: int, k: int, threshold: float, config: SuiteConfig,
        judge: Judge | None, needs_judge: bool, factory: Callable[..., Any] | None, given: Orchestrator | None,
    ):  # fmt: skip
        self.spec, self.index, self.n, self.k, self.threshold = spec, index, n, k, threshold
        self.config = config
        self.judge = judge
        self.factory = factory
        self.given = given
        self.case: AgentTestCase = spec.case
        self.seeds = [derive_seed(config.seed, spec.case.id, r) for r in range(n)]
        self.timeout_s = spec.run.timeout_s or config.run_timeout_s
        self.seeded = factory is not None and takes_seed(factory)
        use = needs_judge or config.eval_config.use_judge
        self.eval_config = (
            dataclasses.replace(config.eval_config, use_judge=use)
            if use != config.eval_config.use_judge
            else config.eval_config
        )
        self.lock = threading.RLock()
        self.outcomes: dict[int, UnitOutcome] = {}
        self.pending: set[int] = set()
        self.cancel_reason: str | None = None
        self.started_at: float | None = None
        self.failure: CaseFailure | None = None
        self.cutoff: int | None = None
        self.result: CaseResult | None = None
        self._orch: Orchestrator | None = None
        self.t0 = time.monotonic()

    # ------------------------------------------------------------------ orchestrators
    def _build(self, factory: Callable[..., Any]) -> Orchestrator:
        llm = None
        if self.spec.run.llm is not None:
            script = tuple(
                x if isinstance(x, str) else from_dict(Message, x) for x in self.spec.run.llm
            )
            llm = FakeLLM(script=script)
        options: dict[str, Any] = {"timeout_s": self.timeout_s}
        if llm is not None:
            options["llm"] = llm
        return build_test_orchestrator(factory, self.case.fakes, **options)

    def orchestrator_for(self, r: int) -> Orchestrator:
        if self.given is not None:
            return self.given
        assert self.factory is not None
        if self.seeded:
            return self._build(bind_seed(self.factory, self.seeds[r]))
        with self.lock:
            if self._orch is None:
                self._orch = self._build(self.factory)
            return self._orch

    # ------------------------------------------------------------------ state
    def mark_started(self) -> None:
        with self.lock:
            if self.started_at is None:
                self.started_at = time.monotonic()

    def past_case_deadline(self) -> bool:
        limit = self.config.case_timeout_s
        return (
            limit is not None
            and self.started_at is not None
            and time.monotonic() > self.started_at + limit
        )

    def fail(self, code: str, message: str, exc: BaseException | None = None) -> None:
        with self.lock:
            if self.failure is None:
                err = (
                    ErrorInfo(
                        type(exc).__name__, type(exc).__module__, f"{type(exc).__name__}: {exc}"
                    )
                    if exc is not None
                    else None
                )
                self.failure = CaseFailure(code, message, err)
            if self.cancel_reason is None:
                self.cancel_reason = "cancelled"

    def skip_reason(self, suite: SuiteState) -> str | None:
        if suite.cancelled:
            return "fail_fast" if suite.fail_fast_hit else "cancelled"
        if self.cancel_reason:
            return self.cancel_reason
        if suite.past_deadline():
            return "suite_timeout"
        if self.past_case_deadline():
            self.cancel_reason = "case_timeout"
            return "case_timeout"
        return None

    def record(self, outcome: UnitOutcome) -> bool:
        """Store an outcome (first one wins) and apply early stopping. True if it was stored."""
        with self.lock:
            if outcome.index not in self.pending:
                return False
            self.pending.discard(outcome.index)
            self.outcomes[outcome.index] = outcome
            if self.config.early_stop and self.cutoff is None and self.cancel_reason is None:
                self._early_stop_check()
            return True

    def _early_stop_check(self) -> None:
        """The prefix rule: walk 0, 1, 2... while results exist; the cutoff depends only on the
        outcome sequence, never on completion order or timing."""
        p = f = 0
        j = 0
        while j in self.outcomes and j < self.n:
            o = self.outcomes[j]
            p += o.status == "passed"
            f += o.status == "failed"
            j += 1
            if decide(p, f, self.n - p - f, self.k, self.threshold) != "undecided":
                self.cutoff = j
                self.cancel_reason = "early_stop"
                return


def resolve_n_k(
    spec: CaseSpec, config: SuiteConfig, suite: SuiteState | None = None
) -> tuple[int, int, float]:
    case = spec.case
    n = (
        config.repeats
        if config.repeats is not None
        else (case.repeats if case.repeats > 1 else config.default_repeats)
    )
    k = case.pass_k or n
    if k > n:
        if suite is not None:
            suite.warn(format_code("DSQ-W04", n=n, k=k))
        k = n
    return n, k, case.pass_k_threshold


def make_judge(judge: Judge | None, config: SuiteConfig, limiter: Limiter | None) -> Judge | None:
    if judge is None:
        return None
    if limiter is not None:
        return LimitedJudge(judge, limiter)
    if config.judge_concurrency or config.judge_rps:
        return LimitedJudge(judge, JudgeLimiter(config.judge_concurrency or 0, config.judge_rps))
    return judge


def _skipped(
    spec: CaseSpec, plan_info: tuple[int, int, float], code: str, reason: str
) -> CaseResult:
    n, k, t = plan_info
    return CaseResult(
        spec.case.id, "skipped", source=spec.source, tags=spec.case.tags, reason_code=code, reason=reason,
        n_planned=n, k=k, threshold=t, case_digest=case_digest(spec, n, k, t),
    )  # fmt: skip


def _error(
    spec: CaseSpec,
    plan_info: tuple[int, int, float],
    code: str,
    reason: str,
    error: ErrorInfo | None = None,
) -> CaseResult:
    n, k, t = plan_info
    return CaseResult(
        spec.case.id, "error", source=spec.source, tags=spec.case.tags, reason_code=code, reason=reason,
        error=error, n_planned=n, k=k, threshold=t, case_digest=case_digest(spec, n, k, t),
    )  # fmt: skip


def prepare(
    spec: CaseSpec, index: int, config: SuiteConfig, suite: SuiteState, judge: Judge | None,
    agent_factory: Callable[..., Any] | None, given: Orchestrator | None,
) -> "Plan | CaseResult":  # fmt: skip
    """Decide before any work whether the case runs at all (skip / error) or build its plan."""
    n, k, t = resolve_n_k(spec, config, suite)
    info = (n, k, t)
    cid = spec.case.id
    if spec.run.skip:
        return _skipped(spec, info, "skipped_by_case", format_code("DSQ-S03", reason=spec.run.skip))
    for module in spec.run.requires:
        if importlib.util.find_spec(module) is None:
            return _skipped(
                spec, info, "missing_dependency", format_code("DSQ-S05", module=module, id=cid)
            )
    if suite.past_deadline():
        return CaseResult(
            cid, "incomplete", source=spec.source, tags=spec.case.tags, reason_code="suite_timeout",
            reason=format_code("DSQ-S02", s=config.suite_timeout_s, id=cid),
            n_planned=n, k=k, threshold=t, case_digest=case_digest(spec, n, k, t), comparable=False,
        )  # fmt: skip
    if config.only_repeat is not None and not 0 <= config.only_repeat < n:
        raise SuiteConfigError(format_code("DSQ-U02", i=config.only_repeat, id=cid, n=n))
    needs_judge = bool(spec.case.use_judge)
    if needs_judge and judge is None:
        if config.judge_missing == "skip":
            return _skipped(spec, info, "judge_unavailable", format_code("DSQ-S01", id=cid))
        if config.judge_missing == "fail":
            return _error(spec, info, "judge_required", format_code("DSQ-E05", id=cid))
    factory = agent_factory
    if given is None:
        if spec.run.agent:
            try:
                factory = import_factory(spec.run.agent)
            except FactoryImportError as exc:
                return _error(
                    spec,
                    info,
                    "agent_import",
                    format_code(
                        "DSQ-E02", path=exc.path, id=cid, type=exc.exc_type, msg=exc.detail
                    ),
                )
        if factory is None:
            return _error(spec, info, "no_agent_factory", format_code("DSQ-E01", id=cid))
    return Plan(spec, index, n, k, t, config, judge, needs_judge, factory, given)


# ------------------------------------------------------------------------------------ one unit
def outcome_of(r: int, result: Any, report: DecisionReport, config: SuiteConfig) -> UnitOutcome:
    calls = list(report.judge_calls)
    return UnitOutcome(
        r, report.outcome if report.outcome in ("passed", "failed") else "incomplete",
        run_status=result.status, decision=str(report.decisions[0]["label"]) if report.decisions else None,
        codes=tuple(failure_codes_of(report)), trace_id=result.trace_id, duration_ms=result.duration_ms,
        judge_calls=len(calls), judge_cached=sum(1 for c in calls if c.get("cached")),
        judge_tokens=sum(int(c.get("tokens") or 0) for c in calls),
        judge_cost=sum(float(c.get("cost") or 0.0) for c in calls),
        judge_unknown=sum(1 for c in calls if c.get("cost") is None),
        judge_error=any(c.get("status") == "error" for c in calls), report=report,
        setup_error=result.error if result.status == "setup_failed" else None,
        zombie=any("worker thread still running" in w for w in result.warnings),
    )  # fmt: skip


def classify_failure(plan: Plan, exc: Exception) -> None:
    cid = plan.case.id
    if isinstance(exc, HarnessConfigError):
        plan.fail("harness_config", format_code("DSQ-E03", id=cid, msg=str(exc)), exc)
    elif isinstance(exc, ExpectConfigError):
        plan.fail("expect_config", format_code("DSQ-E07", id=cid, msg=str(exc)), exc)
    else:
        plan.fail(
            "internal", format_code("DSQ-E06", id=cid, type=type(exc).__name__, msg=str(exc)), exc
        )


def run_unit(plan: Plan, r: int, suite: SuiteState) -> UnitOutcome:
    reason = plan.skip_reason(suite)
    if reason is not None:
        return not_run(r, reason)
    plan.mark_started()
    token = _CURRENT_REPEAT.set(RepeatContext(plan.case.id, r, plan.seeds[r], plan.n))
    try:
        orch = plan.orchestrator_for(r)
        result = orch.run(plan.case.input, follow_ups=plan.case.follow_ups)
        report = evaluate(
            result, plan.case.expect, case_id=plan.case.id, judge=plan.judge, config=plan.eval_config,
            fakes=orch.fakes,
        )  # fmt: skip
        return outcome_of(r, result, report, plan.config)
    except (HarnessConfigError, ExpectConfigError) as exc:
        classify_failure(plan, exc)
        return not_run(r, "cancelled")
    except Exception as exc:
        classify_failure(plan, exc)
        return not_run(r, "cancelled")
    except BaseException as exc:  # KeyboardInterrupt, pytest.skip inside a handler, ...
        suite.abort(exc)
        raise
    finally:
        _CURRENT_REPEAT.reset(token)


# ------------------------------------------------------------------------------------ results
def _breakdown(plan: Plan, outcomes: list[UnitOutcome]) -> str:
    parts = []
    inc = [o for o in outcomes if o.status == "incomplete"]
    if inc:
        codes: dict[str, int] = {}
        for o in inc:
            for c in o.codes or ("unknown",):
                codes[c] = codes.get(c, 0) + 1
        detail = "; ".join(f"{c}×{n}" for c, n in sorted(codes.items()))
        parts.append(f"{len(inc)} incomplete: {detail}")
    runs: dict[str, int] = {}
    for o in outcomes:
        if o.status == "not_run":
            runs[o.not_run_reason or "cancelled"] = runs.get(o.not_run_reason or "cancelled", 0) + 1
    for reason, count in sorted(runs.items()):
        parts.append(f"{count} not run: {reason}")
    return "; ".join(parts) or "none"


def finalize(plan: Plan, suite: SuiteState) -> CaseResult:
    """Build the ``CaseResult`` once every unit is recorded."""
    with plan.lock:
        if plan.result is not None:
            return plan.result
        spec, n, k, t = plan.spec, plan.n, plan.k, plan.threshold
        outs: list[UnitOutcome] = []
        for r in range(n):
            o = plan.outcomes.get(r) or not_run(r, plan.cancel_reason or "cancelled")
            if plan.cutoff is not None and r >= plan.cutoff:
                o = not_run(r, "early_stop")
            if suite.config.only_repeat is not None and r != suite.config.only_repeat:
                o = not_run(r, "only_repeat")
            outs.append(o)
        digest = case_digest(spec, n, k, t)
        duration = (time.monotonic() - plan.started_at) * 1000.0 if plan.started_at else None
        ran = [o for o in outs if o.status != "not_run"]
        if plan.failure is not None:
            f = plan.failure
            result = CaseResult(
                spec.case.id, "error", source=spec.source, tags=spec.case.tags, reason_code=f.code,
                reason=f.message, error=f.error, n_planned=n, k=k, threshold=t, case_digest=digest,
            )  # fmt: skip
        elif not ran:
            reason = not_run_reason_of(o.not_run_reason for o in outs) or "cancelled"
            text = {
                "suite_timeout": format_code(
                    "DSQ-S02", s=suite.config.suite_timeout_s, id=spec.case.id
                ),
                "fail_fast": format_code("DSQ-S04"),
            }.get(reason or "", f"no repeat ran ({reason})")
            # Cut off by a timeout or a cancellation, the case is undecided (exit 2), never a green
            # skip; fail-fast and only-repeat/early-stop are deliberate skips.
            cut_off = reason in CUT_OFF_REASONS
            result = CaseResult(
                spec.case.id, "incomplete" if cut_off else "skipped", source=spec.source, tags=spec.case.tags,
                reason_code=reason, reason=text,
                n_planned=n, k=k, threshold=t, case_digest=digest, comparable=False,
                repeats=tuple(_record(plan, o, None, "failing") for o in outs),
            )  # fmt: skip
        elif (
            all(o.run_status == "setup_failed" for o in ran)
            and len({o.setup_error.type for o in ran if o.setup_error}) == 1
        ):
            err = next(o.setup_error for o in ran if o.setup_error)
            result = CaseResult(
                spec.case.id, "error", source=spec.source, tags=spec.case.tags, reason_code="agent_setup_failed",
                reason=format_code("DSQ-E04", n=len(ran), id=spec.case.id, msg=err.message), error=err,
                n_planned=n, k=k, threshold=t, case_digest=digest,
            )  # fmt: skip
        else:
            reports = [o.report if o.status != "not_run" else None for o in outs]
            rr = summarize_repeats(reports, k=k, threshold=t, n_planned=n, case_id=spec.case.id)
            status = rr.outcome
            only = suite.config.only_repeat
            if only is not None and status == "incomplete" and outs[only].status == "passed":
                status = "passed"  # a replay of one passing repeat is not 'undecided'
            code, reason = None, ""
            d = rr.n_passed + rr.n_failed
            undecided = n - d
            if status == "failed":
                if undecided > 0:
                    code, reason = "threshold_unreachable", format_code(
                        "DSQ-F02", id=spec.case.id, t=t, U=undecided, P=rr.n_passed, D=d, n=n
                    )
                else:
                    code, reason = "below_threshold", format_code(
                        "DSQ-F01",
                        id=spec.case.id,
                        k=k,
                        v=rr.pass_hat_k or 0.0,
                        t=t,
                        P=rr.n_passed,
                        D=d,
                        n=n,
                    )
            elif status == "incomplete":
                code = cut_off_reason(o.not_run_reason for o in outs) or "undecided"
                reason = format_code(
                    "DSQ-I01",
                    id=spec.case.id,
                    U=undecided,
                    n=n,
                    breakdown=_breakdown(plan, outs),
                    k=k,
                    t=t,
                )
            worst = int(rr.worst) if rr.worst is not None else None
            keep = suite.config.keep_reports
            records = tuple(_record(plan, o, worst, keep) for o in outs)
            stopped = plan.cutoff is not None
            result = CaseResult(
                spec.case.id, status, source=spec.source, tags=spec.case.tags, reason_code=code, reason=reason,
                n_planned=n, k=k, threshold=t, repeat_report=rr, repeats=records, stopped_early=stopped,
                comparable=not stopped and all(o.status != "not_run" for o in outs), case_digest=digest,
                duration_ms=duration,
            )  # fmt: skip
            plan.result = result = settle_xfail(result, spec.run.xfail)
            return result
        plan.result = result = settle_xfail(result, spec.run.xfail)
        return result


def _record(plan: Plan, o: UnitOutcome, worst: int | None, keep: str) -> RepeatRecord:
    keep_report = o.report is not None and (
        keep == "all"
        or (keep == "failing" and o.status in ("failed", "incomplete"))
        or (keep == "worst" and o.index == worst)
    )
    return RepeatRecord(
        o.index, plan.seeds[o.index], o.status, not_run_reason=o.not_run_reason, run_status=o.run_status,
        decision=o.decision, failure_codes=o.codes, trace_id=o.trace_id, duration_ms=o.duration_ms,
        judge_calls=o.judge_calls, judge_tokens=o.judge_tokens, judge_cost_usd=o.judge_cost,
        judge_cost_unknown=o.judge_unknown, judge_error=o.judge_error,
        report=o.report if keep_report else None,
    )  # fmt: skip


def build_result(
    results: list[CaseResult], errors: tuple[CollectionErrorRecord, ...], config: SuiteConfig, suite: SuiteState,
    t0: float, zombies: int,
) -> SuiteResult:  # fmt: skip
    cases = tuple(sorted(results, key=lambda c: c.source_key))
    totals = Totals(
        cases=len(cases), passed=sum(c.status == "passed" for c in cases), failed=sum(c.status == "failed" for c in cases),
        incomplete=sum(c.status == "incomplete" for c in cases), skipped=sum(c.status == "skipped" for c in cases),
        error=sum(c.status == "error" for c in cases),
        xfailed=sum(c.status == "xfailed" for c in cases),
        repeats_run=sum(1 for c in cases for r in c.repeats if r.status != "not_run"),
        repeats_not_run=sum(1 for c in cases for r in c.repeats if r.status == "not_run"),
    )  # fmt: skip
    with_rr = [c for c in cases if c.repeat_report is not None]
    p1 = [
        c.repeat_report.pass_at_1
        for c in with_rr
        if c.repeat_report and c.repeat_report.pass_at_1 is not None
    ]
    hk = [
        c.repeat_report.pass_hat_k
        for c in with_rr
        if c.repeat_report and c.repeat_report.pass_hat_k is not None
    ]
    curve: list[tuple[int, float]] = []
    for k in range(1, max((c.n_planned for c in with_rr), default=0) + 1):
        vals = [
            v for c in with_rr if c.repeat_report
            for v in [pass_hat_k(c.repeat_report.n_passed, c.repeat_report.n_passed + c.repeat_report.n_failed, k)]
            if v is not None
        ]  # fmt: skip
        if vals:
            curve.append((k, sum(vals) / len(vals)))
    aggregate = Aggregate(
        sum(p1) / len(p1) if p1 else None, sum(hk) / len(hk) if hk else None,
        tuple(sorted(c.case_id for c in with_rr if c.repeat_report and 0 < c.repeat_report.n_passed < c.repeat_report.n_passed + c.repeat_report.n_failed)),
        tuple(sorted(c.case_id for c in with_rr if c.repeat_report and c.repeat_report.n_passed == 0 and c.repeat_report.n_failed > 0)),
        tuple(curve),
    )  # fmt: skip
    reps = [r for c in cases for r in c.repeats]
    usage = JudgeUsage(
        sum(r.judge_calls for r in reps), 0, sum(r.judge_tokens for r in reps), sum(r.judge_cost_usd for r in reps),
        sum(r.judge_cost_unknown for r in reps), sum(1 for r in reps if r.judge_error),
    )  # fmt: skip
    try:
        from importlib.metadata import version

        sdk = version("disseqt-ai-sdk")
    except Exception:
        sdk = "unknown"
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "disseqt_ai_sdk": sdk,
        "zombie_threads": zombies,
    }
    return SuiteResult(
        cases, suite_id=uuid.uuid4().hex, run_label=config.run_label, suite_digest=suite_digest_of(cases),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_ms=(time.monotonic() - t0) * 1000.0, seed=config.seed, config=config, environment=env,
        totals=totals, aggregate=aggregate, judge_usage=usage, collection_errors=errors,
        interrupted=suite.interrupted,
        exit_code=compute_exit_code(cases, errors, interrupted=suite.interrupted, incomplete_policy=config.incomplete_policy),
    )  # fmt: skip


# ------------------------------------------------------------------------------------ entry points
def to_specs(
    cases: Any, filter: Any = None
) -> tuple[list[CaseSpec], tuple[CollectionErrorRecord, ...]]:
    if isinstance(cases, LoadedSuite):
        specs = list(cases.cases)
        if filter is not None and filter.active:
            specs = [s for s in specs if filter.matches(s.case.id, s.case.tags)]
        return specs, cases.errors
    if isinstance(cases, CaseSpec):
        cases = [cases]
    loaded = load_cases(cases, filter=filter)
    return list(loaded.cases), loaded.errors


def _finish_if_done(plan: Plan, suite: SuiteState, results: list[CaseResult]) -> None:
    with plan.lock:
        if plan.pending or plan.result is not None:
            return
    result = finalize(plan, suite)
    results.append(result)
    suite.emit(CaseFinished(plan.case.id, result.status))
    if suite.config.fail_fast and result.status in ("failed", "error") and not suite.cancelled:
        suite.fail_fast_hit = True
        suite.cancelled = True


def run_suite(
    cases: Any, *, agent_factory: Callable[..., Any] | None = None, config: SuiteConfig = SuiteConfig(),
    judge: Judge | None = None, limiter: Limiter | None = None, filter: Any = None,
    on_event: EventCallback | None = None, orchestrator: Orchestrator | None = None,
    suite_deadline: float | None = None,
) -> SuiteResult:  # fmt: skip
    """Run every case's repeats and return a deterministic ``SuiteResult``.

    Raises only ``SuiteConfigError`` (before any work), ``KeyboardInterrupt`` (with the partial
    result on ``exc.suite_result``) or a non-``Exception`` ``BaseException`` that came out of an
    agent. Everything else becomes a record.
    """
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
    pool = None
    futures: dict[concurrent.futures.Future[UnitOutcome], tuple[Plan, int]] = {}
    try:
        if plans:
            pool = _make_pool(config)
            for plan in plans:
                suite.emit(CaseStarted(plan.case.id, plan.n))
                for r in range(plan.n):
                    plan.pending.add(r)
                for r in range(plan.n):
                    if config.only_repeat is not None and r != config.only_repeat:
                        plan.record(not_run(r, "only_repeat"))
                        continue
                    futures[pool.submit(run_unit, plan, r, suite)] = (plan, r)
                _finish_if_done(plan, suite, results)
            while futures:
                done, _ = concurrent.futures.wait(
                    list(futures), timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    plan, r = futures.pop(fut)
                    if fut.cancelled():
                        continue
                    exc = fut.exception()
                    if exc is not None and not isinstance(exc, Exception):
                        raise exc  # an agent's BaseException (for example pytest.skip) surfaces here
                    out = fut.result() if exc is None else not_run(r, "cancelled")
                    stored = plan.record(out)
                    if stored:
                        suite.emit(RepeatFinished(plan.case.id, r, out.status))
                    _finish_if_done(plan, suite, results)
                _enforce_case_deadlines(plans, futures, suite, results)
    except KeyboardInterrupt as exc:
        suite.interrupted = True
        suite.cancelled = True
        if pool is not None:
            pool.shutdown()
        _abandon_all(plans, suite, results)
        partial = build_result(
            results, tuple(_with_collection(errors)), config, suite, t0, _zombies(plans)
        )
        exc.suite_result = partial  # type: ignore[attr-defined]
        raise
    except BaseException:
        if pool is not None:
            pool.shutdown()
        raise
    if pool is not None:
        pool.shutdown()
    _abandon_all(plans, suite, results)
    result = build_result(
        results, tuple(_with_collection(errors)), config, suite, t0, _zombies(plans)
    )
    suite.emit(SuiteFinished(len(result.cases), result.exit_code))
    return result


def _with_collection(errors: tuple[CollectionErrorRecord, ...]) -> list[CollectionErrorRecord]:
    return [e for e in errors if e.code != "DSQ-W10"]


def _zombies(plans: list[Plan]) -> int:
    return sum(1 for p in plans for o in p.outcomes.values() if o.zombie)


def _make_pool(config: SuiteConfig) -> Any:
    from .pool import DaemonPool

    return DaemonPool(config.max_workers)


def _enforce_case_deadlines(
    plans: list[Plan],
    futures: dict[Any, tuple[Plan, int]],
    suite: SuiteState,
    results: list[CaseResult],
) -> None:
    """A case past its deadline stops starting units; units still running get ``GRACE_S`` to
    return before they are abandoned (recorded ``not_run/case_timeout``). The futures of an
    abandoned case leave the wait set, so a stuck unit cannot hold the run open."""
    limit = suite.config.case_timeout_s
    if limit is None:
        return
    now = time.monotonic()
    for plan in plans:
        if plan.result is not None or plan.started_at is None:
            continue
        if now > plan.started_at + limit:
            plan.cancel_reason = plan.cancel_reason or "case_timeout"
            if now > plan.started_at + limit + GRACE_S:
                for r in sorted(plan.pending):
                    plan.record(not_run(r, "case_timeout"))
                _finish_if_done(plan, suite, results)
    for fut in [f for f, (plan, _) in futures.items() if plan.result is not None]:
        fut.cancel()
        del futures[fut]


def _abandon_all(plans: list[Plan], suite: SuiteState, results: list[CaseResult]) -> None:
    done_ids = {r.case_id for r in results}
    for plan in plans:
        if plan.case.id in done_ids:
            continue
        for r in sorted(plan.pending):
            plan.record(
                not_run(
                    r, plan.cancel_reason or ("cancelled" if suite.cancelled else "suite_timeout")
                )
            )
        results.append(finalize(plan, suite))


def run_case(
    case: Any, *, agent_factory: Callable[..., Any] | None = None, orchestrator: Orchestrator | None = None,
    config: SuiteConfig = SuiteConfig(), judge: Judge | None = None, limiter: Limiter | None = None,
    on_event: EventCallback | None = None, suite_deadline: float | None = None,
) -> CaseResult:  # fmt: skip
    """One case, same engine as ``run_suite`` (so the verdict is identical either way)."""
    spec = case if isinstance(case, CaseSpec) else CaseSpec(case)
    result = run_suite(
        [spec], agent_factory=agent_factory, orchestrator=orchestrator, config=config, judge=judge,
        limiter=limiter, on_event=on_event, suite_deadline=suite_deadline,
    )  # fmt: skip
    return result.cases[0]


_ = (field, Mapping, RepeatReport)
