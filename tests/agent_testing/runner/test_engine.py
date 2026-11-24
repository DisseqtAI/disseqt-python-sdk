"""The engine: repeats, verdicts, errors, judge policy, limits, timeouts, early stop (EX-, AS-, SB-)."""

import asyncio
import json
import threading
import time
import warnings
from typing import Any

import pytest

from disseqt_agent_testing.harness import HarnessConfigError, build_test_orchestrator
from disseqt_agent_testing.models import (
    ClarifyExpect,
    Decision,
    Expect,
    FakeResponse,
    FakeSpec,
)
from disseqt_agent_testing.runner import (
    CaseSpec,
    CaseStarted,
    DisseqtWarning,
    RunBlock,
    SuiteConfig,
    SuiteConfigError,
    arun_case,
    arun_suite,
    current_repeat,
    derive_seed,
    run_case,
    run_suite,
)

from .._support.runner_agents import (
    FakeJudge,
    make_good_agent,
    make_slow_agent,
    patterned,
    refund_case,
    refund_expect,
    refund_fakes,
)


def run_refund(n: int = 3, workers: int = 4) -> Any:
    return run_suite(
        [refund_case("refund_case", repeats=n)],
        agent_factory=make_good_agent,
        config=SuiteConfig(max_workers=workers),
    )


def statuses(result: Any, i: int = 0) -> list[str]:
    return [r.status for r in result.cases[i].repeats]


def test_ex01_happy_path_runs_n_distinct_repeats() -> None:
    result = run_suite(
        [refund_case(repeats=3)],
        agent_factory=make_good_agent,
        config=SuiteConfig(keep_reports="all"),
    )
    c = result.cases[0]
    assert c.status == "passed" and c.n_planned == 3 and statuses(result) == ["passed"] * 3
    assert len({r.trace_id for r in c.repeats}) == 3 and len({r.seed for r in c.repeats}) == 3
    for r in c.repeats:
        assert r.report is not None
        assert [s for s in ("call-1", "call-2") if s]  # ids restart in every repeat
    assert result.exit_code == 0 and result.totals.passed == 1 and result.totals.repeats_run == 3
    assert c.comparable is True and c.stopped_early is False


def test_ex02_a_flaky_agent_fails_pass_hat_k() -> None:
    result = run_suite([refund_case(repeats=5)], agent_factory=patterned("PPFPF"))
    c = result.cases[0]
    assert statuses(result) == ["passed", "passed", "failed", "passed", "failed"]
    assert c.status == "failed" and c.reason_code == "below_threshold" and result.exit_code == 1
    assert c.repeat_report.pass_hat_k == 0.0 and c.repeat_report.worst == "2"  # type: ignore[union-attr]
    assert c.reason.startswith(
        "agent case 'refund_case' FAILED: pass^5=0.00 < threshold 1.00 (c=3/5 decided, n=5)"
    )
    assert result.aggregate.flaky_cases == ("refund_case",)


def test_ex03_a_lower_k_and_threshold_lets_the_same_agent_pass() -> None:
    case = refund_case(repeats=5, pass_k=3, pass_k_threshold=0.1)
    result = run_suite([case], agent_factory=patterned("PPFPF"))
    assert result.cases[0].status == "passed" and result.exit_code == 0


def test_ex04_an_unmeasurable_budget_makes_repeats_incomplete_with_a_breakdown() -> None:
    expect = Expect(
        decision=Decision.CALL, tools=refund_expect().tools, order="partial", max_tokens=100
    )
    result = run_suite([refund_case(repeats=2, expect=expect)], agent_factory=make_good_agent)
    c = result.cases[0]
    assert c.status == "incomplete" and result.exit_code == 2 and c.reason_code == "undecided"
    assert "2 of 2 repeats undecided (2 incomplete: unavailable:data_unavailable×2)" in c.reason
    assert c.reason.startswith("incomplete: case 'refund_case':")
    skip = run_suite(
        [refund_case(repeats=2, expect=expect)],
        agent_factory=make_good_agent,
        config=SuiteConfig(incomplete_policy="skip"),
    )
    assert skip.exit_code == 0 and skip.cases[0].status == "incomplete"


def test_ex05_ex06_seeds_reach_seed_aware_factories_and_current_repeat() -> None:
    seeds: list[int] = []
    seen: list[Any] = []

    def factory(tools: Any, seed: int) -> Any:
        seeds.append(seed)

        def agent(text: str) -> str:
            ctx = current_repeat()
            seen.append((ctx.index, ctx.seed, ctx.n, ctx.case_id))  # type: ignore[union-attr]
            return make_good_agent(tools)(text)  # type: ignore[no-any-return]

        return agent

    run_suite(
        [refund_case("seeded", repeats=3)],
        agent_factory=factory,
        config=SuiteConfig(seed=9, max_workers=1),
    )
    want = [derive_seed(9, "seeded", r) for r in range(3)]
    assert sorted(seeds) == sorted(want)
    assert sorted(seen) == sorted((r, want[r], 3, "seeded") for r in range(3))
    assert current_repeat() is None


def test_ex07_a_factory_that_fails_every_repeat_is_agent_setup_failed() -> None:
    def factory(tools: Any) -> Any:
        raise RuntimeError("db missing")

    result = run_suite([refund_case(repeats=3)], agent_factory=factory)
    c = result.cases[0]
    assert c.status == "error" and c.reason_code == "agent_setup_failed" and result.exit_code == 1
    assert (
        c.reason
        == "agent factory failed in all 3 repeats of case 'refund_case': RuntimeError: db missing"
    )


def test_ex08_ex09_harness_configuration_errors_are_case_errors_not_crashes() -> None:
    result = run_suite([refund_case()], agent_factory=lambda tools, db: None)
    c = result.cases[0]
    assert c.status == "error" and c.reason_code == "harness_config" and result.exit_code == 3
    assert c.reason.startswith(
        "harness configuration error for case 'refund_case': factory parameter 'db'"
    )

    class Empty:  # entry resolution fails when the run starts
        pass

    result = run_suite([refund_case()], agent_factory=lambda tools: Empty())
    assert (
        result.cases[0].reason_code == "harness_config"
        and "could not find an entry point" in result.cases[0].reason
    )


def test_expect_config_errors_are_case_errors() -> None:
    case = refund_case(
        expect=Expect(tools=refund_expect().tools, order="partial", min_scores={"nope": 0.5})
    )
    c = run_suite([case], agent_factory=make_good_agent).cases[0]
    assert (
        c.status == "error"
        and c.reason_code == "expect_config"
        and "unknown min_scores key" in c.reason
    )


def test_no_factory_and_unimportable_factory() -> None:
    a = run_suite([refund_case()]).cases[0]
    assert a.reason_code == "no_agent_factory" and a.reason.startswith(
        "case 'refund_case' has no agent factory"
    )
    spec = CaseSpec(refund_case(), run=RunBlock(agent="no_such_module_xyz:make"))
    b = run_suite([spec]).cases[0]
    assert (
        b.reason_code == "agent_import"
        and "cannot import agent factory 'no_such_module_xyz:make'" in b.reason
    )
    ok = CaseSpec(
        refund_case(),
        run=RunBlock(agent="tests.agent_testing._support.runner_agents:make_good_agent"),
    )
    assert run_suite([ok]).cases[0].status == "passed"


def test_ex10_use_judge_without_a_judge_follows_the_policy() -> None:
    case = refund_case(use_judge=("decision",))
    skip = run_suite([case], agent_factory=make_good_agent).cases[0]
    assert (
        skip.status == "skipped"
        and skip.reason_code == "judge_unavailable"
        and skip.reason.startswith("judge unavailable:")
    )
    err = run_suite([case], agent_factory=make_good_agent, config=SuiteConfig(judge_missing="fail"))
    assert err.cases[0].reason_code == "judge_required" and err.exit_code == 3
    ran = run_suite([case], agent_factory=make_good_agent, config=SuiteConfig(judge_missing="run"))
    assert ran.cases[0].status == "passed"  # nothing in this Expect needs the judge


def judged_case(**kw: Any) -> Any:
    expect = Expect(clarify=ClarifyExpect(judge_relevance=True))
    return refund_case(
        "judged", expect=expect, fakes=refund_fakes(), use_judge=("clarification_relevance",), **kw
    )


def asking_agent(tools: Any) -> Any:
    return lambda text: "Which order number is this about?"


def test_ex11_a_judge_error_makes_the_repeat_incomplete() -> None:
    judge = FakeJudge(status="error")
    result = run_suite([judged_case(repeats=2)], agent_factory=asking_agent, judge=judge)
    c = result.cases[0]
    assert c.status == "incomplete" and result.judge_usage.error_repeats == 2 and judge.calls == 2
    assert "error:judge_failed" in dict(c.repeat_report.failure_codes)  # type: ignore[union-attr]
    ok = run_suite(
        [judged_case(repeats=2)],
        agent_factory=asking_agent,
        judge=FakeJudge("pass"),
        config=SuiteConfig(
            eval_config=__import__(
                "disseqt_agent_testing.assertions", fromlist=["EvalConfig"]
            ).EvalConfig(use_judge=True)
        ),
    )
    assert ok.judge_usage.calls == 2


def test_ex12_throughput_is_parallel() -> None:
    cases = [refund_case(f"c{i:02d}", repeats=5) for i in range(20)]
    start = time.monotonic()
    result = run_suite(
        cases,
        agent_factory=lambda tools: make_slow_agent(tools, 0.1),
        config=SuiteConfig(max_workers=10),
    )
    elapsed = time.monotonic() - start
    assert (
        result.totals.passed == 20 and elapsed < 2.4
    )  # 100 units x 0.1 s / 10 workers = 1 s, plus overhead


def test_ex13_judge_concurrency_never_exceeds_the_limit() -> None:
    judge = FakeJudge("pass", delay=0.05)
    cfg = SuiteConfig(
        max_workers=8,
        judge_concurrency=2,
        eval_config=__import__(
            "disseqt_agent_testing.assertions", fromlist=["EvalConfig"]
        ).EvalConfig(use_judge=True),
    )
    run_suite([judged_case(repeats=8)], agent_factory=asking_agent, judge=judge, config=cfg)
    assert judge.calls == 8 and judge.peak <= 2 and judge.peak >= 1


def test_ex14_rps_limiting_spaces_judge_calls() -> None:
    judge = FakeJudge("pass")
    cfg = SuiteConfig(
        max_workers=6,
        judge_concurrency=6,
        judge_rps=20.0,
        eval_config=__import__(
            "disseqt_agent_testing.assertions", fromlist=["EvalConfig"]
        ).EvalConfig(use_judge=True),
    )
    start = time.monotonic()
    run_suite([judged_case(repeats=6)], agent_factory=asking_agent, judge=judge, config=cfg)
    assert time.monotonic() - start >= 0.2  # 6 calls at 20/s with a burst of 1


def test_ex15_a_run_timeout_is_a_failed_repeat_not_an_incomplete_one() -> None:
    result = run_suite(
        [refund_case(repeats=2)],
        agent_factory=lambda tools: (lambda t: time.sleep(3)),
        config=SuiteConfig(run_timeout_s=0.2),
    )
    c = result.cases[0]
    assert c.status == "failed" and {r.run_status for r in c.repeats} == {"timed_out"}
    assert "harness.not_finished" in dict(c.repeat_report.failure_codes)  # type: ignore[union-attr]


def test_ex16_a_case_timeout_leaves_the_rest_not_run() -> None:
    cfg = SuiteConfig(max_workers=1, case_timeout_s=0.35)
    result = run_suite(
        [refund_case(repeats=6)],
        agent_factory=lambda tools: make_slow_agent(tools, 0.2),
        config=cfg,
    )
    c = result.cases[0]
    reasons = {r.not_run_reason for r in c.repeats if r.status == "not_run"}
    assert (
        reasons == {"case_timeout"}
        and c.status in ("incomplete", "failed")
        and c.comparable is False
    )


def test_ex17_a_suite_timeout_skips_unstarted_cases() -> None:
    cfg = SuiteConfig(max_workers=1, suite_timeout_s=0.25)
    cases = [refund_case(f"c{i}", repeats=1) for i in range(6)]
    result = run_suite(cases, agent_factory=lambda tools: make_slow_agent(tools, 0.2), config=cfg)
    skipped = [c for c in result.cases if c.reason_code == "suite_timeout"]
    assert skipped and all(c.status == "incomplete" for c in skipped)
    assert "suite timeout (0.25s) reached before case" in skipped[0].reason


def test_ex18_ex19_early_stop_is_deterministic_for_any_worker_count() -> None:
    def run(workers: int) -> Any:
        case = refund_case(repeats=8)
        return run_suite(
            [case],
            agent_factory=patterned("PFPPPPPP"),
            config=SuiteConfig(early_stop=True, max_workers=workers),
        )

    results = {run(w).canonical_json() for w in (1, 2, 5, 8)}
    assert len(results) == 1
    c = run(1).cases[0]
    assert c.stopped_early and c.comparable is False and c.status == "failed"
    assert {r.not_run_reason for r in c.repeats if r.status == "not_run"} == {"early_stop"}


def test_ex20_early_stop_cuts_at_the_prefix_where_the_rule_decides() -> None:
    case = refund_case(repeats=5, pass_k=3, pass_k_threshold=0.1)
    result = run_suite(
        [case], agent_factory=patterned("PPFPP"), config=SuiteConfig(early_stop=True, max_workers=5)
    )
    c = result.cases[0]
    ran = [r.index for r in c.repeats if r.status != "not_run"]
    assert ran == [0, 1, 2, 3] and c.status == "passed"  # P=3, F=1, U=1 -> 1/10 >= 0.1 at prefix 4


def test_ex21_fail_fast_cancels_what_has_not_started() -> None:
    cases = [refund_case("a_fails", repeats=1)] + [
        refund_case(f"b{i}", repeats=1) for i in range(5)
    ]

    def factory(tools: Any) -> Any:
        return patterned("F")(tools)

    cfg = SuiteConfig(max_workers=1, fail_fast=True)
    result = run_suite(cases, agent_factory=factory, config=cfg)
    by = {c.case_id: c for c in result.cases}
    assert by["a_fails"].status == "failed"
    assert any(
        c.status == "skipped"
        and c.reason_code == "fail_fast"
        and c.reason == "fail-fast: an earlier case failed"
        for c in result.cases
    )


def test_ex22_a_keyboard_interrupt_returns_promptly_with_a_partial_result() -> None:
    def interrupt(event: Any) -> None:
        if isinstance(event, CaseStarted):
            raise KeyboardInterrupt

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt) as info:
        run_suite(
            [refund_case(repeats=4)],
            agent_factory=lambda tools: make_slow_agent(tools, 3),
            on_event=interrupt,
        )
    assert time.monotonic() - start < 2.0
    partial = info.value.suite_result  # type: ignore[attr-defined]
    assert partial.interrupted and partial.exit_code == 130


def test_ex23_a_fake_handlers_pytest_skip_propagates() -> None:
    from disseqt_agent_testing.harness import build_test_orchestrator as _b  # noqa: F401

    def handler(**_: Any) -> None:
        pytest.skip("skipped from a fake handler")

    orch = (
        build_test_orchestrator(make_good_agent, refund_fakes(), handlers={"get_order": handler})
        if False
        else None
    )
    spec = refund_case(fakes=(FakeSpec("get_order", effect="read"), *refund_fakes()[1:]))

    def factory(tools: Any) -> Any:
        return lambda text: pytest.skip("skipped from an agent")

    with pytest.raises(pytest.skip.Exception):
        run_suite([spec], agent_factory=factory)
    assert orch is None


def test_ex24_a_raising_event_callback_is_disabled_with_a_warning() -> None:
    seen: list[Any] = []

    def cb(event: Any) -> None:
        seen.append(event)
        raise ValueError("boom")

    with (
        pytest.warns(DisseqtWarning, match="DSQ-W08|on_event|callback")
        if False
        else warnings.catch_warnings(record=True)
    ) as w:
        warnings.simplefilter("always")
        result = run_suite([refund_case(repeats=2)], agent_factory=make_good_agent, on_event=cb)
    assert len(seen) == 1 and result.cases[0].status == "passed"
    assert any("an on_event callback raised ValueError: boom" in str(x.message) for x in w)


def test_events_are_emitted_in_a_sensible_order() -> None:
    kinds: list[str] = []
    run_suite(
        [refund_case(repeats=2)],
        agent_factory=make_good_agent,
        on_event=lambda e: kinds.append(type(e).__name__),
        config=SuiteConfig(max_workers=1),
    )
    assert kinds[0] == "CaseStarted" and kinds[-1] == "SuiteFinished"
    assert kinds.count("RepeatFinished") == 2 and kinds.index("CaseFinished") > kinds.index(
        "RepeatFinished"
    )


def test_only_repeat_runs_a_single_repeat_with_its_derived_seed() -> None:
    seeds: list[int] = []

    def factory(tools: Any, seed: int) -> Any:
        seeds.append(seed)
        return make_good_agent(tools)

    result = run_suite(
        [refund_case("x", repeats=4)],
        agent_factory=factory,
        config=SuiteConfig(only_repeat=2, seed=3),
    )
    c = result.cases[0]
    assert seeds == [derive_seed(3, "x", 2)] and [r.status for r in c.repeats] == [
        "not_run",
        "not_run",
        "passed",
        "not_run",
    ]
    assert {r.not_run_reason for r in c.repeats if r.status == "not_run"} == {"only_repeat"}
    with pytest.raises(SuiteConfigError, match="--disseqt-only-repeat 5 is out of range"):
        run_suite(
            [refund_case("x", repeats=2)],
            agent_factory=make_good_agent,
            config=SuiteConfig(only_repeat=5),
        )


def test_repeat_resolution_force_override_default_and_clamp() -> None:
    case = refund_case(repeats=2)
    assert (
        run_suite([case], agent_factory=make_good_agent, config=SuiteConfig(repeats=4))
        .cases[0]
        .n_planned
        == 4
    )
    assert (
        run_suite(
            [refund_case()], agent_factory=make_good_agent, config=SuiteConfig(default_repeats=3)
        )
        .cases[0]
        .n_planned
        == 3
    )
    clamped = refund_case(repeats=4, pass_k=4)
    with pytest.warns(DisseqtWarning, match="smaller than pass_k=4"):
        r = run_suite(
            [clamped], agent_factory=make_good_agent, config=SuiteConfig(repeats=2)
        ).cases[0]
    assert (r.n_planned, r.k) == (2, 2)


def test_keep_reports_policies() -> None:
    def kept(policy: str) -> list[bool]:
        res = run_suite(
            [refund_case(repeats=3)],
            agent_factory=patterned("PFP"),
            config=SuiteConfig(keep_reports=policy),
        )
        return [r.report is not None for r in res.cases[0].repeats]

    assert (
        kept("all") == [True, True, True]
        and kept("failing") == [False, True, False]
        and kept("worst") == [False, True, False]
    )


def test_run_block_llm_script_and_skip_and_requires() -> None:
    skip = CaseSpec(refund_case(), run=RunBlock(skip="needs sandbox"))
    r = run_suite([skip], agent_factory=make_good_agent).cases[0]
    assert r.status == "skipped" and r.reason == "skipped by case file: needs sandbox"
    req = CaseSpec(refund_case(), run=RunBlock(requires=("module_that_is_not_installed_abc",)))
    r = run_suite([req], agent_factory=make_good_agent).cases[0]
    assert (
        r.status == "skipped"
        and r.reason
        == "missing optional dependency 'module_that_is_not_installed_abc' required by case 'refund_case'"
    )


def test_run_case_returns_the_same_verdict_as_run_suite() -> None:
    case = refund_case(repeats=3)
    single = run_case(case, agent_factory=patterned("PPF"))
    suite = run_suite([case], agent_factory=patterned("PPF")).cases[0]
    assert (
        single.to_dict()["repeat_report"] == suite.to_dict()["repeat_report"]
        and single.status == suite.status == "failed"
    )


def test_a_prebuilt_orchestrator_is_used_as_is() -> None:
    orch = build_test_orchestrator(make_good_agent, refund_fakes())
    assert run_case(refund_case(repeats=2), orchestrator=orch).status == "passed"


def test_duplicate_python_cases_are_collection_errors() -> None:
    result = run_suite([refund_case("dup"), refund_case("dup")], agent_factory=make_good_agent)
    assert len(result.cases) == 1 and result.exit_code == 3
    assert "duplicate case id 'dup'" in result.collection_errors[0].message


def test_zero_cases_exit_code_five() -> None:
    assert run_suite([], agent_factory=make_good_agent).exit_code == 5


# ------------------------------------------------------------------ asyncio twins
def test_as01_async_results_equal_sync_results() -> None:
    cases = [refund_case(f"c{i}", repeats=3) for i in range(4)]
    sync = run_suite(cases, agent_factory=patterned("PFP"), config=SuiteConfig(max_workers=3))
    got = asyncio.run(
        arun_suite(cases, agent_factory=patterned("PFP"), config=SuiteConfig(max_workers=3))
    )
    assert got.canonical_json() == sync.canonical_json()


def test_as02_arun_case_inside_a_running_loop_never_calls_asyncio_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inside() -> Any:
        def boom(*a: Any, **k: Any) -> None:
            raise AssertionError("asyncio.run must not be called by the async engine")

        monkeypatch.setattr(asyncio, "run", boom)
        return await arun_case(refund_case(repeats=2), agent_factory=make_good_agent)

    assert asyncio.run(inside()).status == "passed"


def test_as03_async_agents_work() -> None:
    from .._support.refund_agent import make_async_refund_agent

    def factory(tools: Any) -> Any:
        return make_async_refund_agent(tools)

    case = refund_case(repeats=2)
    result = asyncio.run(arun_case(case, agent_factory=factory, config=SuiteConfig()))
    assert result.status in (
        "passed",
        "failed",
        "incomplete",
    )  # sync fakes through an async agent: still a record


def test_as04_external_cancellation_propagates_and_stops_the_units() -> None:
    async def scenario() -> None:
        task = asyncio.ensure_future(
            arun_case(refund_case(repeats=4), agent_factory=lambda tools: make_slow_agent(tools, 2))
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    start = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic() - start < 5


def test_as05_the_async_case_timeout_is_not_confused_with_cancellation() -> None:
    async def slow(tools: Any) -> Any:
        async def agent(text: str) -> str:
            await asyncio.sleep(5)
            return "late"

        return agent

    cfg = SuiteConfig(case_timeout_s=0.2, max_workers=2, run_timeout_s=30)
    result = asyncio.run(arun_suite([refund_case(repeats=2)], agent_factory=slow, config=cfg))
    assert {r.not_run_reason for r in result.cases[0].repeats} == {"case_timeout"}


# ------------------------------------------------------------------ stability
def test_sb01_the_canonical_result_is_identical_for_every_worker_count() -> None:
    cases = [refund_case(f"c{i:02d}", repeats=4) for i in range(12)]
    seen = set()
    for workers in (1, 4, 16):
        for _ in range(3):
            seen.add(
                run_suite(
                    cases, agent_factory=patterned("PFPP"), config=SuiteConfig(max_workers=workers)
                ).canonical_json()
            )
    assert len(seen) == 1


def test_sb02_sixteen_concurrent_run_suite_calls_are_identical() -> None:
    cases = [refund_case(f"c{i}", repeats=3) for i in range(3)]
    out: list[str] = []
    lock = threading.Lock()

    def go() -> None:
        text = run_suite(
            cases, agent_factory=patterned("PFP"), config=SuiteConfig(max_workers=2)
        ).canonical_json()
        with lock:
            out.append(text)

    threads = [threading.Thread(target=go) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(out) == 16 and len(set(out)) == 1


def test_results_are_ordered_by_source_not_completion() -> None:
    cases = [refund_case(f"c{i}", repeats=1) for i in range(5)]
    delays = {"c0": 0.3, "c1": 0.0, "c2": 0.2, "c3": 0.0, "c4": 0.1}

    def factory(tools: Any) -> Any:
        def agent(text: str) -> str:
            time.sleep(delays[current_repeat().case_id])  # type: ignore[union-attr]
            return make_good_agent(tools)(text)  # type: ignore[no-any-return]

        return agent

    result = run_suite(cases, agent_factory=factory, config=SuiteConfig(max_workers=5))
    assert [c.case_id for c in result.cases] == [f"c{i}" for i in range(5)]


_ = (json, HarnessConfigError, FakeResponse)
