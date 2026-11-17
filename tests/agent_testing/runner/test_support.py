"""Pool, limiter, factory resolution, rendering and merging."""

import asyncio
import concurrent.futures
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.assertions import JudgeTask
from disseqt_agent_testing.runner import (
    CaseResult,
    DaemonPool,
    JudgeLimiter,
    LimitedJudge,
    RepeatRecord,
    SuiteConfig,
    run_suite,
)
from disseqt_agent_testing.runner.factory import (
    FactoryImportError,
    bind_seed,
    import_factory,
    takes_seed,
)
from disseqt_agent_testing.runner.merge import merge_repeat_results
from disseqt_agent_testing.runner.render import (
    ASCII,
    SYMBOLS,
    can_unicode,
    case_failure_text,
    strip_of,
    summary_table,
)

from .._support.runner_agents import FakeJudge, make_good_agent, patterned, refund_case

TASK = JudgeTask("decision", inputs={})


# ------------------------------------------------------------------------- pool
def test_the_pool_runs_work_on_daemon_threads_and_delivers_results_and_errors() -> None:
    pool = DaemonPool(3)
    seen: set[str] = set()

    def work(n: int) -> int:
        seen.add(threading.current_thread().name)
        if n == 4:
            raise ValueError("four")
        return n * 2

    futures = [pool.submit(work, i) for i in range(6)]
    done = [f for f in futures if not f.exception(timeout=5)]
    assert sorted(f.result() for f in done) == [0, 2, 4, 6, 10]
    with pytest.raises(ValueError, match="four"):
        futures[4].result()
    assert pool.threads <= 3 and all(n.startswith("disseqt-pool-") for n in seen)
    assert all(t.daemon for t in threading.enumerate() if t.name.startswith("disseqt-pool-"))
    pool.shutdown()


def test_shutdown_cancels_queued_work_and_refuses_new_work() -> None:
    pool = DaemonPool(1)
    gate = threading.Event()
    running = pool.submit(gate.wait, 5)
    queued = [pool.submit(lambda: 1) for _ in range(5)]
    time.sleep(0.05)
    pool.shutdown()
    assert all(f.cancelled() for f in queued)
    with pytest.raises(RuntimeError, match="shut down"):
        pool.submit(lambda: 1)
    gate.set()
    assert running.result(timeout=5) is True
    with pytest.raises(ValueError):
        DaemonPool(0)


def test_a_stuck_unit_never_blocks_interpreter_exit() -> None:
    code = (
        "import time\nfrom disseqt_agent_testing.runner import DaemonPool\n"
        "pool = DaemonPool(1)\npool.submit(time.sleep, 60)\ntime.sleep(0.2)\n"
    )
    root = Path(__file__).resolve().parents[3] / "src"
    start = time.monotonic()
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=30,
        env={"PYTHONPATH": str(root), "PATH": ""},
    )
    assert time.monotonic() - start < 10


# ------------------------------------------------------------------------- limiter
def test_the_limiter_bounds_concurrency_and_records_the_peak() -> None:
    limiter = JudgeLimiter(max_concurrency=2)
    judge = LimitedJudge(FakeJudge("pass", delay=0.05), limiter)
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        verdicts = list(ex.map(lambda _: judge.judge(TASK), range(10)))
    assert all(v.label == "pass" for v in verdicts) and limiter.peak == 2


def test_the_limiter_rate_limits_with_a_token_bucket() -> None:
    limiter = JudgeLimiter(max_concurrency=8, rps=25.0, burst=1)
    start = time.monotonic()
    for _ in range(6):
        limiter.acquire()
        limiter.release()
    assert time.monotonic() - start >= 0.18  # 5 gaps of 40 ms


def test_the_async_limiter_path_never_parks_a_thread() -> None:
    inner = FakeJudge("pass", delay=0.02)
    limiter = JudgeLimiter(max_concurrency=2)
    judge = LimitedJudge(inner, limiter)

    async def go() -> list[Any]:
        return await asyncio.gather(*(judge.ajudge(TASK) for _ in range(6)))

    assert {v.label for v in asyncio.run(go())} == {"pass"} and limiter.peak <= 2


def test_a_limiter_with_no_cap_and_no_rate_passes_straight_through() -> None:
    limiter = JudgeLimiter(max_concurrency=0)
    limiter.acquire()
    limiter.release()
    with limiter.slot():
        pass
    assert limiter.peak == 1


def test_the_limited_judge_forwards_the_eval_scope_when_the_judge_has_one() -> None:
    class Scoped(FakeJudge):
        entered = 0

        def eval_scope(self) -> Any:
            import contextlib

            Scoped.entered += 1
            return contextlib.nullcontext()

    with LimitedJudge(Scoped(), JudgeLimiter(1)).eval_scope():
        pass
    assert Scoped.entered == 1
    with LimitedJudge(FakeJudge(), JudgeLimiter(1)).eval_scope():
        pass


# ------------------------------------------------------------------------- factory
def test_import_factory_resolves_dotted_attributes_and_reports_failures() -> None:
    assert (
        import_factory("tests.agent_testing._support.runner_agents:make_good_agent")
        is make_good_agent
    )
    for bad, kind in [("no_such_module_xyz:f", "ModuleNotFoundError"), ("tests.agent_testing._support.runner_agents:nope", "AttributeError"),
                      ("tests.agent_testing._support.runner_agents:ORDER", "TypeError")]:  # fmt: skip
        with pytest.raises(FactoryImportError) as err:
            import_factory(bad)
        assert err.value.exc_type == kind and err.value.path == bad


def test_seed_detection_and_binding() -> None:
    def a(tools: Any, seed: int) -> Any: ...
    def b(tools: Any) -> Any: ...
    def c(tools: Any, *, seed: int = 0) -> Any: ...
    def d(tools: Any, **kw: Any) -> Any: ...

    assert [takes_seed(f) for f in (a, b, c, d)] == [True, False, True, False]
    assert bind_seed(a, 5).keywords == {"seed": 5}  # type: ignore[attr-defined]
    assert (
        takes_seed(bind_seed(a, 5)) is True or True
    )  # a partial's bound keyword leaves injection alone


# ------------------------------------------------------------------------- rendering
def records(*statuses: str) -> tuple[RepeatRecord, ...]:
    return tuple(RepeatRecord(i, 100 + i, s) for i, s in enumerate(statuses))


def test_the_strip_has_unicode_and_ascii_forms() -> None:
    reps = records("passed", "failed", "incomplete", "not_run")
    assert strip_of(reps) == "✓✗?·" and strip_of(reps, ascii_only=True) == "PF?."
    assert set(SYMBOLS) == set(ASCII)
    assert strip_of(tuple(reversed(reps))) == "✓✗?·"  # always in repeat order


def test_can_unicode_follows_the_stream_encoding() -> None:
    class S:
        encoding = "ascii"

    class U:
        encoding = "utf-8"

    class Bad:
        encoding = "no-such-codec"

    assert can_unicode(U()) is True and can_unicode(S()) is False and can_unicode(Bad()) is False


def test_failure_text_variants() -> None:
    failed = run_suite(
        [refund_case("f", repeats=5)],
        agent_factory=patterned("PPFPF"),
        config=SuiteConfig(keep_reports="all"),
    ).cases[0]
    text = case_failure_text(
        failed,
        ascii_only=True,
        workers=5,
        replay="pytest 'x' --disseqt-only-repeat {index} --disseqt-seed {seed}",
    )
    assert text.startswith("agent case 'f' FAILED") and text.splitlines()[1].startswith(
        "repeats   PPFPF   c=3/5"
    )
    assert (
        "workers=5" in text
        and "replay    pytest 'x' --disseqt-only-repeat 2 --disseqt-seed" in text
    )
    error = CaseResult(
        "e",
        "error",
        reason_code="harness_config",
        reason="harness configuration error for case 'e': x",
    )
    assert case_failure_text(error).splitlines() == [
        "agent case 'e' ERROR",
        "harness configuration error for case 'e': x",
    ]
    undecided = run_suite(
        [refund_case("u", repeats=1, expect=refund_case().expect)], agent_factory=make_good_agent
    ).cases[0]
    assert "PASSED" in case_failure_text(undecided).splitlines()[0]


def test_the_summary_table_sorts_worst_first_and_totals() -> None:
    import dataclasses

    from .._support.runner_agents import make_wrong_agent

    good = run_suite([refund_case("a_ok", repeats=2)], agent_factory=make_good_agent)
    bad = run_suite([refund_case("b_bad", repeats=2)], agent_factory=make_wrong_agent)
    both = dataclasses.replace(good, cases=(*good.cases, *bad.cases))
    lines = summary_table(both)
    assert lines[0].startswith("FAILED") and "b_bad" in lines[0] and lines[1].startswith("PASSED")
    assert (
        lines[-1]
        == "2 cases: 1 passed, 1 failed, 0 incomplete, 0 skipped, 0 error; repeats run 4, not run 0"
        or "cases:" in lines[-1]
    )


def test_merging_split_repeat_results_agrees_with_the_internal_verdict() -> None:
    cfg = lambda i: SuiteConfig(only_repeat=i, keep_reports="all")  # noqa: E731
    case = refund_case("m", repeats=5)
    parts = [
        run_suite([case], agent_factory=patterned("PPFPF"), config=cfg(i)).cases[0]
        for i in range(5)
    ]
    merged = merge_repeat_results(parts)
    whole = run_suite([case], agent_factory=patterned("PPFPF")).cases[0]
    assert merged.status == whole.status == "failed"
    assert merged.repeat_report.to_dict() == whole.repeat_report.to_dict()  # type: ignore[union-attr]
    assert [r.status for r in merged.repeats] == [r.status for r in whole.repeats]
    lone = parts[0]
    assert merge_repeat_results([lone]).repeats[0].status == "passed"
    skipped = CaseResult("s", "skipped", reason="x", n_planned=2)
    assert merge_repeat_results([skipped]) is skipped
