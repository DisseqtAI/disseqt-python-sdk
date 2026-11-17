"""Run-level behaviour: statuses, async, timeout, crash, determinism (T32-T35, T38-T47, T52, T57, T58)."""

import asyncio
import dataclasses
import threading
import time
from typing import Any

import pytest

from disseqt_agent_testing.harness import (
    HarnessConfigError,
    RunAbandonedError,
    build_test_orchestrator,
)
from disseqt_agent_testing.models import (
    FakeResponse,
    FakeSpec,
    Message,
    RunResult,
    from_dict,
    to_dict,
)
from tests.agent_testing._support.refund_agent import FAKES, make_async_refund_agent

from .helpers import normalized, refund, run_body, wrong_order


def test_acceptance_happy_wrong_order_and_ask() -> None:
    ok = refund().run("refund order A12")
    assert ok.status == "finished" and ok.violations == ()
    assert [s.name for s in ok.run.steps] == ["get_order", "check_policy", "issue_refund"]
    assert ok.run.final_text == "Refunded 59.99 for A12."
    assert dict(ok.run.steps[2].args) == {"order_id": "A12", "amount": 59.99}
    bad = wrong_order().run("refund order A12")
    assert dict(bad.run.steps[2].args)["order_id"] == "Z999"
    ask = refund().run("I want a refund")
    assert ask.run.steps == () and ask.run.final_text == "Which order number is this about?"


def test_t32_async_agent_via_run_and_via_arun_are_the_same() -> None:
    orch = build_test_orchestrator(make_async_refund_agent, FAKES, async_tools=True)
    a = orch.run("refund order A12")
    b = asyncio.run(orch.arun("refund order A12"))
    assert a.status == b.status == "finished" and normalized(a) == normalized(b)
    assert normalized(a) == normalized(refund().run("refund order A12"))


def test_t33_async_factory_and_agent_share_one_loop() -> None:
    loops: list[Any] = []

    async def factory(tools: Any) -> Any:
        loops.append(asyncio.get_running_loop())

        async def agent(text: str) -> str:
            loops.append(asyncio.get_running_loop())
            return "ok"

        return agent

    r = build_test_orchestrator(factory, {"t": {}}).run("go")
    assert r.status == "finished" and loops[0] is loops[1]


def test_t34_sync_agent_that_calls_asyncio_run_works_in_run_and_arun() -> None:
    def factory(tools: Any) -> Any:
        async def inner() -> str:
            return "inner"

        return lambda t: asyncio.run(inner())

    orch = build_test_orchestrator(factory, {"t": {}})
    assert orch.run("go").run.final_text == "inner"
    assert asyncio.run(orch.arun("go")).run.final_text == "inner"


def test_t35_async_tools_are_coroutine_functions() -> None:
    import inspect

    seen: dict[str, Any] = {}

    async def factory(tools: Any) -> Any:
        seen["is_coro"] = inspect.iscoroutinefunction(tools["t"])

        async def agent(text: str) -> Any:
            return await tools["t"]()

        return agent

    r = build_test_orchestrator(factory, {"t": {"v": 1}}, async_tools=True).run("go")
    assert seen["is_coro"] is True and r.run.final_answer == {"v": 1}


def test_async_fake_called_from_sync_code_gets_the_value_not_a_hang() -> None:
    r = run_body({"t": {"v": 1}}, lambda tools: tools["t"](), async_tools=True)
    assert r.status == "finished" and r.run.final_answer == {"v": 1}


def test_t38_setup_failure() -> None:
    def factory(tools: Any) -> Any:
        raise RuntimeError("db missing")

    r = build_test_orchestrator(factory, {"t": {}}).run("go")
    assert r.status == "setup_failed" and r.error is not None
    assert r.error.message == "RuntimeError: db missing" and r.run.steps == ()
    assert r.setup_ms >= 0


def test_t39_crash_keeps_the_steps_made_before_it() -> None:
    def body(tools: Any) -> None:
        tools["t"](n=1)
        tools["t"](n=2)
        raise ValueError("boom")

    r = run_body({"t": {}}, body, traceback_limit=2)
    assert r.status == "crashed" and len(r.run.steps) == 2
    assert r.error is not None and r.error.message == "ValueError: boom"
    assert "ValueError: boom" in (r.error.traceback or "")
    assert r.run.final_answer is None and r.run.error == "ValueError: boom"


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_t40_base_exceptions_propagate(exc: type[BaseException]) -> None:
    with pytest.raises(exc):
        run_body({"t": {}}, lambda tools: (_ for _ in ()).throw(exc()))


def test_t41_pytest_outcomes_are_not_agent_crashes() -> None:
    def handler(**_: Any) -> None:
        pytest.fail("from handler")

    with pytest.raises(pytest.fail.Exception, match="from handler"):
        run_body((FakeSpec("t"),), lambda tools: tools["t"](), handlers={"t": handler})


def test_t42_a_sync_agent_that_never_returns_is_abandoned() -> None:
    stop = threading.Event()
    late: list[BaseException] = []

    def body(tools: Any) -> None:
        while not stop.is_set():
            try:
                tools["t"]()
            except RunAbandonedError as exc:
                late.append(exc)
                return
            time.sleep(0.01)

    start = time.monotonic()
    r = run_body({"t": {}}, body, timeout_s=0.5)
    elapsed = time.monotonic() - start
    assert r.status == "timed_out" and elapsed < 2.0
    assert any("still running after timeout" in w for w in r.warnings)
    steps_at_return = len(r.run.steps)
    assert steps_at_return > 0
    time.sleep(0.2)  # the zombie keeps calling; it is refused and never recorded
    stop.set()
    assert late and "already ended (timed_out)" in str(late[0])
    assert len(r.run.steps) == steps_at_return


def test_t43_async_timeout_marks_the_in_flight_step_abandoned() -> None:
    spec = FakeSpec("t", default=FakeResponse(returns=1, latency_s=30))

    async def factory(tools: Any) -> Any:
        async def agent(text: str) -> Any:
            return await tools["t"]()

        return agent

    start = time.monotonic()
    r = build_test_orchestrator(factory, (spec,), async_tools=True, timeout_s=0.3).run("go")
    assert r.status == "timed_out" and time.monotonic() - start < 3
    (step,) = r.run.steps
    assert step.error == "Abandoned: run timed out" and step.duration_ms is None


def test_t45_stability_16_threads_x_50_runs() -> None:
    orch = refund()
    results: list[Any] = []
    lock = threading.Lock()

    def go() -> None:
        for _ in range(50):
            res = orch.run("refund order A12")
            with lock:
                results.append(res)

    threads = [threading.Thread(target=go) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(results) == 800
    first = normalized(results[0])
    assert all(normalized(r) == first for r in results)
    assert [s.call_id for s in results[0].run.steps] == ["call-1", "call-2", "call-3"]
    assert len({r.trace_id for r in results}) == 800


def test_t47_results_are_frozen_and_round_trip() -> None:
    r = refund().run("refund order A12")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.run.steps = ()  # type: ignore[misc]
    assert from_dict(RunResult, to_dict(r)) == r


def test_t52_follow_ups_run_as_further_turns() -> None:
    r = refund().run("I want a refund", follow_ups=("it is A12",))
    assert len(r.run.turn_outputs) == 2
    assert r.run.turn_outputs[0] == "Which order number is this about?"
    assert {s.turn for s in r.run.steps} == {1} and r.run.final_text == "Refunded 59.99 for A12."


def test_t57_one_orchestrator_two_runs_each_get_the_rule_once() -> None:
    spec = FakeSpec(
        "t",
        responses=(FakeResponse(returns="once", times=1),),
        default=FakeResponse(returns="after"),
    )
    seen: list[Any] = []
    orch = build_test_orchestrator(
        lambda tools: (lambda t: seen.append([tools["t"](), tools["t"]()])), (spec,)
    )
    orch.run("a")
    orch.run("b")
    assert seen == [["once", "after"], ["once", "after"]]


def test_t58_no_fake_llm_means_model_calls_are_not_captured() -> None:
    r = refund().run("refund order A12")
    assert r.run.model_steps_captured is False
    assert any(w.startswith("model calls not captured") for w in r.warnings)


def test_input_validation_is_a_config_error() -> None:
    orch = refund()
    for bad in ("   ", (), (Message("assistant", "hi"),), 5):
        with pytest.raises(HarnessConfigError):
            orch.run(bad)  # type: ignore[arg-type]
    with pytest.raises(HarnessConfigError):
        orch.run("x", follow_ups="again")  # type: ignore[arg-type]
    with pytest.raises(HarnessConfigError):
        orch.run("x", follow_ups=("",))


def test_message_history_input() -> None:
    seen: list[str] = []

    def factory(tools: Any) -> Any:
        return lambda t: seen.append(t) or t

    r = build_test_orchestrator(factory, {"t": {}}).run(
        (Message("user", "hi"), Message("assistant", "hello"), Message("user", "refund"))
    )
    assert seen == ["refund"] and r.run.input[-1].content == "refund"  # type: ignore[index]


def test_t66_a_clock_and_id_factory_make_runs_reproducible() -> None:
    ticks = iter(range(0, 10**9, 1_000_000))
    orch = refund(clock=lambda: next(ticks), id_factory=lambda: "fixed-trace-id")
    r = orch.run("refund order A12")
    assert r.trace_id == "fixed-trace-id" and r.status == "finished"


def test_t48_t49_t50_harness_half_of_the_refund_proof() -> None:
    """The assertion halves (assert_agent) arrive with plan 03; this pins what they will read."""
    ok = refund().run("refund order A12")
    assert [s.name for s in ok.run.steps] == ["get_order", "check_policy", "issue_refund"]
    # T48: the refund amount comes from the order (the data Ref will point at)
    assert dict(ok.run.steps[0].result)["total"] == dict(ok.run.steps[2].args)["amount"]
    # T49: the wrong-order bug is visible as expected "A12" vs actual "Z999" on issue_refund
    bad = wrong_order().run("refund order A12")
    assert dict(bad.run.steps[0].result)["order_id"] == "A12"
    assert dict(bad.run.steps[2].args)["order_id"] == "Z999"
    # T50: no order id -> no steps and a clarifying question
    ask = refund().run("I want a refund")
    assert ask.run.steps == () and "order number" in (ask.run.final_text or "")
