"""Context propagation and cancellation: T75-T78 (T79: this file runs on 3.10 and 3.13 in CI)."""

import asyncio
import contextvars
import threading
import time
from typing import Any

import pytest

from disseqt_agent_testing.harness import (
    RunAbandonedError,
    build_test_orchestrator,
    current_recorder,
)
from disseqt_agent_testing.models import FakeResponse, FakeSpec

VAR: contextvars.ContextVar[str] = contextvars.ContextVar("test_var", default="unset")


def test_t75_the_callers_contextvars_reach_every_execution_path() -> None:
    seen: dict[str, Any] = {}

    def handler(**_: Any) -> str:
        seen["handler"] = (VAR.get(), current_recorder() is not None)
        return "h"

    def sync_factory(tools: Any) -> Any:
        def agent(text: str) -> str:
            seen["sync"] = (VAR.get(), current_recorder() is not None)
            tools["t"]()
            return "ok"

        return agent

    async def async_factory(tools: Any) -> Any:
        async def agent(text: str) -> str:
            seen["async"] = (VAR.get(), current_recorder() is not None)

            def blocking() -> tuple[str, bool]:
                return VAR.get(), current_recorder() is not None

            seen["to_thread"] = await asyncio.to_thread(blocking)
            return "ok"

        return agent

    spec = (FakeSpec("t"),)
    VAR.set("from-test")
    build_test_orchestrator(sync_factory, spec, handlers={"t": handler}).run("go")
    build_test_orchestrator(async_factory, spec).run("go")

    async def via_arun() -> None:
        VAR.set("from-arun")
        await build_test_orchestrator(async_factory, spec).arun("go")

    asyncio.run(via_arun())
    assert seen["sync"] == ("from-test", True) and seen["handler"] == ("from-test", True)
    assert seen["async"][1] is True and seen["to_thread"][1] is True
    assert seen["to_thread"][0] == seen["async"][0] == "from-arun"


def test_t76_concurrent_runs_do_not_leak_context() -> None:
    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def factory(tools: Any) -> Any:
        def agent(text: str) -> str:
            barrier.wait(5)  # both runs are inside their agents at the same time
            time.sleep(0.05)
            return VAR.get()

        return agent

    orch = build_test_orchestrator(factory, {"t": {}})

    def go(value: str) -> None:
        VAR.set(value)
        results[value] = orch.run("go").run.final_text or ""

    threads = [threading.Thread(target=go, args=(v,)) for v in ("one", "two")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == {"one": "one", "two": "two"}


def test_t77_cancelling_arun_cancels_the_agent_and_is_not_a_timeout() -> None:
    state: dict[str, Any] = {}

    async def factory(tools: Any) -> Any:
        async def agent(text: str) -> str:
            state["started"] = True
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            return "late"

        return agent

    orch = build_test_orchestrator(factory, {"t": {}}, timeout_s=60)

    async def scenario() -> None:
        task = asyncio.ensure_future(orch.arun("go"))
        while not state.get("started"):
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert others == []  # no orphan agent task

    asyncio.run(scenario())
    assert state["cancelled"] is True


def test_t78_an_agent_that_swallows_cancellation_is_abandoned_not_hung() -> None:
    late: list[BaseException] = []
    spec = FakeSpec("t", default=FakeResponse(returns=1))
    release = threading.Event()

    async def factory(tools: Any) -> Any:
        async def agent(text: str) -> str:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                pass  # swallow once
            while not release.is_set():
                try:
                    tools["t"]()
                except RunAbandonedError as exc:
                    late.append(exc)
                    break
                await asyncio.sleep(0.01)
            return "late"

        return agent

    start = time.monotonic()
    r = build_test_orchestrator(factory, (spec,), timeout_s=0.3).run("go")
    release.set()
    assert r.status == "timed_out" and time.monotonic() - start < 5
    assert len(late) == 1 and "already ended (timed_out)" in str(late[0])
    assert r.run.steps == ()  # the refused late call was never recorded


def test_timeouts_never_leave_the_deadline_state_behind() -> None:
    async def slow(text: str) -> str:
        await asyncio.sleep(30)
        return "x"

    orch = build_test_orchestrator(lambda tools: slow, {"t": {}}, timeout_s=0.2)
    assert orch.run("go").status == "timed_out"
    assert orch.run("go").status == "timed_out"  # reusable afterwards


def test_arun_with_a_sync_agent_keeps_the_callers_context_on_its_turn_thread() -> None:
    seen: dict[str, Any] = {}

    def factory(tools: Any) -> Any:
        def agent(text: str) -> str:
            seen["value"] = VAR.get()
            seen["recorder"] = current_recorder() is not None
            seen["thread"] = threading.current_thread().daemon
            tools["t"]()
            return "ok"

        return agent

    async def scenario() -> Any:
        VAR.set("from-async-caller")
        return await build_test_orchestrator(factory, {"t": {}}).arun("go")

    r = asyncio.run(scenario())
    assert r.status == "finished" and len(r.run.steps) == 1
    assert seen == {"value": "from-async-caller", "recorder": True, "thread": True}
