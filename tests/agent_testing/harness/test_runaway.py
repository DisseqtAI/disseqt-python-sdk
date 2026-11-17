"""Runaway agents, schema-less handlers, latency clamp (fix pass)."""

import threading
import time

import pytest

from disseqt_agent_testing.harness import build_test_orchestrator
from disseqt_agent_testing.models import FakeResponse, FakeSpec, RunResult
from tests.agent_testing.harness.helpers import run_body


def spin(tools):
    while True:
        tools["ping"]()


def swallow(tools):
    n = 0
    while True:
        try:
            tools["ping"]()
        except Exception:
            n += 1


class TestStepsCap:
    def test_tool_loop_is_aborted_at_the_cap(self):
        result = run_body({"ping": 1}, spin, max_recorded_steps=50, timeout_s=10)
        assert result.status == "aborted"
        assert len(result.run.steps) == 50
        assert [v.code for v in result.violations] == ["steps_cap_exceeded"]
        assert result.error is None

    def test_broad_except_cannot_swallow_the_abort(self):
        t0 = time.monotonic()
        result = run_body({"ping": 1}, swallow, max_recorded_steps=50, timeout_s=10)
        assert result.status == "aborted" and time.monotonic() - t0 < 5
        assert len(result.run.steps) == 50
        assert not result.warnings or all("still running" not in w for w in result.warnings)

    def test_default_cap_is_10_000(self):
        result = run_body({"ping": 1}, spin, timeout_s=30)
        assert result.status == "aborted" and len(result.run.steps) == 10_000

    def test_async_agent_is_aborted_too(self):
        def factory(tools):
            async def agent(text):
                while True:
                    await tools["ping"]()

            return agent

        result = build_test_orchestrator(
            factory, {"ping": 1}, max_recorded_steps=20, async_tools=True, timeout_s=10
        ).run("go")
        assert result.status == "aborted" and len(result.run.steps) == 20

    def test_run_below_cap_is_unaffected(self):
        result = run_body({"ping": 1}, lambda tools: tools["ping"](), max_recorded_steps=1)
        assert result.status == "finished" and not result.violations

    def test_cap_validated(self):
        for bad in (0, -1, True, 1.5):
            with pytest.raises(Exception, match="max_recorded_steps"):
                build_test_orchestrator(lambda: (lambda t: t), max_recorded_steps=bad)

    def test_finish_builds_the_run_once(self, monkeypatch):
        from disseqt_agent_testing.harness import recorder

        calls = []
        original = recorder.Recorder.build_run

        def counting(self, **fields):
            calls.append(1)
            return original(self, **fields)

        monkeypatch.setattr(recorder.Recorder, "build_run", counting)
        run_body({"ping": 1}, lambda tools: tools["ping"]())
        assert len(calls) == 1


class TestSchemaLessHandlers:
    def test_positional_and_keyword_calls_reach_the_handler(self):
        seen = []

        def lookup(order_id, note="n"):
            seen.append((order_id, note))
            return order_id.lower()

        out: list = []

        def body(tools):
            out.append(tools["lookup"]("A1"))
            out.append(tools["lookup"](order_id="B2", note="x"))

        result = run_body((FakeSpec("lookup", effect="read"),), body, handlers={"lookup": lookup})
        assert result.status == "finished", result.error
        assert out == ["a1", "b2"] and seen == [("A1", "n"), ("B2", "x")]
        assert result.run.steps[0].args["arg0"] == "A1"  # recording form unchanged

    def test_schema_binding_unchanged(self):
        spec = FakeSpec(
            "lookup",
            effect="read",
            schema={"properties": {"order_id": {}}, "required": ["order_id"]},
        )
        result = run_body(
            (spec,),
            lambda tools: tools["lookup"]("A1"),
            handlers={"lookup": lambda order_id: order_id},
        )
        assert result.status == "finished" and dict(result.run.steps[0].args) == {"order_id": "A1"}


class TestLatencyClamp:
    def test_sleep_is_clamped_to_the_deadline(self):
        spec = FakeSpec("slow", default=FakeResponse(returns=1, latency_s=3000))
        t0 = time.monotonic()
        result = build_test_orchestrator(
            lambda tools: (lambda t: tools["slow"]()), (spec,), timeout_s=0.3
        ).run("go")
        assert isinstance(result, RunResult) and result.status == "timed_out"
        assert time.monotonic() - t0 < 5
        assert result.run.steps[0].error == "Abandoned: run timed out"

    def test_sleeping_worker_is_released_when_the_run_ends(self):
        before = set(threading.enumerate())
        spec = FakeSpec("slow", default=FakeResponse(returns=1, latency_s=3000))
        build_test_orchestrator(
            lambda tools: (lambda t: tools["slow"]()), (spec,), timeout_s=0.2
        ).run("go")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(
            t.is_alive() for t in set(threading.enumerate()) - before
        ):
            time.sleep(0.05)
        assert not [t for t in set(threading.enumerate()) - before if t.is_alive()]
