"""The recorder (task 3): ids, ordering, close semantics, listeners, threads, routing."""

import threading
from typing import Any

import pytest

from disseqt_agent_testing.harness import (
    RunAbandonedError,
    build_test_orchestrator,
    current_recorder,
)
from disseqt_agent_testing.harness.recorder import Recorder
from disseqt_agent_testing.models import StepEvent

from .helpers import refund


class Clock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += int(ms * 1e6)


def test_ids_follow_start_order_even_if_a_later_call_finishes_first() -> None:
    clock = Clock()
    rec = Recorder(clock, "trace-1")
    a = rec.start("tool", "slow", {}, "write")
    b = rec.start("tool", "fast", {}, "read")
    clock.advance_ms(5)
    rec.finish(b, result=1)
    clock.advance_ms(20)
    rec.finish(a, result=2)
    rec.close("finished")
    steps = rec.build_run().steps
    assert [(s.seq, s.call_id, s.name) for s in steps] == [
        (1, "call-1", "slow"),
        (2, "call-2", "fast"),
    ]
    assert steps[0].duration_ms == 25 and steps[1].duration_ms == 5


def test_close_finalizes_open_steps_as_abandoned() -> None:
    rec = Recorder(Clock(), "t")
    rec.start("tool", "x", {}, "write")
    rec.close("timed_out")
    (step,) = rec.build_run().steps
    assert step.error == "Abandoned: run timed out" and step.duration_ms is None
    other = Recorder(Clock(), "t")
    other.start("tool", "x", {}, "write")
    other.close("crashed")
    assert other.build_run().steps[0].error == "Abandoned: run ended"


def test_close_is_idempotent_and_blocks_new_steps() -> None:
    rec = Recorder(Clock(), "0123456789abcdef")
    rec.close("finished")
    rec.close("crashed")
    assert rec.closed == "finished"
    with pytest.raises(RunAbandonedError) as err:
        rec.start("tool", "late", {}, None)
    assert str(err.value) == (
        "run 01234567... already ended (finished); fake 'late' was called after the run closed"
    )


def test_late_finish_and_late_violation_are_dropped() -> None:
    rec = Recorder(Clock(), "t")
    pending = rec.start("tool", "x", {}, "write")
    rec.close("timed_out")
    rec.finish(pending, result=1)
    rec.violation("unexpected_call", "late")
    assert rec.build_run().steps[0].result is None and rec.violations == ()


def test_arguments_are_captured_frozen_and_lossy_flagged() -> None:
    rec = Recorder(Clock(), "t")
    rec.start("tool", "x", {"o": object(), "n": {"k": [1]}}, "write")
    rec.close("finished")
    (step,) = rec.build_run().steps
    assert step.lossy is True
    with pytest.raises(TypeError):
        step.args["n"]["k2"] = 1  # type: ignore[index]


def test_time_never_goes_backwards() -> None:
    clock = Clock()
    rec = Recorder(clock, "t")
    clock.t = -10  # a hostile clock
    assert rec.now_ms() == 0.0


def test_t55_listeners_get_events_in_order_and_a_broken_one_is_disabled() -> None:
    events: list[StepEvent] = []

    def good(event: StepEvent) -> None:
        events.append(event)

    def bad(event: StepEvent) -> None:
        raise RuntimeError("listener bug")

    orch = refund()
    orch.add_listener(good)
    orch.add_listener(bad)
    r = orch.run("refund order A12")
    assert r.status == "finished"
    assert [(e.phase, e.seq) for e in events] == [
        ("start", 1), ("end", 1), ("start", 2), ("end", 2), ("start", 3), ("end", 3),
    ]  # fmt: skip
    assert [v.code for v in r.violations] == ["listener_error"]
    assert "RuntimeError: listener bug" in r.violations[0].message


def test_listener_is_registered_for_future_runs_only_and_per_run_state_is_fresh() -> None:
    orch = refund()
    seen: list[str] = []
    orch.run("refund order A12")
    orch.add_listener(lambda e: seen.append(e.phase))
    orch.run("refund order A12")
    assert len(seen) == 6


def test_t44_calls_from_internal_threads_get_unique_contiguous_ids() -> None:
    def body(tools: Any) -> None:
        def go() -> None:
            for _ in range(10):
                tools["t"]()

        threads = [threading.Thread(target=go) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]

    r = build_test_orchestrator(lambda tools: (lambda t: body(tools)), {"t": {"ok": 1}}).run("go")
    seqs = [s.seq for s in r.run.steps]
    assert seqs == list(range(1, 41)) and [s.call_id for s in r.run.steps] == [
        f"call-{n}" for n in seqs
    ]


def test_current_recorder_is_set_inside_the_run_and_unset_outside() -> None:
    seen: list[Any] = []

    def factory(tools: Any) -> Any:
        return lambda t: seen.append(current_recorder())

    build_test_orchestrator(factory, {"t": {}}).run("go")
    assert isinstance(seen[0], Recorder) and current_recorder() is None
