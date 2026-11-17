"""The ``Recorder``: the only mutable trajectory, private to one run.

Steps are numbered when they start (``call-<seq>``), so order reflects call order even when a long
call finishes after a later short one. ``build_run`` produces the one frozen ``AgentRun``.
"""

import contextvars
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import (
    AgentRun,
    FrozenDict,
    JSONValue,
    Step,
    StepEvent,
    Usage,
    Violation,
    capture_json,
)
from .errors import RunAbandonedError, RunAbortedError

Listener = Callable[[StepEvent], None]
_NONE: Any = object()

_current_recorder: "contextvars.ContextVar[Recorder | None]" = contextvars.ContextVar(
    "disseqt_agent_testing_recorder", default=None
)


def current_recorder() -> "Recorder | None":
    """The recorder of the run the current context belongs to (secondary routing route)."""
    return _current_recorder.get()


@dataclass(slots=True)
class _Pending:
    seq: int
    call_id: str
    kind: str
    name: str
    args: "FrozenDict[str, Any]"
    lossy: bool
    effect: str | None
    ts: float
    turn: int
    result: JSONValue = None
    error: str | None = None
    duration_ms: float | None = None
    usage: Usage | None = None


class _ListenerSlot:
    __slots__ = ("fn", "enabled")

    def __init__(self, fn: Listener):
        self.fn = fn
        self.enabled = True


class Recorder:
    def __init__(
        self,
        clock: Callable[[], int],
        trace_id: str,
        listeners: Sequence[Listener] = (),
        max_steps: int = 10_000,
    ):
        self._lock = threading.Lock()
        self._clock = clock
        self._t0 = clock()
        self._seq = 0
        self._open: dict[int, _Pending] = {}
        self._done: list[_Pending] = []
        self._closed: str | None = None
        self._closed_event = threading.Event()
        self._max_steps = max_steps
        self.deadline: float | None = None  # time.monotonic() value the run must end by
        self._violations: list[Violation] = []
        self._listeners = [_ListenerSlot(fn) for fn in listeners]
        self.trace_id = trace_id
        self.turn = 0

    # ------------------------------------------------------------------ time and state

    def now_ms(self) -> float:
        """Milliseconds since the run started (monotonic, never negative)."""
        return max(0.0, (self._clock() - self._t0) / 1e6)

    @property
    def closed(self) -> str | None:
        return self._closed

    @property
    def violations(self) -> tuple[Violation, ...]:
        with self._lock:
            return tuple(self._violations)

    def violation(
        self, code: str, message: str, *, fake: str | None = None, call_id: str | None = None
    ) -> None:
        with self._lock:
            if self._closed is None:
                self._violations.append(Violation(code, message, fake=fake, call_id=call_id))

    # ------------------------------------------------------------------ steps

    def start(self, kind: str, name: str, args: object, effect: str | None) -> _Pending:
        """Open a step. Raises ``RunAbandonedError`` if the run already ended."""
        captured, lossy = capture_json(args)
        assert isinstance(captured, FrozenDict)
        with self._lock:
            if self._closed == "aborted":
                raise RunAbortedError(self.trace_id, self._max_steps)
            if self._closed is not None:
                raise RunAbandonedError(self.trace_id, self._closed, name)
            if self._seq >= self._max_steps:
                self._violations.append(
                    Violation(
                        "steps_cap_exceeded",
                        f"run aborted: more than {self._max_steps} recorded steps "
                        f"(started '{name}' as step {self._seq + 1})",
                    )
                )
                self._close_locked("aborted")
                raise RunAbortedError(self.trace_id, self._max_steps)
            self._seq += 1
            pending = _Pending(
                seq=self._seq,
                call_id=f"call-{self._seq}",
                kind=kind,
                name=name,
                args=captured,
                lossy=lossy,
                effect=effect,
                ts=self.now_ms(),
                turn=self.turn,
            )
            self._open[pending.seq] = pending
        self._emit("start", pending)
        return pending

    def finish(
        self,
        pending: _Pending,
        result: Any = _NONE,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        """Close a step. A late finish after the run ended is dropped."""
        captured: JSONValue = None
        lossy = False
        if result is not _NONE:
            captured, lossy = capture_json(result)
        with self._lock:
            if self._closed is not None or pending.seq not in self._open:
                return
            pending.duration_ms = max(0.0, self.now_ms() - pending.ts)
            pending.result = captured
            pending.lossy = pending.lossy or lossy
            pending.error = error
            pending.usage = usage
            self._done.append(self._open.pop(pending.seq))
        self._emit("end", pending)

    def close(self, status: str) -> None:
        """End the run (idempotent). Steps still open are finalized as abandoned, not dropped."""
        with self._lock:
            self._close_locked(status)

    def _close_locked(self, status: str) -> None:
        if self._closed is not None:
            return
        self._closed = status
        note = "Abandoned: run timed out" if status == "timed_out" else "Abandoned: run ended"
        for pending in self._open.values():
            pending.error = note
            pending.duration_ms = None
            self._done.append(pending)
        self._open.clear()
        self._closed_event.set()

    def wait_closed(self, timeout: float) -> bool:
        """Block up to ``timeout`` seconds; True when the run closed meanwhile."""
        return self._closed_event.wait(max(0.0, timeout))

    def usage_totals(self) -> Usage:
        """Summed token usage of the finished steps (no run is built)."""
        with self._lock:
            done = list(self._done)
        return Usage(
            input_tokens=sum(p.usage.input_tokens for p in done if p.usage),
            output_tokens=sum(p.usage.output_tokens for p in done if p.usage),
            latency_s=self.now_ms() / 1000.0,
        )

    def build_run(self, **fields: Any) -> AgentRun:
        """Return the frozen ``AgentRun`` (call after ``close``)."""
        with self._lock:
            records = sorted(self._done, key=lambda p: p.seq)
        steps = tuple(
            Step(
                p.seq,
                p.kind,
                p.name,
                call_id=p.call_id,
                args=p.args,
                result=p.result,
                error=p.error,
                effect=p.effect,  # type: ignore[arg-type]
                ts=p.ts,
                duration_ms=p.duration_ms,
                turn=p.turn,
                usage=p.usage,
                lossy=p.lossy,
            )
            for p in records
        )
        return AgentRun(steps=steps, **fields)

    # ------------------------------------------------------------------ listeners

    def _emit(self, phase: str, pending: _Pending) -> None:
        if not self._listeners:
            return
        event = StepEvent(
            phase,
            self.trace_id,
            pending.seq,
            pending.call_id,
            pending.kind,
            pending.name,
            pending.turn,
            args=pending.args,
            result=pending.result if phase == "end" else None,
            error=pending.error if phase == "end" else None,
            duration_ms=pending.duration_ms if phase == "end" else None,
        )
        for slot in self._listeners:
            if not slot.enabled:
                continue
            try:
                slot.fn(event)
            except Exception as exc:  # a broken listener must never break the run
                slot.enabled = False
                self.violation(
                    "listener_error",
                    f"step listener raised {type(exc).__name__}: {exc}; "
                    "listener disabled for this run",
                )
