"""``build_test_orchestrator`` and ``Orchestrator``: run one agent invocation, offline, and return
a frozen ``RunResult`` instead of raising for agent failures.

Execution model (plan 02, sections 3b and 10):
* every run gets its **own daemon thread**; a sync agent runs there with no event loop running (so
  agents that call ``asyncio.run`` internally keep working) and an async agent runs under one
  ``asyncio.run`` (factory and all turns share a loop);
* the run's deadline is enforced in-loop with a task and ``asyncio.wait`` (works on Python 3.10;
  ``asyncio.timeout`` and ``Task.cancelling`` are 3.11+), and a sync agent that ignores the
  deadline is abandoned (daemon thread, ``W_ZOMBIE``) with its later fake calls refused;
* ``arun`` runs an async agent on the caller's loop; a sync agent's turns run on daemon threads.
"""

import asyncio
import concurrent.futures
import contextvars
import inspect
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from typing_extensions import Unpack

from ..models import (
    ErrorInfo,
    FakeSpec,
    Message,
    RunResult,
    ToolInfo,
    Violation,
)
from .adapters.base import Adapter
from .adapters.plain import BoundEntry, PlainAdapter, is_async_callable
from .config import (
    AgentInput,
    OrchestratorConfig,
    OrchestratorOptions,
    RunOptions,
    resolve_config,
)
from .errors import (
    E_HANDLER_DEFAULT,
    E_HANDLER_UNKNOWN,
    W_NO_MODEL_CAPTURE,
    W_ZOMBIE,
    HarnessConfigError,
)
from .extract import Extracted, extract_answer
from .fake_llm import FakeLLM, FakeLLMClient
from .fakes import RunFakes, build_run_fakes, normalize_fakes
from .injection import InjectionPlan, plan_injection
from .recorder import Listener, Recorder, _current_recorder
from .registry import adapters

GRACE_S = 2.0  # extra time an async run gets to finish cancelling after its deadline
TRACEBACK_CHARS = 8000


# --------------------------------------------------------------------------- per-run state


@dataclass
class _Box:
    """What the worker learned; written only by the worker (or the guard), read after the run."""

    status: str = "finished"
    error: ErrorInfo | None = None
    base_exc: BaseException | None = None
    config_exc: HarnessConfigError | None = None
    outputs: list[Extracted] = field(default_factory=list)
    setup_ms: float = 0.0
    async_deadline_active: bool = False

    def fail(self, status: str, exc: Exception, traceback_limit: int) -> None:
        self.status = status
        self.error = error_info(exc, traceback_limit)


def error_info(exc: BaseException, traceback_limit: int) -> ErrorInfo:
    text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, limit=-traceback_limit)
    )
    if len(text) > TRACEBACK_CHARS:
        marker = "...(truncated)\n"
        text = marker + text[-(TRACEBACK_CHARS - len(marker)) :]
    return ErrorInfo(
        type(exc).__qualname__,
        type(exc).__module__,
        f"{type(exc).__name__}: {exc}",
        traceback=text,
    )


@dataclass
class _Run:
    trace_id: str
    started_at: str
    rec: Recorder
    fakes: RunFakes
    inputs: list[AgentInput]
    input_original: "str | tuple[Message, ...]"
    follow_ups: tuple[str, ...]
    timeout_s: float | None
    deadline: float | None
    extra: Any
    box: _Box = field(default_factory=_Box)

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())


def _normalize_input(
    input: object, follow_ups: object, extra: Any
) -> tuple[list[AgentInput], "str | tuple[Message, ...]", tuple[str, ...]]:
    if isinstance(input, str):
        if not input.strip():
            raise HarnessConfigError("run() input must not be empty")
        original: str | tuple[Message, ...] = input
        first = AgentInput(input, messages=(Message("user", input),), turn=0, extra=extra)
    elif isinstance(input, (list, tuple)) and input and all(isinstance(m, Message) for m in input):
        messages = tuple(input)
        if messages[-1].role != "user":
            raise HarnessConfigError("the last input message must be from the user")
        original = messages
        first = AgentInput(messages[-1].content, messages=messages, turn=0, extra=extra)
    else:
        raise HarnessConfigError("run() input must be a string or a non-empty tuple of Message")
    if isinstance(follow_ups, str) or not isinstance(follow_ups, (list, tuple)):
        raise HarnessConfigError("follow_ups must be a tuple of strings")
    turns = [first]
    for number, text in enumerate(follow_ups, start=1):
        if not isinstance(text, str) or not text.strip():
            raise HarnessConfigError("follow_ups must be non-empty strings")
        turns.append(AgentInput(text, messages=(Message("user", text),), turn=number, extra=extra))
    return turns, original, tuple(follow_ups)


# --------------------------------------------------------------------------- helpers


async def _in_daemon_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a sync callable on its own daemon thread (no loop running there) and await it.

    Never the default executor: its threads are joined at interpreter exit, so a hung agent
    would hang pytest. The caller's context is copied so contextvars propagate.
    """
    future: concurrent.futures.Future[Any] = concurrent.futures.Future()
    context = contextvars.copy_context()

    def target() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = context.run(fn, *args)
        except BaseException as exc:
            _settle(future.set_exception, exc)
        else:
            _settle(future.set_result, result)

    threading.Thread(target=target, daemon=True, name="disseqt-sync-turn").start()
    return await asyncio.wrap_future(future)


def _settle(setter: Callable[[Any], None], value: Any) -> None:
    try:
        setter(value)
    except concurrent.futures.InvalidStateError:  # cancelled while we were running: abandoned
        pass


def _drain_sync(raw: Any) -> Any:
    """Consume a (sync) generator keeping its last item; anything else is returned as is."""
    if inspect.isgenerator(raw):
        last = None
        for item in raw:
            last = item
        return last
    return raw


async def _drain_async(raw: Any) -> Any:
    if inspect.isasyncgen(raw):
        last = None
        async for item in raw:
            last = item
        return last
    return raw


async def _await_value(value: Any) -> Any:
    return await value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- the orchestrator


class Orchestrator:
    """Holds only immutable compiled configuration; all per-run state is created per run, so
    repeats and parallel runs are independent trials."""

    __test__ = False  # never collected by pytest, even when imported as TestOrchestrator

    def __init__(
        self,
        factory: Callable[..., Any],
        specs: tuple[FakeSpec, ...],
        config: OrchestratorConfig,
        plan: InjectionPlan,
    ):
        self._factory = factory
        self._specs = specs
        self._config = config
        self._plan = plan
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._async_factory = is_async_callable(factory)

    @property
    def fakes(self) -> tuple[FakeSpec, ...]:
        return self._specs

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    def add_listener(self, fn: Listener) -> None:
        """Register a step listener; it applies to every future run."""
        with self._lock:
            self._listeners.append(fn)

    # ------------------------------------------------------------------ run (sync)

    def run(
        self,
        input: "str | tuple[Message, ...]",
        *,
        follow_ups: tuple[str, ...] = (),
        options: RunOptions | None = None,
    ) -> RunResult:
        """Run the agent once and return a ``RunResult``. Never raises for agent failures; raises
        ``HarnessConfigError`` for a broken test and re-raises ``KeyboardInterrupt`` and friends."""
        run = self._prepare(input, follow_ups, options)
        context = contextvars.copy_context()
        context.run(_current_recorder.set, run.rec)
        thread = threading.Thread(
            target=context.run,
            args=(self._thread_main, run),
            daemon=True,
            name=f"disseqt-run-{run.trace_id[:8]}",
        )
        thread.start()
        self._join(thread, run)
        if thread.is_alive() and run.box.async_deadline_active:
            thread.join(GRACE_S)  # an async run enforces its own deadline; give it time to cancel
        return self._finish(run, abandoned=thread.is_alive())

    @staticmethod
    def _join(thread: threading.Thread, run: _Run) -> None:
        """Wait for the worker until the deadline; wake early when the run was aborted."""
        while thread.is_alive():
            left = run.remaining()
            if left is not None and left <= 0.0:
                return
            thread.join(0.05 if left is None else min(left, 0.05))
            if run.rec.closed == "aborted":
                thread.join(0.5)  # the worker is unwinding from RunAbortedError
                return

    def _thread_main(self, run: _Run) -> None:
        box = run.box
        try:
            if self._async_factory:
                box.async_deadline_active = True
                asyncio.run(self._guarded(run, self._async_body(run)))
            else:
                self._sync_body(run)
        except HarnessConfigError as exc:
            box.config_exc = exc
        except BaseException as exc:  # KeyboardInterrupt, SystemExit, pytest outcomes
            box.base_exc = exc

    def _sync_body(self, run: _Run) -> None:
        box, limit = run.box, self._config.traceback_limit
        started = time.monotonic()
        try:
            product = self._factory(**self._plan.kwargs(run.fakes))
            if inspect.isawaitable(product):
                product = asyncio.run(_await_value(product))
        except HarnessConfigError:
            raise
        except Exception as exc:
            box.fail("setup_failed", exc, limit)
            return
        box.setup_ms = (time.monotonic() - started) * 1000.0
        adapter, entry = self._resolve_entry(product)
        if entry.is_async:  # one loop for all turns
            box.async_deadline_active = True
            asyncio.run(self._guarded(run, self._async_turns(run, adapter, entry)))
            return
        for turn, inp in enumerate(run.inputs):
            run.rec.turn = turn
            try:
                raw = entry.call(adapter.shape_input(inp))
                if inspect.isawaitable(raw):
                    raw = asyncio.run(_await_value(raw))
                raw = _drain_sync(raw)
                if inspect.isasyncgen(raw):
                    raw = asyncio.run(_drain_async(raw))
                output = extract_answer(raw, self._config.extract, adapter.extract)
            except HarnessConfigError:
                raise
            except Exception as exc:
                box.fail("crashed", exc, limit)
                return
            box.outputs.append(output)

    # ------------------------------------------------------------------ run (async)

    async def arun(
        self,
        input: "str | tuple[Message, ...]",
        *,
        follow_ups: tuple[str, ...] = (),
        options: RunOptions | None = None,
    ) -> RunResult:
        """Like ``run`` for async callers: an async agent runs on the caller's loop (a sync
        factory is called on the caller's thread); a sync agent's turns run on daemon threads."""
        run = self._prepare(input, follow_ups, options)
        token = _current_recorder.set(run.rec)
        try:
            try:
                await self._guarded(run, self._async_body(run))
            except HarnessConfigError as exc:
                run.box.config_exc = exc
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                run.box.base_exc = exc
        finally:
            _current_recorder.reset(token)
        return self._finish(run, abandoned=False)

    async def _async_body(self, run: _Run) -> None:
        box, limit = run.box, self._config.traceback_limit
        started = time.monotonic()
        try:
            product = self._factory(**self._plan.kwargs(run.fakes))
            if inspect.isawaitable(product):
                product = await product
        except HarnessConfigError:
            raise
        except Exception as exc:
            box.fail("setup_failed", exc, limit)
            return
        box.setup_ms = (time.monotonic() - started) * 1000.0
        adapter, entry = self._resolve_entry(product)
        await self._async_turns(run, adapter, entry)

    async def _async_turns(self, run: _Run, adapter: Adapter, entry: BoundEntry) -> None:
        box, limit = run.box, self._config.traceback_limit
        for turn, inp in enumerate(run.inputs):
            run.rec.turn = turn
            try:
                shaped = adapter.shape_input(inp)
                if entry.is_async:
                    raw = await entry.call(shaped)
                else:
                    raw = await _in_daemon_thread(entry.call, shaped)
                if inspect.isawaitable(raw):
                    raw = await raw
                raw = await _drain_async(_drain_sync(raw))
                output = extract_answer(raw, self._config.extract, adapter.extract)
            except HarnessConfigError:
                raise
            except Exception as exc:
                box.fail("crashed", exc, limit)
                return
            box.outputs.append(output)

    async def _guarded(self, run: _Run, body: Any) -> None:
        """Run ``body`` under the run's deadline.

        Our own deadline is detected from ``asyncio.wait`` returning with the task not done;
        external cancellation arrives as ``CancelledError`` raised at the ``await`` itself. The
        two never share a code path, so no ``Task.cancelling()`` (3.11+) is needed.
        """
        task = asyncio.ensure_future(body)
        try:
            done, _ = await asyncio.wait({task}, timeout=run.remaining())
        except asyncio.CancelledError:  # somebody else cancelled us: stop the agent, pass it on
            run.rec.close("cancelled")
            task.cancel()
            await self._settle(task)
            raise
        if task in done:
            task.result()  # re-raises HarnessConfigError / BaseException from the body
            if run.remaining() != 0.0:
                return
            # Done, but only after the deadline: the body blocked the loop, so the timeout could
            # not interrupt it. Same outcome as a sync agent that overruns.
        run.rec.close("timed_out")  # in-flight steps become "Abandoned: run timed out"
        if task not in done:
            task.cancel()
            await self._settle(task)

    @staticmethod
    async def _settle(task: "asyncio.Future[Any]") -> None:
        """Wait (bounded) for a cancelled task to finish; a task that swallows the cancellation
        is left running (documented limit: its later fake calls are refused)."""
        await asyncio.wait({task}, timeout=GRACE_S)
        if task.done() and not task.cancelled():
            task.exception()  # mark retrieved; the run is already closed

    # ------------------------------------------------------------------ shared pieces

    def _resolve_entry(self, product: Any) -> tuple[Adapter, BoundEntry]:
        plain = PlainAdapter()
        if self._config.entry is not None:
            return plain, plain.bind_entry(product, self._config.entry)
        for adapter in adapters():  # registered framework adapters first (phase 1b)
            if adapter.matches(product):
                return adapter, adapter.bind_entry(product, None)
        return plain, plain.bind_entry(product, None)

    def _prepare(self, input: object, follow_ups: object, options: RunOptions | None) -> _Run:
        options = options or RunOptions()
        inputs, original, follow = _normalize_input(input, follow_ups, options.extra)
        config = self._config
        clock = config.clock or time.monotonic_ns
        trace_id = (config.id_factory or (lambda: uuid.uuid4().hex))()
        with self._lock:
            listeners = tuple(self._listeners)
        rec = Recorder(clock, trace_id, listeners, config.max_recorded_steps)
        fakes = build_run_fakes(self._specs, config.handlers, rec, config.async_tools)
        if config.llm is not None:
            fakes.llm_client = FakeLLMClient(config.llm, rec)
        timeout = options.timeout_s if options.timeout_s is not None else config.timeout_s
        deadline = None if timeout is None else time.monotonic() + timeout
        rec.deadline = deadline
        return _Run(
            trace_id=trace_id,
            started_at=_utc_now(),
            rec=rec,
            fakes=fakes,
            inputs=inputs,
            input_original=original,
            follow_ups=follow,
            timeout_s=timeout,
            deadline=deadline,
            extra=options.extra,
        )

    def _finish(self, run: _Run, *, abandoned: bool) -> RunResult:
        box, rec = run.box, run.rec
        warnings: list[str] = []
        if rec.closed == "aborted":  # max_recorded_steps exceeded; closed by the recorder
            status = "aborted"
            if abandoned:
                warnings.append(W_ZOMBIE)
        elif abandoned:  # the worker ignored the deadline: abandon its thread
            rec.close("timed_out")
            status = "timed_out"
            warnings.append(W_ZOMBIE)
        elif box.base_exc is not None:
            rec.close("crashed")
            raise box.base_exc
        elif box.config_exc is not None:
            rec.close("crashed")
            raise box.config_exc
        else:
            # The first close is final: a run the deadline already closed stays timed out even
            # if the agent failed afterwards (for example in cleanup that calls a refused fake).
            rec.close(box.status)
            status = rec.closed or box.status
        outputs = list(box.outputs)
        finished = status == "finished" and len(outputs) == len(run.inputs)
        last = outputs[-1] if finished else None
        for output in outputs:
            warnings.extend(output.warnings)
        llm = run.fakes.llm_client
        if llm is None:
            warnings.append(W_NO_MODEL_CAPTURE)
        error = box.error if status in ("crashed", "setup_failed") else None
        agent_run = rec.build_run(
            input=run.input_original,
            follow_ups=run.follow_ups,
            final_answer=last.answer if last else None,
            final_text=last.text if last else None,
            turn_outputs=tuple(o.answer for o in outputs),
            messages=llm.last_exchange if llm else (),
            available_tools=tuple(
                ToolInfo(
                    s.name,
                    s.kind,
                    s.effect,
                    description=s.description,
                    schema=s.schema,
                )
                for s in self._specs
            ),
            model_steps_captured=llm is not None,
            started_at=run.started_at,
            usage=rec.usage_totals(),
            error=error.message if error else None,
            extra=run.extra,
        )
        violations = list(rec.violations)
        if self._config.assert_all_called and status == "finished":
            violations.extend(self._unused_violations(run))
        return RunResult(
            status,
            agent_run,
            error=error,
            violations=tuple(violations),
            warnings=tuple(warnings),
            trace_id=run.trace_id,
            duration_ms=rec.now_ms(),
            setup_ms=box.setup_ms,
        )

    @staticmethod
    def _unused_violations(run: _Run) -> list[Violation]:
        out: list[Violation] = []
        for name in sorted(run.fakes.states):
            for index, times in run.fakes.states[name].unused_rules():
                used = run.fakes.states[name].used[index]
                out.append(
                    Violation(
                        "unused_response",
                        f"fake '{name}' response #{index} declared times={times} "
                        f"but was used {used} times",
                        fake=name,
                    )
                )
        return out


# --------------------------------------------------------------------------- public builder


def build_test_orchestrator(
    agent_factory: Callable[..., Any],
    fakes: object = (),
    *,
    config: OrchestratorConfig | None = None,
    **options: Unpack[OrchestratorOptions],
) -> Orchestrator:
    """Validate the test setup and compile an ``Orchestrator``.

    ``fakes`` is a tuple of ``FakeSpec`` or a ``{name: FakeSpec | stub(...) | JSON value}``
    mapping. Options (``entry``, ``llm``, ``handlers``, ``timeout_s``, ``extract``,
    ``async_tools``, ``assert_all_called``, ...) are fields of ``OrchestratorConfig`` and may be
    passed as keywords. Every setup mistake raises ``HarnessConfigError`` here, before any run.
    """
    cfg = resolve_config(config, options)
    if not callable(agent_factory):
        raise HarnessConfigError(f"agent_factory must be callable, got {agent_factory!r}")
    specs = normalize_fakes(fakes)
    by_name = {spec.name: spec for spec in specs}
    for name in cfg.handlers:
        if name not in by_name:
            raise HarnessConfigError(E_HANDLER_UNKNOWN.format(name=name))
        if by_name[name].default is not None:
            raise HarnessConfigError(E_HANDLER_DEFAULT.format(name=name))
    for spec in specs:
        rules = [*spec.responses, *([spec.default] if spec.default else [])]
        for index, rule in enumerate(rules):
            if rule.error == "timeout":
                raise HarnessConfigError(
                    f"fake '{spec.name}' response #{index} uses the reserved error 'timeout'"
                )
    if cfg.llm is not None and not isinstance(cfg.llm, FakeLLM):
        raise HarnessConfigError(f"llm must be a FakeLLM, got {type(cfg.llm).__name__}")
    plan = plan_injection(agent_factory, specs, cfg.llm is not None)
    return Orchestrator(agent_factory, specs, cfg, plan)
