"""Fake tools and sub-agents: normalization, ``stub``/``spy``, signatures, dispatch.

A fake is delivered to the agent as a real function (``__name__``, ``__doc__`` and a typed
``__signature__`` built from the spec's JSON schema, so framework introspection such as
``function_tool`` works). Calls are bound, recorded, matched against ordered rules and answered.
"""

import asyncio
import inspect
import json
import keyword
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..models import (
    FakeResponse,
    FakeSpec,
    JSONValue,
    capture_json,
    thaw,
)
from ..models._serde import to_dict
from ._when import when_matches
from .errors import (
    E_ASYNC_HANDLER,
    E_DRAFT,
    E_DUP,
    E_KEY,
    FakeToolError,
    HarnessConfigError,
    RunAbandonedError,
    UnexpectedFakeCall,
)
from .recorder import Recorder
from .registry import kind_spec

_UNSET: Any = object()
_EXTRA_NAME = "__extra_kwargs__"
CUT_GRACE_S = 2.0  # how long a deadline-cut sleep waits for the run to be closed
_TYPES: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


@dataclass(slots=True, frozen=True)
class FakeDraft:
    """A nameless fake; it takes its name from the key of the ``fakes`` mapping."""

    returns: JSONValue = None
    effect: str = "write"
    kind: str = "tool"
    description: str = ""
    schema: Mapping[str, Any] | None = None


def stub(
    returns: JSONValue = None,
    *,
    effect: str = "write",
    kind: str = "tool",
    description: str = "",
    schema: Mapping[str, Any] | None = None,
) -> FakeDraft:
    """A fake that always returns ``returns``. ``effect`` defaults to ``"write"`` (the safe
    default: an unannotated state-changing tool must not pass ``extra_calls="allow_reads"``)."""
    return FakeDraft(returns, effect, kind, description, schema)


def spy(returns: JSONValue = None, *, kind: str = "tool") -> FakeDraft:
    """``stub(effect="read")``: a lookup whose output the test does not care about."""
    return stub(returns, effect="read", kind=kind)


def _spec_from_draft(name: str, draft: FakeDraft) -> FakeSpec:
    return FakeSpec(
        name,
        kind=draft.kind,  # type: ignore[arg-type]
        description=draft.description,
        effect=draft.effect,  # type: ignore[arg-type]
        schema=draft.schema,
        default=FakeResponse(returns=draft.returns),
    )


def normalize_fakes(fakes: object) -> tuple[FakeSpec, ...]:
    """Turn ``fakes`` (a tuple of specs or a name-keyed mapping) into specs sorted by name."""
    specs: list[FakeSpec] = []
    if isinstance(fakes, Mapping):
        for key, value in fakes.items():
            if not isinstance(key, str) or not key:
                raise HarnessConfigError(f"fakes keys must be non-empty strings, got {key!r}")
            if isinstance(value, FakeSpec):
                if value.name != key:
                    raise HarnessConfigError(E_KEY.format(key=key, name=value.name))
                specs.append(value)
            elif isinstance(value, FakeDraft):
                specs.append(_spec_from_draft(key, value))
            else:
                specs.append(_spec_from_draft(key, FakeDraft(returns=value)))
    elif isinstance(fakes, Iterable) and not isinstance(fakes, (str, bytes)):
        for value in fakes:
            if isinstance(value, FakeDraft):
                raise HarnessConfigError(E_DRAFT)
            if not isinstance(value, FakeSpec):
                raise HarnessConfigError(
                    f"fakes must contain FakeSpec objects, got {type(value).__name__}"
                )
            specs.append(value)
    else:
        raise HarnessConfigError(f"fakes must be a tuple of FakeSpec or a mapping, got {fakes!r}")
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise HarnessConfigError(E_DUP.format(name=spec.name))
        seen.add(spec.name)
        kind_spec(spec.kind)  # unknown kinds fail at build time
    return tuple(sorted(specs, key=lambda s: s.name))


# ------------------------------------------------------------------------ signatures and binding


def _schema_active(spec: FakeSpec) -> bool:
    schema = spec.schema
    return bool(schema) and bool(schema.get("properties") or schema.get("required"))  # type: ignore[union-attr]


def signature_for(spec: FakeSpec) -> inspect.Signature:
    """The signature the agent's framework sees: required properties first (schema order), then
    optional ones with a private default; ``**kwargs`` only if ``additionalProperties`` is true."""
    P = inspect.Parameter
    if not _schema_active(spec):
        return inspect.Signature([P("args", P.VAR_POSITIONAL), P("kwargs", P.VAR_KEYWORD)])
    schema = spec.schema or {}
    properties: Mapping[str, Any] = schema.get("properties") or {}
    required = [str(name) for name in (schema.get("required") or ())]
    ordered = [n for n in properties if n in required]
    ordered += [n for n in required if n not in properties and n not in ordered]
    optional = [n for n in properties if n not in required]
    extra = schema.get("additionalProperties") is True
    params: list[inspect.Parameter] = []
    for name in (*ordered, *optional):
        if not name.isidentifier() or keyword.iskeyword(name):
            extra = True  # reachable only through keyword arguments
            continue
        raw_type = (properties.get(name) or {}).get("type") if name in properties else None
        annotation = _TYPES.get(raw_type, Any) if isinstance(raw_type, str) else Any
        default = P.empty if name in ordered else _UNSET
        params.append(P(name, P.POSITIONAL_OR_KEYWORD, default=default, annotation=annotation))
    if extra:
        params.append(P(_EXTRA_NAME, P.VAR_KEYWORD))
    return inspect.Signature(params)


def bind_args(
    spec: FakeSpec, signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Bind a call to named arguments. Defaults are never applied (so ``Expect.defaults`` logic
    stays meaningful). Raises ``TypeError`` prefixed with ``<name>()`` like a real function."""
    name = spec.name
    if _schema_active(spec):
        # A real function reports an unknown keyword before a missing argument; match it.
        accepted = signature.parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
            for key in kwargs:
                if key not in accepted:
                    raise TypeError(f"{name}() got an unexpected keyword argument '{key}'")
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{name}() {exc}") from None
        out: dict[str, Any] = {}
        for key, value in bound.arguments.items():
            if key == _EXTRA_NAME:
                out.update(value)
            else:
                out[key] = value
        return out
    if spec.kind == "subagent" and len(args) == 1 and not kwargs and isinstance(args[0], str):
        return {"input": args[0]}
    out = {f"arg{i}": value for i, value in enumerate(args)}
    for key, value in kwargs.items():
        if key in out:
            raise TypeError(f"{name}() got multiple values for argument '{key}'")
        out[key] = value
    return out


def _loose_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Best-effort argument record for a call that failed to bind."""
    out = {f"arg{i}": value for i, value in enumerate(args)}
    out.update(kwargs)
    return out


# ------------------------------------------------------------------------ dispatch


def _canonical(value: object) -> str:
    return json.dumps(thaw(value), sort_keys=True, ensure_ascii=False, default=str)


def _matcher_json(matcher: Any) -> Any:
    from ..models import Eq

    if isinstance(matcher, Eq) and matcher.coerce_numeric:
        return thaw(matcher.value)
    return to_dict(matcher)


class FakeState:
    """Per-run state of one fake: rule counters, lock, handler. Never shared between runs."""

    def __init__(
        self,
        spec: FakeSpec,
        handler: Callable[..., Any] | None,
        recorder: Recorder,
        step_kind: str,
    ):
        self.spec = spec
        self.handler = handler
        self.recorder = recorder
        self.step_kind = step_kind
        self.signature = signature_for(spec)
        self.remaining: list[int | None] = [rule.times for rule in spec.responses]
        self.used: list[int] = [0] * len(spec.responses)
        self.lock = threading.Lock()
        self.effect: str | None = spec.effect if step_kind != "model" else None

    # -- rule selection

    def select(self, args_json: Mapping[str, Any]) -> FakeResponse | None:
        """First rule that still has uses left and whose ``when`` matches (atomic per fake)."""
        with self.lock:
            for index, rule in enumerate(self.spec.responses):
                if self.remaining[index] == 0 or not when_matches(rule.when, args_json):
                    continue
                if self.remaining[index] is not None:
                    self.remaining[index] -= 1  # type: ignore[operator]
                self.used[index] += 1
                return rule
        return None

    def rules_text(self) -> str:
        with self.lock:
            parts = []
            for index, rule in enumerate(self.spec.responses):
                when = _canonical({k: _matcher_json(m) for k, m in rule.when.items()})
                left = self.remaining[index]
                parts.append(
                    f"#{index} when={when} remaining={'unlimited' if left is None else left}"
                )
        return "; ".join(parts)

    def unused_rules(self) -> list[tuple[int, int]]:
        """``(index, declared times)`` of explicit-``times`` rules that were never fully used."""
        with self.lock:
            return [
                (index, rule.times)
                for index, rule in enumerate(self.spec.responses)
                if rule.times is not None and self.used[index] < rule.times
            ]

    # -- one call

    def _begin(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        rec = self.recorder
        name = self.spec.name
        try:
            raw = bind_args(self.spec, self.signature, args, kwargs)
        except TypeError as exc:
            pending = rec.start(self.step_kind, name, _loose_args(args, kwargs), self.effect)
            rec.finish(pending, error=f"TypeError: {exc}")
            raise
        pending = rec.start(self.step_kind, name, raw, self.effect)
        frozen, _ = capture_json(raw)
        return raw, frozen, pending

    def _unexpected(self, frozen: Any, pending: Any) -> UnexpectedFakeCall:
        err = UnexpectedFakeCall(self.spec.name, _canonical(frozen), self.rules_text())
        self.recorder.violation(
            "unexpected_call", str(err), fake=self.spec.name, call_id=pending.call_id
        )
        return err

    def _outcome(self, response: FakeResponse, pending: Any) -> Any:
        if response.error is not None:
            raise FakeToolError(response.error, fake=self.spec.name, call_id=pending.call_id)
        return thaw(response.returns)

    def _invoke(self, raw: dict[str, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Call the handler: schema-bound names when there is a schema, else the original call."""
        assert self.handler is not None
        if _schema_active(self.spec):
            return self.handler(**raw)
        return self.handler(*args, **kwargs)

    def _sleep(self, latency_s: float) -> None:
        """Scripted latency, cut at the run deadline (then the run is over: refuse to go on)."""
        deadline = self.recorder.deadline
        left = None if deadline is None else max(0.0, deadline - time.monotonic())
        wait = latency_s if left is None else min(latency_s, left)
        closed = self.recorder.wait_closed(wait)
        if not closed and left is not None and latency_s > left:
            closed = self.recorder.wait_closed(CUT_GRACE_S)  # let the deadline close the run
        if closed or (left is not None and latency_s > left):
            raise RunAbandonedError(
                self.recorder.trace_id, self.recorder.closed or "timed_out", self.spec.name
            )

    async def _asleep(self, latency_s: float) -> None:
        """Async twin of ``_sleep``: cut at the deadline, then wait for the guard to cancel us."""
        deadline = self.recorder.deadline
        left = None if deadline is None else max(0.0, deadline - time.monotonic())
        if left is None or latency_s <= left:
            await asyncio.sleep(latency_s)
            return
        await asyncio.sleep(left)
        await asyncio.sleep(CUT_GRACE_S)
        raise RunAbandonedError(
            self.recorder.trace_id, self.recorder.closed or "timed_out", self.spec.name
        )

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        raw, frozen, pending = self._begin(args, kwargs)
        try:
            rule = self.select(frozen)
            if rule is not None:
                if rule.latency_s:
                    self._sleep(rule.latency_s)
                value = self._outcome(rule, pending)
            elif self.handler is not None:
                value = self._invoke(raw, args, kwargs)
                if inspect.isawaitable(value):
                    value = _await_from_sync(value, self.spec.name)
            elif self.spec.default is not None:
                if self.spec.default.latency_s:
                    self._sleep(self.spec.default.latency_s)
                value = self._outcome(self.spec.default, pending)
            else:
                raise self._unexpected(frozen, pending)
        except BaseException as exc:
            self.recorder.finish(pending, error=f"{type(exc).__name__}: {exc}")
            raise
        self.recorder.finish(pending, result=value)
        return value

    async def acall(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        raw, frozen, pending = self._begin(args, kwargs)
        try:
            rule = self.select(frozen)
            if rule is not None:
                if rule.latency_s:
                    await self._asleep(rule.latency_s)
                value = self._outcome(rule, pending)
            elif self.handler is not None:
                value = self._invoke(raw, args, kwargs)
                if inspect.isawaitable(value):
                    value = await value
            elif self.spec.default is not None:
                if self.spec.default.latency_s:
                    await self._asleep(self.spec.default.latency_s)
                value = self._outcome(self.spec.default, pending)
            else:
                raise self._unexpected(frozen, pending)
        except BaseException as exc:
            self.recorder.finish(pending, error=f"{type(exc).__name__}: {exc}")
            raise
        self.recorder.finish(pending, result=value)
        return value


def _await_from_sync(awaitable: Any, name: str) -> Any:
    """Run an async handler's result from sync code (only when no loop is running here)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise HarnessConfigError(E_ASYNC_HANDLER.format(name=name))

    async def run() -> Any:
        return await awaitable

    return asyncio.run(run())


# ------------------------------------------------------------------------ callables per run


def make_fake_callable(state: FakeState, async_tools: bool) -> Callable[..., Any]:
    """A real function (sync, or ``async def`` for ``async_tools``) that looks like the tool."""
    if async_tools:

        async def fake(*args: Any, **kwargs: Any) -> Any:
            return await state.acall(args, kwargs)

    else:

        def fake(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
            return state.call(args, kwargs)

    spec = state.spec
    fake.__name__ = spec.name
    fake.__qualname__ = spec.name
    fake.__doc__ = spec.description or None
    fake.__signature__ = state.signature  # type: ignore[attr-defined]
    fake.acall = lambda *args, **kwargs: state.acall(args, kwargs)  # type: ignore[attr-defined]
    fake.spec = spec  # type: ignore[attr-defined]
    return fake


@dataclass
class RunFakes:
    """Everything built fresh for one run: callables, states, the optional LLM client."""

    recorder: Recorder
    specs: tuple[FakeSpec, ...]
    states: dict[str, FakeState] = field(default_factory=dict)
    callables: dict[str, Callable[..., Any]] = field(default_factory=dict)
    llm_client: Any = None

    def for_slot(self, slot: str, shape: str) -> Any:
        """The callables delivered through ``slot`` (``tools`` / ``subagents`` / ...)."""
        chosen = [
            (spec.name, self.callables[spec.name])
            for spec in self.specs
            if kind_spec(spec.kind).slot == slot
        ]
        if shape == "list":
            return [fn for _, fn in chosen]
        return dict(chosen)


def build_run_fakes(
    specs: tuple[FakeSpec, ...],
    handlers: Mapping[str, Callable[..., Any]],
    recorder: Recorder,
    async_tools: bool,
) -> RunFakes:
    run = RunFakes(recorder, specs)
    for spec in specs:
        state = FakeState(spec, handlers.get(spec.name), recorder, kind_spec(spec.kind).step_kind)
        run.states[spec.name] = state
        run.callables[spec.name] = make_fake_callable(state, async_tools)
    return run
