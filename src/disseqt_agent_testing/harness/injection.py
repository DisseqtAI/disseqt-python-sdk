"""Decide, from the factory's signature, what to pass it (fakes, slots) and fail clearly if the
test and the factory disagree. Nothing is passed that the factory did not declare."""

import collections.abc
import functools
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..models import FakeSpec
from .errors import (
    E_LLM_REQUIRED,
    E_LLM_UNDELIVERABLE,
    E_POSONLY,
    E_SHAPE,
    E_UNDELIVERABLE,
    E_UNKNOWN_PARAM,
    HarnessConfigError,
    names,
)
from .fakes import RunFakes
from .registry import SlotSpec, kind_spec, register_slot, slot_specs

_P = inspect.Parameter


@dataclass(frozen=True)
class Delivery:
    kind: str  # "slot" or "fake"
    name: str  # slot name or fake name
    shape: str = "dict"  # "dict" or "list" (fake-carrying slots)


@dataclass(frozen=True)
class InjectionPlan:
    """The compiled result: factory parameter name -> what to deliver."""

    params: Mapping[str, Delivery]

    def kwargs(self, run: RunFakes) -> dict[str, Any]:
        slots = slot_specs()
        out: dict[str, Any] = {}
        for param, delivery in self.params.items():
            if delivery.kind == "fake":
                out[param] = run.callables[delivery.name]
            else:
                out[param] = slots[delivery.name].builder(run, delivery.shape)
        return out


# ----------------------------------------------------------------------------- annotation shape


def _hints_target(factory: Callable[..., Any]) -> Any:
    while isinstance(factory, functools.partial):
        factory = factory.func
    return factory.__init__ if inspect.isclass(factory) else factory


def _annotation(factory: Callable[..., Any], param: inspect.Parameter) -> Any:
    ann = param.annotation
    if ann is _P.empty:
        return typing.Any
    if isinstance(ann, str):
        try:
            return typing.get_type_hints(_hints_target(factory)).get(param.name, ann)
        except Exception:  # unresolvable names (framework types): fall back to the text
            return ann
    return ann


def _is_callable_type(tp: Any) -> bool:
    return (
        tp is typing.Any
        or tp is collections.abc.Callable
        or typing.get_origin(tp) is collections.abc.Callable
    )


def _ann_text(ann: Any) -> str:
    return ann if isinstance(ann, str) else str(ann).replace("typing.", "")


def _shape_from_text(text: str, param: str) -> str:
    compact = text.replace(" ", "").replace("typing.", "")
    head = compact.split("[", 1)[0]
    if head in ("dict", "Dict", "Mapping", "MutableMapping"):
        return "dict"
    if head in ("list", "List", "Sequence", "tuple", "Tuple"):
        if "[" not in compact:
            return "list"
        inner = compact[compact.index("[") + 1 : compact.rindex("]")]
        first = inner.split(",", 1)[0]
        if first in ("Any", "object") or first.startswith(("Callable", "collections.abc.Callable")):
            return "list"
        raise HarnessConfigError(E_SHAPE.format(param=param, ann=text))
    return "dict"


def tool_shape(factory: Callable[..., Any], param: inspect.Parameter) -> str:
    """``"dict"`` (default) or ``"list"`` from the parameter annotation; anything else is an error."""
    ann = _annotation(factory, param)
    if isinstance(ann, str):
        return _shape_from_text(ann, param.name)
    if ann is typing.Any or ann is object:
        return "dict"
    origin = typing.get_origin(ann) or ann
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        rest = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(rest) == 1:
            fake = _P(param.name, _P.KEYWORD_ONLY, annotation=rest[0])
            return tool_shape(factory, fake)
    if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
        return "dict"
    if origin in (list, tuple, collections.abc.Sequence):
        args = typing.get_args(ann)
        if not args or _is_callable_type(args[0]):
            return "list"
    raise HarnessConfigError(E_SHAPE.format(param=param.name, ann=_ann_text(ann)))


# ----------------------------------------------------------------------------- planning


def plan_injection(
    factory: Callable[..., Any], specs: tuple[FakeSpec, ...], llm_configured: bool
) -> InjectionPlan:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise HarnessConfigError(f"cannot inspect the agent factory's parameters: {exc}") from None
    params = list(signature.parameters.values())
    slots = slot_specs()
    fake_names = sorted(spec.name for spec in specs)
    ctx = {
        "params": names(p.name for p in params),
        "injectable": names([*sorted(slots), *fake_names]),
    }
    deliveries: dict[str, Delivery] = {}
    for p in params:
        if p.kind in (_P.VAR_POSITIONAL, _P.VAR_KEYWORD):
            continue
        if p.kind is _P.POSITIONAL_ONLY:
            if p.default is _P.empty:
                raise HarnessConfigError(E_POSONLY.format(param=p.name, **ctx))
            continue
        if p.name in slots:
            slot = slots[p.name]
            if slot.optional and not llm_configured:
                if p.default is _P.empty:
                    if p.name == "llm":
                        raise HarnessConfigError(E_LLM_REQUIRED)
                    raise HarnessConfigError(E_UNKNOWN_PARAM.format(param=p.name, **ctx))
                continue
            shape = tool_shape(factory, p) if slot.kinds else "dict"
            deliveries[p.name] = Delivery("slot", p.name, shape)
        elif p.name in fake_names:
            deliveries[p.name] = Delivery("fake", p.name)
        elif p.default is _P.empty:
            raise HarnessConfigError(E_UNKNOWN_PARAM.format(param=p.name, **ctx))
    if any(p.kind is _P.VAR_KEYWORD for p in params):
        for name, slot in slots.items():  # **kwargs receives slots only, never per-name fakes
            if slot.via_kwargs and name not in deliveries and (not slot.optional or llm_configured):
                deliveries[name] = Delivery("slot", name, "dict")
    for spec in specs:
        slot_name = kind_spec(spec.kind).slot
        delivered = deliveries.get(slot_name)
        if not (delivered and delivered.kind == "slot") and spec.name not in deliveries:
            raise HarnessConfigError(
                E_UNDELIVERABLE.format(fake=spec.name, kind=spec.kind, slot=slot_name, **ctx)
            )
    if llm_configured and "llm" not in deliveries:
        raise HarnessConfigError(E_LLM_UNDELIVERABLE.format(params=ctx["params"]))
    return InjectionPlan(deliveries)


def _register_builtin_slots() -> None:
    register_slot(
        SlotSpec("tools", lambda run, shape: run.for_slot("tools", shape), True, ("tool",)),
        override=True,
    )
    register_slot(
        SlotSpec(
            "subagents", lambda run, shape: run.for_slot("subagents", shape), True, ("subagent",)
        ),
        override=True,
    )
    register_slot(
        SlotSpec("llm", lambda run, shape: run.llm_client, True, (), optional=True), override=True
    )


_register_builtin_slots()
