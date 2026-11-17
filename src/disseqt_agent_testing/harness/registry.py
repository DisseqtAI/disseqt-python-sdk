"""Registries that keep the harness open for extension without editing its core.

* ``SLOTS``: names a factory can declare to receive something (``tools``, ``subagents``, ``llm``;
  later ``memory`` and ``history``).
* ``FAKE_KIND_SPECS``: how each kind of fake is delivered and which step kind it records. Its keys
  must equal ``models.FAKE_KINDS``.
* ``ADAPTERS``: framework adapters tried before the plain-Python one (public in phase 1b).
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import HarnessConfigError

_LOCK = threading.RLock()


@dataclass(frozen=True)
class SlotSpec:
    """A factory parameter name that receives a built object.

    ``builder(run_fakes, shape)`` returns the value (``shape`` is ``"dict"`` or ``"list"`` for
    the fake-carrying slots and ignored by others). ``via_kwargs`` slots are also delivered to a
    factory that only declares ``**kwargs``. ``kinds`` lists the fake kinds the slot carries.
    """

    name: str
    builder: Callable[[Any, str], Any]
    via_kwargs: bool = False
    kinds: tuple[str, ...] = ()
    optional: bool = False  # delivered only when something is configured (``llm``)


@dataclass(frozen=True)
class KindSpec:
    """How one kind of fake is delivered and recorded."""

    kind: str
    slot: str
    step_kind: str


SLOTS: dict[str, SlotSpec] = {}
FAKE_KIND_SPECS: dict[str, KindSpec] = {}
ADAPTERS: list[Any] = []


def register_slot(spec: SlotSpec, *, override: bool = False) -> None:
    with _LOCK:
        if spec.name in SLOTS and not override:
            raise HarnessConfigError(f"slot '{spec.name}' is already registered")
        SLOTS[spec.name] = spec


def unregister_slot(name: str) -> None:
    with _LOCK:
        SLOTS.pop(name, None)


def slot_specs() -> dict[str, SlotSpec]:
    with _LOCK:
        return dict(SLOTS)


def register_fake_kind(spec: KindSpec, *, override: bool = False) -> None:
    with _LOCK:
        if spec.kind in FAKE_KIND_SPECS and not override:
            raise HarnessConfigError(f"fake kind '{spec.kind}' is already registered")
        FAKE_KIND_SPECS[spec.kind] = spec


def unregister_fake_kind(kind: str) -> None:
    with _LOCK:
        FAKE_KIND_SPECS.pop(kind, None)


def kind_spec(kind: str) -> KindSpec:
    with _LOCK:
        try:
            return FAKE_KIND_SPECS[kind]
        except KeyError:
            raise HarnessConfigError(f"unknown fake kind '{kind}'") from None


def register_adapter(adapter: Any, *, first: bool = True) -> None:
    with _LOCK:
        if adapter not in ADAPTERS:
            ADAPTERS.insert(0, adapter) if first else ADAPTERS.append(adapter)


def unregister_adapter(adapter: Any) -> None:
    with _LOCK:
        if adapter in ADAPTERS:
            ADAPTERS.remove(adapter)


def adapters() -> list[Any]:
    with _LOCK:
        return list(ADAPTERS)


register_fake_kind(KindSpec("tool", "tools", "tool"))
register_fake_kind(KindSpec("subagent", "subagents", "subagent"))
