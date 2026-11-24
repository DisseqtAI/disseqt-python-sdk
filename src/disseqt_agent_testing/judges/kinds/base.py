"""Judge kinds: the spec the engine drives (inputs, blocks, precheck, post-validation)."""

import threading
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class Precheck:
    """A decisive deterministic answer: no model is consulted."""

    label: str
    reason: str
    rule: str


@dataclass(slots=True, frozen=True)
class JudgeKindSpec:
    name: str
    required: tuple[str, ...]
    blocks: Callable[[Mapping[str, Any], int, int], list[tuple[str, str]]]
    _: KW_ONLY
    optional: tuple[str, ...] = ()
    role: str = "grader"
    orders: int = 1  # semantic_arg judges both orderings of the two values
    precheck: Callable[[Mapping[str, Any]], Precheck | None] | None = None
    # A problem with the inputs beyond "key missing" (reported as ``input_missing``), or None.
    check_inputs: Callable[[Mapping[str, Any]], str | None] | None = None
    post_validate: (
        Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[str, str] | None] | None
    ) = None
    labels_for: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None
    never_gates: bool = False  # advisory kinds (root cause triage)
    family: str = "binary"
    builtin_calibratable: bool = True
    # The label is derived in code from the structured output (groundedness, plan quality),
    # never taken from a self-reported field; ``summarize`` is kept in ``raw["derived"]``.
    derive: Callable[[Mapping[str, Any], Mapping[str, Any]], str] | None = None
    summarize: Callable[[Mapping[str, Any]], Any] | None = None
    positive: str | None = (
        None  # the "passing" label of a binary kind (calibration's positive class)
    )


_LOCK = threading.RLock()
_KINDS: dict[str, JudgeKindSpec] = {}


def register_kind(spec: JudgeKindSpec, *, override: bool = False) -> None:
    with _LOCK:
        if spec.name in _KINDS and not override:
            raise ValueError(f"judge kind {spec.name!r} is already registered")
        _KINDS[spec.name] = spec


def unregister_kind(name: str) -> None:
    with _LOCK:
        _KINDS.pop(name, None)


def get_kind(name: str) -> JudgeKindSpec | None:
    with _LOCK:
        return _KINDS.get(name)


def list_kinds() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_KINDS))
