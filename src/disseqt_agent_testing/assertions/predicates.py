"""Predicates are registered **names**, never import paths: a case file is data and must not be
able to make the test runner import and run arbitrary code."""

import threading
from collections.abc import Callable, Mapping
from typing import Any

from .config import ExpectConfigError

_LOCK = threading.RLock()
_PREDICATES: dict[str, Callable[..., Any]] = {}


def register_predicate(name: str, fn: Callable[..., Any], *, override: bool = False) -> None:
    """Register ``fn`` under ``name``. For ``Predicate(name)`` it receives the argument value;
    for ``PlanExpect.constraints`` it receives the whole run (a thawed copy). It returns a ``bool``,
    a ``(bool, reason)`` pair or a ``MatchResult``."""
    if not callable(fn):
        raise TypeError("a predicate must be callable")
    with _LOCK:
        if name in _PREDICATES and not override:
            raise ValueError(f"predicate {name!r} is already registered")
        _PREDICATES[name] = fn


def unregister_predicate(name: str) -> None:
    with _LOCK:
        _PREDICATES.pop(name, None)


def lookup_predicate(
    name: str, local: Mapping[str, Callable[..., Any]] | None = None
) -> Callable[..., Any]:
    """``EvalConfig.predicates`` shadows the global registry. Unknown names are test bugs."""
    if local and name in local:
        return local[name]
    with _LOCK:
        fn = _PREDICATES.get(name)
        known = sorted(_PREDICATES)
    if fn is None:
        extra = sorted(local or ())
        raise ExpectConfigError(
            f"unknown predicate {name!r}; register it with register_predicate() or pass it in "
            f"EvalConfig.predicates. Known: {sorted(set(known) | set(extra))}"
        )
    return fn
