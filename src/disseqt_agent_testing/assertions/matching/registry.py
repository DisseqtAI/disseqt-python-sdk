"""Matcher semantics registry: ``match_value`` dispatches through it (no hard-coded union)."""

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...models import AgentRun, FakeSpec, Matcher, Step
from ..canonical import clip
from ..config import ExpectConfigError

if TYPE_CHECKING:
    from .ref import RefResolver

Segments = tuple[str | int, ...]


@dataclass(slots=True, frozen=True)
class MatchResult:
    """One leaf outcome. ``ok`` is only meaningful when ``status == "ok"``."""

    status: str  # ok | skipped | error
    ok: bool | None
    score: float
    code: str | None
    path: Segments
    expected: str
    actual: Any = None
    reason: str = ""
    judged: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)


def passed(path: Segments, expected: str, actual: Any) -> MatchResult:
    return MatchResult("ok", True, 1.0, None, path, expected, actual)


def failed(
    path: Segments,
    expected: str,
    actual: Any,
    code: str,
    reason: str = "",
    details: Mapping[str, Any] | None = None,
) -> MatchResult:
    return MatchResult(
        "ok", False, 0.0, code, path, expected, actual, reason, details=details or {}
    )


@dataclass(slots=True)
class MatchContext:
    """What a matcher may need beyond the two values."""

    run: AgentRun | None = None
    consumer: Step | None = None
    refs: "RefResolver | None" = None
    fakes: tuple[FakeSpec, ...] = ()
    judge: Any = None
    use_judge: bool = False
    predicates: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    judge_phase: bool = True  # False while scoring candidate pairs: Semantic waits for phase C
    judge_log: list[dict[str, Any]] | None = None


Impl = Callable[[Any, Any, Segments, MatchContext], list[MatchResult]]
Describer = Callable[[Any], str]

_LOCK = threading.RLock()
_IMPLS: dict[type, Impl] = {}
_DESCRIBERS: dict[type, Describer] = {}


def register_matcher_impl(
    cls: type, impl: Impl, *, describe: Describer | None = None, override: bool = False
) -> None:
    """Give a matcher class its scoring semantics (``impl(matcher, actual, path, ctx)``)."""
    with _LOCK:
        if cls in _IMPLS and not override:
            raise ValueError(f"a matcher implementation for {cls.__name__} is already registered")
        _IMPLS[cls] = impl
        if describe is not None:
            _DESCRIBERS[cls] = describe


def unregister_matcher_impl(cls: type) -> None:
    with _LOCK:
        _IMPLS.pop(cls, None)
        _DESCRIBERS.pop(cls, None)


def describe(matcher: Any) -> str:
    with _LOCK:
        describer = _DESCRIBERS.get(type(matcher))
    if describer is not None:
        return describer(matcher)
    if isinstance(matcher, Matcher):
        return type(matcher).__name__
    return clip(matcher)


def match_value(
    expected: Any, actual: Any, ctx: MatchContext, path: Segments = ()
) -> list[MatchResult]:
    """Score ``actual`` against one matcher (a plain JSON value means ``Eq``)."""
    if not isinstance(expected, Matcher):
        from ...models import Eq

        expected = Eq(expected)
    with _LOCK:
        impl = _IMPLS.get(type(expected))
    if impl is None:
        raise ExpectConfigError(
            f"no scoring implementation for matcher {type(expected).__name__}; it is either not "
            "supported in this version or needs register_matcher_impl(...)"
        )
    return impl(expected, actual, path, ctx)


_MISSING = object()


def match_dict(
    spec: Mapping[str, Any],
    actual: Mapping[str, Any],
    ctx: MatchContext,
    path: Segments = (),
    *,
    exact: bool = False,
) -> list[MatchResult]:
    """Match every key of ``spec`` (sorted, so results are deterministic) and report **all**
    mismatches. A missing key passes only for ``Optional_`` and ``Absent``."""
    from ...models import Absent, Optional_

    out: list[MatchResult] = []
    for key in sorted(spec):
        matcher = spec[key]
        here = (*path, key)
        value = actual.get(key, _MISSING)
        if value is _MISSING:
            if isinstance(matcher, (Optional_, Absent)):
                out.append(passed(here, describe(matcher), None))
            else:
                out.append(
                    failed(here, describe(matcher), None, "missing_required", "argument is missing")
                )
            continue
        out.extend(match_value(matcher, value, ctx, here))
    if exact:
        for key in sorted(k for k in actual if k not in spec):
            out.append(
                failed(
                    (*path, key),
                    "(not expected)",
                    actual[key],
                    "unexpected_param",
                    "argument was not expected",
                )
            )
    return out
