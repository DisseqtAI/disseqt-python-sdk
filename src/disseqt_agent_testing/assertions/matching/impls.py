"""Built-in matcher semantics for phase 3a: Eq, OneOf, AnyValue, Absent, Optional_, Regex,
Approx and Ref. (3b adds Normalized, TypeIs, Unordered, Contains, DictMatch, DateEq, Predicate,
Semantic.)"""

import math
import re
from typing import Any

from ...models import (
    Absent,
    AnyValue,
    Approx,
    Eq,
    OneOf,
    Optional_,
    Ref,
    Regex,
)
from ..canonical import clip
from .registry import (
    MatchContext,
    MatchResult,
    Segments,
    describe,
    failed,
    match_value,
    passed,
    register_matcher_impl,
)


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def deep_equal(expected: Any, actual: Any, coerce_numeric: bool = True) -> tuple[bool, str | None]:
    """Strict JSON equality. Returns ``(ok, code)``: ``type_error:<exp>-><act>`` for a type
    mismatch, ``value_error:<type>`` for the same type with a different value. The bool guard
    comes first so ``1`` never equals ``True``; ``NaN`` never equals anything."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        if isinstance(expected, bool) and isinstance(actual, bool):
            return (expected is actual, None if expected is actual else "value_error:boolean")
        return False, f"type_error:{json_type(expected)}->{json_type(actual)}"
    if is_number(expected) and is_number(actual):
        if not coerce_numeric and type(expected) is not type(actual):
            return False, f"type_error:{json_type(expected)}->{json_type(actual)}"
        if isinstance(expected, float) and math.isnan(expected):
            return False, "value_error:number"
        if isinstance(actual, float) and math.isnan(actual):
            return False, "value_error:number"
        return (expected == actual, None if expected == actual else "value_error:number")
    et, at = json_type(expected), json_type(actual)
    if et != at:
        return False, f"type_error:{et}->{at}"
    if et == "array":
        if len(expected) != len(actual):
            return False, "value_error:array"
        for e, a in zip(expected, actual, strict=True):
            ok, _ = deep_equal(e, a, coerce_numeric)
            if not ok:
                return False, "value_error:array"
        return True, None
    if et == "object":
        if set(expected) != set(actual):
            return False, "value_error:object"
        for key in expected:
            ok, _ = deep_equal(expected[key], actual[key], coerce_numeric)
            if not ok:
                return False, "value_error:object"
        return True, None
    return (expected == actual, None if expected == actual else f"value_error:{et}")


def _eq(m: Eq, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    ok, code = deep_equal(m.value, actual, m.coerce_numeric)
    shown = describe(m)
    if ok:
        return [passed(path, shown, actual)]
    return [failed(path, shown, actual, code or "value_error:value", f"expected {shown}")]


def _one_of(m: OneOf, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    for value in m.values:
        if deep_equal(value, actual)[0]:
            return [passed(path, shown, actual)]
    return [failed(path, shown, actual, "value_error:one_of", f"expected {shown}")]


def _any(m: AnyValue, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    return [passed(path, describe(m), actual)]


def _absent(m: Absent, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    return [failed(path, describe(m), actual, "unexpected_param", "argument must be absent")]


def _optional(m: Optional_, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    return match_value(m.inner, actual, ctx, path)


MAX_REGEX_TEXT = 100_000


def _regex(m: Regex, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    """``re.search`` over at most ``MAX_REGEX_TEXT`` characters, so a catastrophic pattern on a
    huge agent output is bounded (text beyond the cap is not searched and the result says so in
    its details). Patterns are author-trusted: this is a bound on the input, not a sandbox, and
    no timeout thread is used."""
    shown = describe(m)
    if not isinstance(actual, str):
        return [failed(path, shown, actual, f"type_error:string->{json_type(actual)}")]
    truncated = len(actual) > MAX_REGEX_TEXT
    details = {"truncated": f"searched the first {MAX_REGEX_TEXT} of {len(actual)} characters"}
    found = re.search(m.pattern, actual[:MAX_REGEX_TEXT], m.flags) is not None
    if found:
        return [
            MatchResult("ok", True, 1.0, None, path, shown, actual, details=details if truncated else {})
        ]  # fmt: skip
    return [
        MatchResult(
            "ok", False, 0.0, "value_error:regex", path, shown, actual,
            f"expected {shown}" + (f" ({details['truncated']})" if truncated else ""),
            details=details if truncated else {},
        )
    ]  # fmt: skip


def _approx(m: Approx, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    shown = describe(m)
    if not is_number(actual):
        return [failed(path, shown, actual, f"type_error:number->{json_type(actual)}")]
    if isinstance(actual, float) and not math.isfinite(actual):
        return [failed(path, shown, actual, "value_error:approx", "not a finite number")]
    if math.isclose(actual, m.value, rel_tol=m.rel_tol, abs_tol=m.abs_tol):
        return [passed(path, shown, actual)]
    return [failed(path, shown, actual, "value_error:approx", f"expected {shown}")]


def _ref(m: Ref, actual: Any, path: Segments, ctx: MatchContext) -> list[MatchResult]:
    from .ref import match_ref

    return match_ref(m, actual, path, ctx)


def register_builtin_impls() -> None:
    def desc_eq(m: Eq) -> str:
        return f"Eq({clip(m.value)})"

    register_matcher_impl(Eq, _eq, describe=desc_eq, override=True)
    register_matcher_impl(
        OneOf, _one_of, describe=lambda m: f"OneOf({clip(list(m.values))})", override=True
    )
    register_matcher_impl(AnyValue, _any, describe=lambda m: "AnyValue()", override=True)
    register_matcher_impl(Absent, _absent, describe=lambda m: "Absent()", override=True)
    register_matcher_impl(
        Optional_, _optional, describe=lambda m: f"Optional_({describe(m.inner)})", override=True
    )
    register_matcher_impl(Regex, _regex, describe=lambda m: f"Regex({m.pattern!r})", override=True)

    def desc_approx(m: Approx) -> str:
        return f"Approx({m.value}, abs_tol={m.abs_tol}, rel_tol={m.rel_tol})"

    register_matcher_impl(Approx, _approx, describe=desc_approx, override=True)
    register_matcher_impl(Ref, _ref, describe=lambda m: f"Ref({m.source!r})", override=True)


register_builtin_impls()
