"""Matcher semantics (M-01 ... M-23) and Ref resolution."""

import math
from typing import Any

import pytest

from disseqt_agent_testing.assertions import ExpectConfigError, register_matcher_impl
from disseqt_agent_testing.assertions.matching import (
    MatchContext,
    match_dict,
    match_value,
    unregister_matcher_impl,
)
from disseqt_agent_testing.assertions.matching.registry import failed, passed
from disseqt_agent_testing.models import (
    Absent,
    AnyValue,
    Approx,
    Eq,
    Matcher,
    OneOf,
    Optional_,
    Regex,
    freeze_json,
)

CTX = MatchContext()


def one(matcher: Any, actual: Any) -> Any:
    try:
        actual = freeze_json(actual)
    except Exception:  # NaN cannot be frozen; the matcher must still reject it
        pass
    (r,) = match_value(matcher, actual, CTX)
    return r


def test_m01_int_is_not_bool() -> None:
    r = one(Eq(1), True)
    assert (r.ok, r.code) == (False, "type_error:integer->boolean")


def test_m02_bool_is_not_int() -> None:
    assert one(Eq(True), 1).ok is False


def test_m03_numeric_coercion() -> None:
    assert one(Eq(1), 1.0).ok is True
    assert one(Eq(1, coerce_numeric=False), 1.0).code == "type_error:integer->number"


def test_m04_one_of_respects_the_bool_guard() -> None:
    assert one(OneOf((1, 2)), True).ok is False
    assert one(OneOf((1, 2)), 2).ok is True
    assert one(OneOf((1, 2)), 3).code == "value_error:one_of"


def test_m05_string_is_not_number() -> None:
    assert one(Eq("5"), 5).code == "type_error:string->integer"


def test_m06_approx_float_noise() -> None:
    assert one(Approx(0.3, abs_tol=1e-9), 0.1 + 0.2).ok is True


def test_m07_m08_approx_zero_needs_abs_tol() -> None:
    assert one(Approx(0.0), 1e-12).ok is False
    assert one(Approx(0.0, abs_tol=1e-9), 1e-12).ok is True


def test_m09_approx_rejects_bool_str_and_nan() -> None:
    assert one(Approx(1.0), True).code == "type_error:number->boolean"
    assert one(Approx(1.0), "1").code == "type_error:number->string"
    assert one(Approx(1.0), math.nan).code == "value_error:approx"


def test_nan_never_equals() -> None:
    assert one(Eq(1.0), math.nan).ok is False


def test_m12_regex_is_a_search_and_needs_a_string() -> None:
    assert one(Regex("A\\d+"), "order A12 now").ok is True
    assert one(Regex("^A\\d+$"), "order A12 now").ok is False
    assert one(Regex("A"), 5).code == "type_error:string->integer"


def test_regex_flags_are_honoured() -> None:
    import re

    assert one(Regex("abc", flags=re.IGNORECASE), "xABCx").ok is True


def test_m22_optional_and_m23_absent() -> None:
    spec = {"a": Optional_(Eq(1)), "b": Absent()}
    assert all(r.ok for r in match_dict(spec, {}, CTX))
    assert [r.code for r in match_dict(spec, {"a": 2, "b": 1}, CTX)] == [
        "value_error:number",
        "unexpected_param",
    ]


def test_missing_required_and_all_mismatches_are_reported_sorted() -> None:
    spec = {"z": Eq(1), "a": Eq(2), "m": AnyValue()}
    out = match_dict(spec, {"z": 9}, CTX)
    assert [(r.path, r.code) for r in out] == [
        (("a",), "missing_required"),
        (("m",), "missing_required"),
        (("z",), "value_error:number"),
    ]


def test_any_value_accepts_null_but_not_absence() -> None:
    assert match_dict({"a": AnyValue()}, {"a": None}, CTX)[0].ok is True
    assert match_dict({"a": AnyValue()}, {}, CTX)[0].code == "missing_required"


def test_exact_mode_reports_unexpected_keys() -> None:
    out = match_dict({"a": Eq(1)}, {"a": 1, "b": 2}, CTX, exact=True)
    assert [(r.path, r.code) for r in out if not r.ok] == [(("b",), "unexpected_param")]


def test_nested_dicts_and_lists_are_exact_and_positional() -> None:
    assert one(Eq({"a": [1, 2]}), {"a": [1, 2]}).ok is True
    assert one(Eq({"a": [1, 2]}), {"a": [2, 1]}).ok is False
    assert one(Eq({"a": 1}), {"a": 1, "b": 2}).ok is False
    assert one(Eq([1, True]), [1, 1]).ok is False


def test_plain_values_wrap_to_eq() -> None:
    assert one("x", "x").ok is True


def test_e02_a_registered_matcher_implementation_is_used() -> None:
    from dataclasses import dataclass

    @dataclass(slots=True, frozen=True)
    class Even(Matcher):
        pass

    def impl(m: Even, actual: Any, path: Any, ctx: MatchContext) -> list[Any]:
        ok = isinstance(actual, int) and actual % 2 == 0
        return [
            passed(path, "Even", actual) if ok else failed(path, "Even", actual, "value_error:even")
        ]

    register_matcher_impl(Even, impl, describe=lambda m: "Even")
    try:
        assert one(Even(), 4).ok is True and one(Even(), 3).code == "value_error:even"
        with pytest.raises(ValueError, match="already registered"):
            register_matcher_impl(Even, impl)
    finally:
        unregister_matcher_impl(Even)
    with pytest.raises(ExpectConfigError, match="no scoring implementation"):
        one(Even(), 4)
