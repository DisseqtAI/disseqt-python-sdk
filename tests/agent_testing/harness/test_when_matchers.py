"""Rule ``when`` matching (plan 01 matchers applied to live call arguments)."""

from typing import Any

import pytest

from disseqt_agent_testing.harness._when import when_matches
from disseqt_agent_testing.models import (
    Absent,
    AnyValue,
    Approx,
    Contains,
    DateEq,
    DictMatch,
    Eq,
    FrozenDict,
    Normalized,
    OneOf,
    Optional_,
    Regex,
    TypeIs,
    Unordered,
    freeze_json,
)
from disseqt_agent_testing.models.matchers import matcher_map


def check(when: dict[str, Any], args: dict[str, Any]) -> bool:
    return when_matches(matcher_map(when, "when"), freeze_json(args))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("when", "args", "expected"),
    [
        ({}, {"a": 1}, True),  # an empty when matches every call
        ({"a": 1}, {"a": 1}, True),
        ({"a": 1}, {"a": 2}, False),
        ({"a": 1}, {}, False),  # missing key
        ({"a": 1}, {"a": True}, False),  # bool is not 1
        ({"a": True}, {"a": 1}, False),
        ({"a": 1}, {"a": 1.0}, True),
        ({"a": Eq("x")}, {"a": "x"}, True),
        ({"a": OneOf(("x", "y"))}, {"a": "y"}, True),
        ({"a": OneOf(("x", "y"))}, {"a": "z"}, False),
        ({"a": AnyValue()}, {"a": None}, True),
        ({"a": AnyValue()}, {}, False),
        ({"a": Absent()}, {}, True),
        ({"a": Absent()}, {"a": 1}, False),
        ({"a": Optional_(Eq(1))}, {}, True),
        ({"a": Optional_(Eq(1))}, {"a": 1}, True),
        ({"a": Optional_(Eq(1))}, {"a": 2}, False),
        ({"a": Regex("^A\\d+$")}, {"a": "A12"}, True),
        ({"a": Regex("^A\\d+$")}, {"a": "xA12"}, False),
        ({"a": Regex("^A\\d+$")}, {"a": 12}, False),
        ({"a": Approx(1.0, rel_tol=0.01)}, {"a": 1.005}, True),
        ({"a": Approx(1.0, rel_tol=0.01)}, {"a": 1.5}, False),
        ({"a": TypeIs(("string",))}, {"a": "s"}, True),
        ({"a": TypeIs(("string",))}, {"a": 1}, False),
        ({"a": Normalized(("hello",))}, {"a": "  HELLO "}, True),
        ({"a": Contains(("x",))}, {"a": ["x", "y"]}, True),
        ({"a": Contains(("z",))}, {"a": ["x", "y"]}, False),
        ({"a": Unordered((1, 2))}, {"a": [2, 1]}, True),
        ({"a": Unordered((1, 2))}, {"a": [2, 2]}, False),
        ({"a": DictMatch({"k": Eq(1)})}, {"a": {"k": 1, "z": 2}}, True),
        ({"a": DictMatch({"k": Eq(1)})}, {"a": {"k": 2}}, False),
        ({"a": DateEq("2026-01-01")}, {"a": "2026-01-01T00:00:00Z"}, True),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),  # all keys must match
    ],
)
def test_when_matches(when: dict[str, Any], args: dict[str, Any], expected: bool) -> None:
    assert check(when, args) is expected


def test_a_matcher_that_cannot_decide_never_matches() -> None:
    assert isinstance(freeze_json({"a": 1}), FrozenDict)
    assert check({"a": Regex("x")}, {"a": {"nested": 1}}) is False
