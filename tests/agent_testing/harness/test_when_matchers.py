"""Rule ``when`` matching (plan 01 matchers applied to live call arguments)."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions.matching import MatchContext, match_value
from disseqt_agent_testing.assertions.matching.impls import MAX_REGEX_TEXT
from disseqt_agent_testing.harness import HarnessConfigError
from disseqt_agent_testing.harness._when import matches, when_matches
from disseqt_agent_testing.models import (
    Absent,
    AnyValue,
    Approx,
    Contains,
    DateEq,
    DictMatch,
    Eq,
    FrozenDict,
    Matcher,
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


AGREEMENT_CASES: list[tuple[Matcher, Any]] = [
    (DictMatch({"k": Eq(1)}, defaults={"k": Eq(1)}), {"k": 1}),  # a default that is also a field
    (DictMatch({"k": Eq(1), "tmp_x": Eq(2)}, ignore=("tmp_*",)), {"k": 1}),  # ignored field
    (DictMatch({"k": Eq(1)}, mode="exact", ignore=("tmp_*",)), {"k": 1, "tmp_y": 3}),
    (DictMatch({"k": Eq(1)}, mode="exact", defaults={"d": Eq(0)}), {"k": 1, "d": 0}),
    (DictMatch({"k": Eq(1)}, mode="exact", defaults={"d": Eq(0)}), {"k": 1, "d": 5}),
    (DictMatch({"k": Eq(1)}, mode="exact"), {"k": 1, "z": 2}),
    (DateEq("2026-01-01T10:00:00", tz="UTC"), "2026-01-01T10:00:00Z"),  # naive side localised
    (DateEq("2026-01-01T10:00:00Z", tz="Europe/Berlin"), "2026-01-01T11:00:00"),
    (DateEq("2026-01-01T10:00:00", tz="Europe/Berlin"), "2026-01-01T10:00:00"),
    (DateEq("2026-01-01T10:00:00"), "2026-01-01T10:00:00Z"),
    (DateEq("2026-01-01", tz="Europe/Berlin"), "2025-12-31T23:30:00Z"),
    (Contains(("x", Regex("^a"))), "xyz"),
    (Contains((Regex("^a"),)), ["b", "ab"]),
    (Unordered((Eq(1), TypeIs(("string",)))), ["s", 1]),
    (OneOf((1, "a")), True),
    (Eq(1, coerce_numeric=False), 1.0),
    (Approx(1.0, abs_tol=0.1), True),
    (Normalized(("a b",)), "A-B"),
    (Optional_(Regex("x")), "y"),
]


@pytest.mark.parametrize(("matcher", "value"), AGREEMENT_CASES, ids=repr)
def test_dispatch_and_the_assertion_engine_give_the_same_answer(
    matcher: Matcher, value: Any
) -> None:
    frozen = freeze_json(value)
    scored = match_value(matcher, frozen, MatchContext())
    assert matches(matcher, frozen) is all(r.ok is True for r in scored)
    assert check({"a": matcher}, {"a": value}) is matches(matcher, frozen)


@pytest.mark.parametrize(("matcher", "value"), AGREEMENT_CASES[:3] + AGREEMENT_CASES[6:8], ids=repr)
def test_rules_the_assertion_engine_accepts_also_apply_at_dispatch(
    matcher: Matcher, value: Any
) -> None:
    assert check({"a": matcher}, {"a": value}) is True


def test_regex_searches_the_same_capped_text_as_the_assertion_engine() -> None:
    assert check({"a": Regex("needle")}, {"a": "x" * (MAX_REGEX_TEXT - 6) + "needle"}) is True
    assert check({"a": Regex("needle")}, {"a": "x" * MAX_REGEX_TEXT + "needle"}) is False


def test_a_rule_that_cannot_be_evaluated_is_a_config_error() -> None:
    with pytest.raises(HarnessConfigError, match="No/Such_Zone"):
        check({"a": DateEq("2026-01-01", tz="No/Such_Zone")}, {"a": "2026-01-01"})
