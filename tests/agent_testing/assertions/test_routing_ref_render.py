"""Route-name normalisation, ``Ref("input")`` token matching and report rows for unmeasured
checks."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import EvalConfig, evaluate
from disseqt_agent_testing.assertions.assignment import assign
from disseqt_agent_testing.assertions.matching.ref import occurs_as_token
from disseqt_agent_testing.assertions.metrics.base import MetricResult
from disseqt_agent_testing.assertions.render import LINES
from disseqt_agent_testing.assertions.routing import norm_target, same_target
from disseqt_agent_testing.models import Expect, Ref, RouteExpect, ToolExpect, ToolInfo

from .helpers import run_of, step

# ---------------------------------------------------------------- route names


def route_check(expected: str, taken: str, mode: str = "exact") -> Any:
    offered = (ToolInfo(expected, "subagent", "write"), ToolInfo(taken, "subagent", "write"))
    run = run_of(step(1, taken, kind="subagent"), offered=offered)
    rep = evaluate(run, Expect(routes=RouteExpect(path=(expected,), mode=mode)))  # type: ignore[arg-type]
    return next(c for c in rep.checks if c.name == "routing")


@pytest.mark.parametrize(
    ("expected", "taken"),
    [
        ("客服", "销售"),
        ("поддержка", "продажи"),
        ("बिलिंग", "सहायता"),
        ("🚀", "🔥"),
        ("!!!", "???"),
    ],
)
def test_different_non_ascii_route_names_do_not_compare_equal(expected: str, taken: str) -> None:
    assert norm_target(expected) and norm_target(taken)
    assert norm_target(expected) != norm_target(taken)
    for mode in ("exact", "in_order", "first", "any"):
        r = route_check(expected, taken, mode)
        assert r.passed is False and r.reason_code == "route.mismatch", mode
    assert route_check(expected, expected).passed


def test_route_names_are_normalised_unicode_aware() -> None:
    assert norm_target("客服 Agent") == norm_target("客服_agent") == "客服"
    assert norm_target("Ｂｉｌｌｉｎｇ Agent") == "billing"
    assert norm_target("STRASSE") == norm_target("straße")
    assert norm_target("Café Support") == norm_target("café_support") == "café_support"
    assert norm_target("Billing Agent") == norm_target("billing") == "billing"


def test_an_empty_target_never_equals_another_target() -> None:
    assert same_target("billing", "billing")
    assert not same_target("billing", "sales")
    assert not same_target("", "")
    assert norm_target(" ") == norm_target("\t") == ""


# ---------------------------------------------------------------- Ref("input") tokens


def input_ref_score(value: Any, text: str = "refund order A12") -> float:
    tools = (ToolExpect("get_order", {"order_id": Ref("input")}),)
    run = run_of(step(1, "get_order", {"order_id": value}), input=text)
    return float(assign(tools, run, EvalConfig()).pairs[0].eval.score)


@pytest.mark.parametrize("value", ["1", "a", "2", "A1", "fund", "order A1", "", " "])
def test_a_fragment_of_a_token_in_the_user_text_is_not_grounded(value: str) -> None:
    assert input_ref_score(value) == 0.0


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("A12", "refund order A12"),
        ("a12", "refund order A12"),
        ("order A12", "refund order A12"),
        ("A12", "Refund order A12."),
        ("A12", "refund (A12), please"),
        ("A12", 'order "A12"!'),
        ("A-12", "refund order A-12, thanks"),
        ("#A12", "refund order #A12"),
        (12, "refund order 12"),
        (59.99, "I was charged 59.99 twice"),
        ("ana@example.com", "mail ana@example.com."),
        ("Ａ１２", "refund order A12"),
        ("münchen", "Flug nach München, bitte"),
        ("A12", "退款订单A12谢谢"),
        ("北京", "我想去北京旅游"),
        ("दिल्ली", "मुझे दिल्ली जाना है"),
    ],
)
def test_a_whole_token_from_the_user_text_is_grounded(value: Any, text: str) -> None:
    assert input_ref_score(value, text) == 1.0


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("59", "I was charged 59.99 twice"),
        ("99", "I was charged 59.99 twice"),
        ("12", "refund order A12"),
        ("12", "订单123"),
        ("münch", "Flug nach München"),
        ("cafe", "meet at the café"),
        ("दिल", "मुझे दिल्ली जाना है"),
    ],
)
def test_token_boundaries_hold_for_numbers_and_other_scripts(value: str, text: str) -> None:
    assert not occurs_as_token(value, text)
    assert input_ref_score(value, text) == 0.0


# ---------------------------------------------------------------- report rows


@pytest.mark.parametrize("name", sorted(LINES))
@pytest.mark.parametrize("status", ["skipped", "error"])
def test_an_unmeasured_check_is_never_printed_as_failed(name: str, status: str) -> None:
    r = MetricResult(name, status=status, reason_code="run_crashed", reason="the run crashed")  # type: ignore[arg-type]
    text = "\n".join(LINES[name](r))
    assert "FAILED" not in text and status in text


def test_answer_and_forbidden_rows_still_say_failed_for_a_real_failure() -> None:
    for name in ("answer", "forbidden_calls"):
        bad = MetricResult(name, status="ok", value=0.0, passed=False, reason_code="x.y")
        assert "FAILED" in LINES[name](bad)[0]
        good = MetricResult(name, status="ok", value=1.0, passed=True)
        assert LINES[name](good)[0].endswith("ok")
