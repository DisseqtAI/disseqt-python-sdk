"""The fluent form (FL-01 ... FL-04): sugar that builds Expects and evaluates immediately."""

import pytest

from disseqt_agent_testing.assertions import AgentAssertionError, assert_agent
from disseqt_agent_testing.models import Approx

from .helpers import run_of, step

REFUND = run_of(
    step(1, "get_order", {"order_id": "A12"}, effect="read"),
    step(2, "check_policy", {"order_id": "A12"}, effect="read"),
    step(3, "issue_refund", {"order_id": "A12", "amount": 59.99}),
    tools=("get_order", "check_policy", "issue_refund"),
)


def test_fl01_the_documented_chains_pass() -> None:
    a = assert_agent(REFUND).called("get_order", order_id="A12").before("issue_refund")
    assert len(a.reports) == 2 and all(r.passed for r in a.reports)
    assert_agent(REFUND).called("issue_refund", amount=Approx(59.99, abs_tol=0.01)).once()
    assert_agent(REFUND).never_called("delete_account").stayed_within(max_steps=6)


def test_fl02_failures_raise_on_the_first_failing_call() -> None:
    with pytest.raises(AgentAssertionError, match="tool_selection|missing"):
        assert_agent(REFUND).called("delete_account")
    with pytest.raises(AgentAssertionError):
        assert_agent(REFUND).called("issue_refund", amount=1.0)
    with pytest.raises(AgentAssertionError):
        assert_agent(REFUND).never_called("issue_refund")
    with pytest.raises(AgentAssertionError):
        assert_agent(REFUND).stayed_within(max_steps=2)


def test_fl03_before_names_the_tool_that_never_ran_or_the_observed_order() -> None:
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(REFUND).called("get_order").before("send_email")
    assert str(err.value).startswith(
        "tool_order: expected get_order before send_email; send_email never ran"
    )
    with pytest.raises(AgentAssertionError) as err2:
        assert_agent(REFUND).called("issue_refund").before("check_policy")
    assert (
        "expected issue_refund before check_policy; observed get_order -> check_policy -> issue_refund"
        in str(err2.value)
    )


def test_fl04_repeated_calls_compare_the_first_of_each() -> None:
    run = run_of(step(1, "a"), step(2, "b"), step(3, "a"), step(4, "b"))
    assert_agent(run).called("a").before("b")
    swapped = run_of(step(1, "b"), step(2, "a"), step(3, "b"))
    with pytest.raises(AgentAssertionError):
        assert_agent(swapped).called("a").before("b")


def test_once_fails_when_called_twice() -> None:
    run = run_of(step(1, "a"), step(2, "a"))
    with pytest.raises(AgentAssertionError):
        assert_agent(run).called("a").once()


def test_once_needs_a_preceding_called() -> None:
    with pytest.raises(ValueError, match="needs a preceding .called"):
        assert_agent(REFUND).once()
