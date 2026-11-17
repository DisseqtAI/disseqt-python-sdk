"""The 3a metrics: tool_selection, tool_args, order, forbidden_calls, budgets, answer, decision."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import ExpectConfigError, evaluate
from disseqt_agent_testing.models import (
    Approx,
    Decision,
    Expect,
    ForbiddenCall,
    ModelValidationError,
    Ref,
    Regex,
    ToolExpect,
    Usage,
)

from .helpers import result_of, run_of, step


def report(run: Any, expect: Expect, **kw: Any) -> Any:
    return evaluate(run, expect, **kw)


def only(rep: Any, name: str) -> Any:
    got = rep.check(name)
    assert got is not None, [c.name for c in rep.checks]
    return got


# ---------------------------------------------------------------- tool_selection


def test_selection_all_required_matched() -> None:
    run = run_of(step(1, "a"), step(2, "b"))
    r = only(report(run, Expect(tools=(ToolExpect("a"), ToolExpect("b")))), "tool_selection")
    assert r.passed and r.sub_scores["recall"] == 1.0 and r.sub_scores["precision"] == 1.0


def test_selection_missing_required_call() -> None:
    r = only(
        report(run_of(step(1, "a")), Expect(tools=(ToolExpect("a"), ToolExpect("b")))),
        "tool_selection",
    )
    assert r.passed is False and r.reason_code == "tool.missing"
    assert r.sub_scores["recall"] == 0.5 and list(r.details["missing"]) == ["b#1"]


def test_a07_empty_actual_has_recall_zero_and_no_precision() -> None:
    rep = report(run_of(text="hi"), Expect(tools=(ToolExpect("a"),)))
    r = only(rep, "tool_selection")
    assert r.sub_scores["recall"] == 0.0 and r.sub_scores["precision"] is None
    assert only(rep, "tool_selection").passed is False
    assert rep.check("tool_args") is None  # not enabled: no args were expected


def test_selection_unexpected_write_fails_but_unexpected_read_is_allowed() -> None:
    expect = Expect(tools=(ToolExpect("a"),))
    bad = only(report(run_of(step(1, "a"), step(2, "w", effect="write")), expect), "tool_selection")
    assert bad.passed is False and bad.reason_code == "tool.unexpected"
    ok = only(report(run_of(step(1, "a"), step(2, "r", effect="read")), expect), "tool_selection")
    assert ok.passed is True and ok.details["allowed_reads"][0]["name"] == "r"


def test_a06_forbid_policy_counts_reads_too() -> None:
    expect = Expect(tools=(ToolExpect("a"),), extra_calls="forbid")
    r = only(report(run_of(step(1, "a"), step(2, "r", effect="read")), expect), "tool_selection")
    assert r.passed is False


def test_allow_policy_ignores_extras() -> None:
    expect = Expect(tools=(ToolExpect("a"),), extra_calls="allow")
    assert only(report(run_of(step(1, "a"), step(2, "w")), expect), "tool_selection").passed


def test_a05_explicit_count_cap_is_a_veto() -> None:
    expect = Expect(tools=(ToolExpect("a", count=1),), extra_calls="allow")
    r = only(report(run_of(step(1, "a"), step(2, "a")), expect), "tool_selection")
    assert r.passed is False and r.vetoed is True and r.reason_code == "wrong_count"


def test_allowed_tools_and_hallucinated_names_are_vetoes() -> None:
    run = run_of(step(1, "a"), step(2, "evil"), tools=("a", "b"))
    r = only(report(run, Expect(allowed_tools=("a",))), "tool_selection")
    assert r.vetoed and r.reason_code == "tool.hallucinated"  # evil was never offered
    run2 = run_of(step(1, "a"), step(2, "b"), tools=("a", "b"))
    r2 = only(report(run2, Expect(allowed_tools=("a",))), "tool_selection")
    assert r2.vetoed and r2.reason_code == "tool.not_allowed"


def test_selection_is_not_enabled_by_default() -> None:
    assert report(run_of(step(1, "a")), Expect(max_steps=5)).check("tool_selection") is None


# ---------------------------------------------------------------- tool_args


def test_args_diff_reports_every_wrong_leaf_with_a_path() -> None:
    expect = Expect(tools=(ToolExpect("t", {"a": 1, "b": "x", "c": Approx(1.0)}),))
    rep = report(run_of(step(1, "t", {"a": 2, "b": "x", "c": 5.0})), expect)
    r = only(rep, "tool_args")
    assert r.passed is False and r.reason_code == "value_error:number"
    assert [d["path_str"] for d in r.details["pairs"]] == ["$.a", "$.c"]
    assert r.sub_scores["arg_score"] == pytest.approx(1 / 3, abs=1e-6)


def test_args_no_match_is_skipped_not_a_perfect_score() -> None:
    rep = report(run_of(text="x"), Expect(tools=(ToolExpect("t", {"a": 1}),)))
    r = only(rep, "tool_args")
    assert r.status == "skipped" and r.reason_code == "not_applicable"
    assert rep.outcome == "failed"  # tool_selection owns the failure


def test_args_ref_to_earlier_result() -> None:
    tools = (
        ToolExpect("get_order", id="lookup"),
        ToolExpect("refund", {"order_id": Ref("call:lookup.result.order_id")}),
    )
    steps = (
        step(1, "get_order", effect="read", result={"order_id": "A12"}),
        step(2, "refund", {"order_id": "Z9"}),
    )
    rep = report(run_of(*steps), Expect(tools=tools))
    r = only(rep, "tool_args")
    assert r.passed is False and r.reason_code == "ungrounded_value"
    assert r.details["pairs"][0]["ref"]["reason"] == "value differs"


def test_args_error_leaf_without_failures_is_an_instrument_error() -> None:
    tools = (ToolExpect("t", {"a": Ref("fake:nothing.x")}),)
    r = only(report(run_of(step(1, "t", {"a": 1})), Expect(tools=tools)), "tool_args")
    assert (
        r.status == "error"
        and r.reason_code == "ref_unresolvable"
        or r.reason_code == "predicate_error"
    )


# ---------------------------------------------------------------- order


def abc(**kw: Any) -> tuple[ToolExpect, ...]:
    return (ToolExpect("a", id="a", **kw), ToolExpect("b", id="b"))


def test_o01_in_order_ignores_extra_calls() -> None:
    run = run_of(step(1, "a"), step(2, "x", effect="read"), step(3, "b"))
    assert only(report(run, Expect(tools=abc(), order="in_order")), "order").passed


def test_o02_swapped_calls_fail_the_sequence() -> None:
    tools = (ToolExpect("s", {"q": 1}), ToolExpect("s", {"q": 2}))
    run = run_of(step(1, "s", {"q": 2}), step(2, "s", {"q": 1}))
    r = only(report(run, Expect(tools=tools, order="in_order")), "order")
    assert r.passed is False and r.reason_code == "order.sequence" and r.value == 0.5


def test_o03_identical_duplicates_pass() -> None:
    tools = (ToolExpect("s"), ToolExpect("s"))
    assert only(
        report(run_of(step(1, "s"), step(2, "s")), Expect(tools=tools, order="in_order")), "order"
    ).passed


def test_o04_exact_fails_with_an_interleaved_extra() -> None:
    run = run_of(step(1, "a"), step(2, "x", effect="read"), step(3, "b"))
    r = only(report(run, Expect(tools=abc(), order="exact")), "order")
    assert r.passed is False and r.reason_code == "order.exact"
    clean = run_of(step(1, "a"), step(2, "b"))
    assert only(report(clean, Expect(tools=abc(), order="exact")), "order").passed


def test_o05_an_edge_violated_names_the_observed_order() -> None:
    tools = (ToolExpect("a", id="a", after=("b",)), ToolExpect("b", id="b"))
    r = only(
        report(run_of(step(1, "a"), step(2, "b")), Expect(tools=tools, order="partial")), "order"
    )
    assert r.passed is False and r.reason_code == "order.edge_violated"
    assert "expected b before a; observed a -> b" in r.reason


def test_o06_a_missing_endpoint_is_unverifiable_and_counts_as_violated() -> None:
    tools = (ToolExpect("a", id="a"), ToolExpect("b", id="b", after=("a",)))
    r = only(report(run_of(step(1, "b")), Expect(tools=tools, order="partial")), "order")
    assert r.passed is False and r.reason_code == "order.edge_unverifiable"


def test_o07_diamond_dag_accepts_both_middle_orders() -> None:
    tools = (
        ToolExpect("a", id="a"),
        ToolExpect("b", id="b", after=("a",)),
        ToolExpect("c", id="c", after=("a",)),
        ToolExpect("d", id="d", after=("b", "c")),
    )
    for middle in (("b", "c"), ("c", "b")):
        steps = [step(1, "a"), step(2, middle[0]), step(3, middle[1]), step(4, "d")]
        assert only(report(run_of(*steps), Expect(tools=tools, order="partial")), "order").passed


def test_o08_an_after_cycle_is_rejected_when_the_expect_is_built() -> None:
    with pytest.raises(ModelValidationError):
        Expect(tools=(ToolExpect("a", id="a", after=("b",)), ToolExpect("b", id="b", after=("a",))))


def test_o09_same_step_without_parallel_groups_is_incomplete() -> None:
    tools = (ToolExpect("a", id="a"), ToolExpect("b", same_step_as="a"))
    rep = report(run_of(step(1, "a"), step(2, "b")), Expect(tools=tools))
    r = only(rep, "order")
    assert r.status == "skipped" and r.reason_code == "data_unavailable"
    assert rep.outcome == "incomplete"


def test_o10_same_step_with_recorded_groups() -> None:
    tools = (ToolExpect("a", id="a"), ToolExpect("b", same_step_as="a"))
    same = run_of(step(1, "a", parallel_group=1), step(2, "b", parallel_group=1))
    diff = run_of(step(1, "a", parallel_group=1), step(2, "b", parallel_group=2))
    assert only(report(same, Expect(tools=tools)), "order").passed
    r = only(report(diff, Expect(tools=tools)), "order")
    assert r.passed is False and r.reason_code == "order.same_step"


def test_edges_in_the_same_batch_are_flagged() -> None:
    tools = (ToolExpect("a", id="a"), ToolExpect("b", id="b", after=("a",)))
    run = run_of(step(1, "a", parallel_group=3), step(2, "b", parallel_group=3))
    r = only(report(run, Expect(tools=tools, order="partial")), "order")
    assert r.reason_code == "order.edge_same_batch"


def test_repeated_calls_first_matching_call_decides_an_edge() -> None:
    tools = (
        ToolExpect("a", id="a", count=(1, None)),
        ToolExpect("b", id="b", after=("a",), count=(1, None)),
    )
    run = run_of(step(1, "a"), step(2, "b"), step(3, "a"))
    assert only(
        report(run, Expect(tools=tools, order="partial", extra_calls="allow")), "order"
    ).passed


# ---------------------------------------------------------------- forbidden_calls


def test_f01_a_forbidden_call_is_a_veto_regardless_of_thresholds() -> None:
    expect = Expect(forbidden=(ForbiddenCall("delete_*", reason="never delete"),))
    rep = report(run_of(step(1, "delete_account")), expect)
    r = only(rep, "forbidden_calls")
    assert r.passed is False and r.vetoed and r.reason_code == "forbidden.called"
    assert "never delete" in r.reason and rep.outcome == "failed"


def test_f02_unless_after_exempts_a_prior_approval() -> None:
    fc = ForbiddenCall("issue_refund", unless_after="policy")
    tools = (ToolExpect("check_policy", id="policy"),)
    ok = run_of(step(1, "check_policy", effect="read"), step(2, "issue_refund"))
    bad = run_of(step(1, "issue_refund"), step(2, "check_policy", effect="read"))
    assert only(report(ok, Expect(tools=tools, forbidden=(fc,))), "forbidden_calls").passed
    assert (
        only(report(bad, Expect(tools=tools, forbidden=(fc,))), "forbidden_calls").passed is False
    )


def test_f02_unless_after_can_name_an_expectation() -> None:
    tools = (ToolExpect("approve", id="appr"),)
    fc = ForbiddenCall("pay", unless_after="appr")
    ok = run_of(step(1, "approve"), step(2, "pay"))
    assert only(report(ok, Expect(tools=tools, forbidden=(fc,))), "forbidden_calls").passed


def test_f03_forbidden_args_use_matchers() -> None:
    fc = ForbiddenCall("issue_refund", {"amount": Approx(100.0, abs_tol=1.0)})
    big = run_of(step(1, "issue_refund", {"amount": 100.4}))
    small = run_of(step(1, "issue_refund", {"amount": 5}))
    assert only(report(big, Expect(forbidden=(fc,))), "forbidden_calls").passed is False
    assert only(report(small, Expect(forbidden=(fc,))), "forbidden_calls").passed


def test_f04_undecidable_matchers_in_a_veto_are_rejected_at_construction() -> None:
    with pytest.raises(ModelValidationError):
        ForbiddenCall("x", {"a": Ref("input")})


# ---------------------------------------------------------------- budgets


def test_b01_inclusive_bounds() -> None:
    run = run_of(step(1, "a"), step(2, "b"), step(3, "c"))
    assert only(report(run, Expect(max_steps=3)), "budgets").passed
    r = only(report(run, Expect(max_steps=2)), "budgets")
    assert (
        r.passed is False and r.reason_code == "budget.exceeded" and "used 3, limit 2" in r.reason
    )


def test_b05_tool_call_cap_int_and_mapping() -> None:
    run = run_of(step(1, "get_a", effect="read"), step(2, "get_b", effect="read"), step(3, "pay"))
    assert only(report(run, Expect(max_tool_calls=3)), "budgets").passed
    assert only(report(run, Expect(max_tool_calls=2)), "budgets").passed is False
    r = only(report(run, Expect(max_tool_calls={"get_*": 1, "pay": 1})), "budgets")
    assert r.passed is False and r.components[0].details["over"] == {
        "get_*": {"used": 2, "limit": 1}
    }


def test_b02_llm_call_budget_needs_captured_model_steps() -> None:
    run = run_of(step(1, "a"))
    rep = report(run, Expect(max_llm_calls=2))
    assert rep.outcome == "incomplete"
    warn = report(run, Expect(max_llm_calls=2), on_unavailable="warn")
    assert warn.outcome == "passed" and any("model calls not captured" in w for w in warn.warnings)
    skip = report(run, Expect(max_llm_calls=2), on_unavailable="skip")
    assert skip.outcome == "passed" and not skip.warnings


def test_llm_call_budget_counts_model_steps() -> None:
    run = run_of(
        step(1, "m", kind="model"), step(2, "a"), step(3, "m", kind="model"), captured=True
    )
    assert only(report(run, Expect(max_llm_calls=2)), "budgets").passed
    assert only(report(run, Expect(max_llm_calls=1)), "budgets").passed is False


def test_b03_b04_token_budget_lower_bound() -> None:
    u = Usage(input_tokens=60, output_tokens=60)
    full = run_of(step(1, "m", kind="model", usage=u), captured=True)
    assert only(report(full, Expect(max_tokens=200)), "budgets").passed
    assert only(report(full, Expect(max_tokens=100)), "budgets").passed is False
    partial = run_of(step(1, "m", kind="model", usage=u), step(2, "m", kind="model"), captured=True)
    assert only(report(partial, Expect(max_tokens=100)), "budgets").passed is False  # lower bound
    rep = report(partial, Expect(max_tokens=500))
    assert rep.outcome == "incomplete"
    none = run_of(step(1, "m", kind="model"), captured=True)
    assert report(none, Expect(max_tokens=500)).outcome == "incomplete"


def test_b06_latency_from_a_result_and_from_a_bare_run() -> None:
    run = run_of(step(1, "a", ts=0.0, duration_ms=50.0))
    res = result_of(run, duration_ms=1500.0, setup_ms=500.0)
    assert only(report(res, Expect(max_latency_s=1.5)), "budgets").passed
    assert only(report(res, Expect(max_latency_s=0.5)), "budgets").passed is False
    assert only(report(run, Expect(max_latency_s=0.01)), "budgets").passed is False  # lower bound
    assert report(run, Expect(max_latency_s=5.0)).outcome == "incomplete"


# ---------------------------------------------------------------- answer


def test_c06_answer_contains_strings_and_matchers() -> None:
    run = run_of(text="Your refund of 59.99 for order_A12 is done.")
    expect = Expect(answer_contains=("refund", "order A12", Regex(r"\d+\.\d{2}")))
    assert only(report(run, expect), "answer").passed
    miss = only(report(run_of(text="nothing"), expect), "answer")
    assert miss.passed is False and miss.reason_code == "answer.missing_text"
    assert miss.value == 0.0


def test_answer_excludes() -> None:
    r = only(
        report(run_of(text="password is hunter2"), Expect(answer_excludes=("password",))), "answer"
    )
    assert r.passed is False and r.reason_code == "answer.forbidden_text"


def test_c07_no_answer_fails_with_no_output() -> None:
    r = only(report(run_of(text=None, outputs=(None,)), Expect(answer_contains=("x",))), "answer")
    assert r.reason_code == "answer.no_output"


# ---------------------------------------------------------------- decision_match


def one_turn(text: str | None, *steps: Any) -> Any:
    return run_of(*steps, text=text)


@pytest.mark.parametrize(
    ("text", "steps", "label"),
    [
        ("Done.", (step(1, "issue_refund"),), "call"),  # D-01
        ("Which order number is this about?", (), "ask"),  # D-02
        (
            "Order A1 is not eligible for a refund.",
            (step(1, "check", effect="read"),),
            "refuse",
        ),  # D-03
        (
            "Sorry for the delay, " + "here is the full fix and some more words " * 12,
            (),
            "answer",
        ),  # D-04
        ("I cannot stress enough how important backups are.", (), "answer"),  # D-05
        ("I can’t help with that.", (), "refuse"),  # D-06
        ("As an airbnb host you should know things.", (), "answer"),  # D-07
        ("Here it is. Anything else?", (), "answer"),  # D-08
        ("```\nWhat is your name?\n```\nAll done.", (), "answer"),  # D-09
        ("¿Qué fecha prefieres?", (), "ask"),  # D-10
        (None, (), "empty"),  # D-11
    ],
)
def test_classifier_labels(text: str | None, steps: tuple[Any, ...], label: str) -> None:
    from disseqt_agent_testing.assertions import EvalConfig, classify_turn

    assert classify_turn(one_turn(text, *steps), 0, EvalConfig()).label == label


def test_d11_empty_output_fails_any_expectation() -> None:
    for wanted in (Decision.ANSWER, Decision.REFUSE, Decision.ASK, Decision.CALL):
        r = only(
            report(one_turn(None), Expect(decision=wanted, allowed_tools=("*",))), "decision_match"
        )
        assert r.passed is False and r.components[0].reason_code == "decision.empty"


def test_d12_a_declared_decision_wins() -> None:
    run = run_of(text=None, outputs=({"decision": "refuse"},))
    r = only(
        report(run, Expect(decision=Decision.REFUSE), decision_source="answer.decision"),
        "decision_match",
    )
    assert r.passed and r.components[0].details["rule"] == "declared"


def test_d13_refusal_and_question_is_low_confidence_refuse() -> None:
    from disseqt_agent_testing.assertions import EvalConfig, classify_turn

    c = classify_turn(
        one_turn("I'm sorry, I can't help with that. Which account do you mean?"), 0, EvalConfig()
    )
    assert (c.label, c.confidence, c.low_confidence) == ("refuse", 0.6, True)


def test_d14_decision_by_turn_evaluates_each_turn() -> None:
    run = run_of(
        step(2, "issue_refund", turn=1),
        text="Refunded.",
        outputs=("Which order?", "Refunded."),
        follow_ups=("A12",),
    )
    expect = Expect(
        decision=Decision.ASK, decision_by_turn={1: Decision.CALL}, allowed_tools=("*",)
    )
    r = only(report(run, expect), "decision_match")
    assert r.passed and [c.turn for c in r.components] == [0, 1]


def test_a_missing_turn_fails_with_turn_missing() -> None:
    r = only(
        report(
            one_turn("hi"), Expect(decision=Decision.ANSWER, decision_by_turn={3: Decision.ANSWER})
        ),
        "decision_match",
    )
    missing = next(c for c in r.components if c.turn == 3)
    assert missing.reason_code == "decision.turn_missing"


def test_decision_set_accepts_any_listed_label() -> None:
    expect = Expect(decision=(Decision.ASK, Decision.REFUSE))
    assert only(report(one_turn("Which order?"), expect), "decision_match").passed


def test_unknown_min_scores_key_lists_the_valid_ones() -> None:
    with pytest.raises(ExpectConfigError, match="valid keys"):
        report(run_of(), Expect(min_scores={"nope": 0.5}))
