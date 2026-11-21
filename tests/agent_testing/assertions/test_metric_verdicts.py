"""Verdict rules of the built-in metrics: a check that cannot be measured is incomplete, never a
pass, and a valid run is never failed for a constraint that does not apply to it."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import evaluate
from disseqt_agent_testing.assertions.judge import JudgeVerdict
from disseqt_agent_testing.models import (
    ClarifyExpect,
    Decision,
    Expect,
    ForbiddenCall,
    PlanExpect,
    RefuseExpect,
    ScopeExpect,
    ToolExpect,
)

from .helpers import run_of, step


def only(rep: Any, name: str) -> Any:
    got = rep.check(name)
    assert got is not None, [c.name for c in rep.checks]
    return got


class StaticJudge:
    def __init__(self, label: str | None, status: str = "ok") -> None:
        self.label, self.status = label, status

    def judge(self, task: Any) -> JudgeVerdict:
        return JudgeVerdict(self.status, self.label)  # type: ignore[arg-type]

    async def ajudge(self, task: Any) -> JudgeVerdict:
        return self.judge(task)


FAILED = StaticJudge(None, "error")
UNSURE = StaticJudge("unsure")
UNDECIDED = pytest.mark.parametrize(
    ("judge", "status", "code"),
    [(FAILED, "error", "judge_failed"), (UNSURE, "skipped", "judge_unsure")],
    ids=["judge_failed", "judge_unsure"],
)
# a refusal phrase plus a question is the classifier's low-confidence case, so it asks the judge
AMBIGUOUS = "I'm sorry, I can't help with that request. Which order number is it?"


# ---------------------------------------------------------------- thresholds


def test_a_recall_threshold_does_not_clear_an_unexpected_write_call() -> None:
    run = run_of(step(1, "get_order"), step(2, "delete_account"))
    expect = Expect(tools=(ToolExpect("get_order"),))
    assert only(evaluate(run, expect), "tool_selection").reason_code == "tool.unexpected"
    rep = evaluate(run, Expect(tools=(ToolExpect("get_order"),), min_scores={"tool_recall": 1.0}))
    r = only(rep, "tool_selection")
    assert r.passed is False and r.reason_code == "tool.unexpected" and rep.outcome == "failed"


def test_a_sub_score_threshold_still_replaces_the_bar_for_its_own_dimension() -> None:
    tools = (ToolExpect("a"), ToolExpect("b"))
    missing = run_of(step(1, "a"))
    assert only(evaluate(missing, Expect(tools=tools)), "tool_selection").passed is False
    relaxed = evaluate(missing, Expect(tools=tools, min_scores={"tool_recall": 0.5}))
    assert only(relaxed, "tool_selection").passed is True
    extra = run_of(step(1, "a"), step(2, "b"), step(3, "w"))
    by_precision = evaluate(extra, Expect(tools=tools, min_scores={"tool_precision": 0.6}))
    assert only(by_precision, "tool_selection").passed is True
    by_f1 = evaluate(run_of(step(1, "a"), step(2, "w")), Expect(tools=tools, min_scores={"tool_f1": 0.5}))  # fmt: skip
    assert only(by_f1, "tool_selection").passed is True


def test_a_threshold_must_cover_every_failing_dimension() -> None:
    run = run_of(step(1, "a"), step(2, "w"))  # b is missing and w is unexpected
    tools = (ToolExpect("a"), ToolExpect("b"))
    r = only(evaluate(run, Expect(tools=tools, min_scores={"tool_recall": 0.5})), "tool_selection")
    assert r.passed is False and r.reason_code == "tool.missing"
    both = {"tool_recall": 0.5, "tool_precision": 0.5}
    assert only(evaluate(run, Expect(tools=tools, min_scores=both)), "tool_selection").passed


def test_an_arg_score_threshold_still_relaxes_wrong_arguments() -> None:
    run = run_of(step(1, "a", {"x": 1, "y": 9}))
    tools = (ToolExpect("a", args={"x": 1, "y": 2}),)
    assert only(evaluate(run, Expect(tools=tools)), "tool_args").passed is False
    relaxed = evaluate(run, Expect(tools=tools, min_scores={"arg_score": 0.5}))
    assert only(relaxed, "tool_args").passed is True


# ---------------------------------------------------------------- undecided decision judge


@UNDECIDED
def test_refusal_is_incomplete_when_the_decision_judge_did_not_decide(
    judge: StaticJudge, status: str, code: str
) -> None:
    rep = evaluate(run_of(text=AMBIGUOUS), Expect(refuse=RefuseExpect()), judge=judge, use_judge=True)  # fmt: skip
    r = only(rep, "refusal")
    assert (r.status, r.reason_code, r.passed) == (status, code, None)
    assert rep.outcome == "incomplete"


@UNDECIDED
def test_scope_is_incomplete_when_the_decision_judge_did_not_decide(
    judge: StaticJudge, status: str, code: str
) -> None:
    for in_scope in (True, False):
        rep = evaluate(
            run_of(text=AMBIGUOUS), Expect(scope=ScopeExpect(in_scope)), judge=judge, use_judge=True
        )
        r = only(rep, "scope")
        assert (r.status, r.reason_code, r.passed) == (status, code, None)
        assert rep.outcome == "incomplete"


@UNDECIDED
def test_clarification_is_incomplete_when_the_decision_judge_did_not_decide(
    judge: StaticJudge, status: str, code: str
) -> None:
    rep = evaluate(
        run_of(text="Which order number is this about?"),
        Expect(clarify=ClarifyExpect()),
        judge=judge,
        use_judge=True,
        judge_confidence_threshold=0.9,
    )
    r = only(rep, "clarification")
    assert (r.status, r.reason_code, r.passed) == (status, code, None)
    assert r.components[0].turn == 0 and rep.outcome == "incomplete"


@UNDECIDED
def test_decision_family_agrees_with_decision_match(
    judge: StaticJudge, status: str, code: str
) -> None:
    expect = Expect(decision=Decision.REFUSE, refuse=RefuseExpect(), scope=ScopeExpect(False))
    rep = evaluate(run_of(text=AMBIGUOUS), expect, judge=judge, use_judge=True)
    got = {n: (only(rep, n).status, only(rep, n).reason_code) for n in ("refusal", "scope")}
    assert only(rep, "decision_match").reason_code == code
    assert got == {"refusal": (status, code), "scope": (status, code)}


def test_a_decided_judge_label_still_drives_refusal_and_scope() -> None:
    judge = StaticJudge("refuse")
    rep = evaluate(run_of(text=AMBIGUOUS), Expect(refuse=RefuseExpect()), judge=judge, use_judge=True)  # fmt: skip
    assert only(rep, "refusal").passed is True
    asked = evaluate(
        run_of(text=AMBIGUOUS), Expect(refuse=RefuseExpect()), judge=StaticJudge("ask"), use_judge=True
    )  # fmt: skip
    assert only(asked, "refusal").reason_code == "refuse.not_refused"


# ---------------------------------------------------------------- clarification


def test_the_clarify_spec_applies_to_turn_zero_only() -> None:
    run = run_of(
        text="And which item should I refund?",
        outputs=("What is your order number?", "And which item should I refund?"),
        follow_ups=("A12",),
    )
    expect = Expect(
        clarify=ClarifyExpect(must_mention_any=("order number",), max_questions=1),
        decision=Decision.ASK,
        decision_by_turn={1: Decision.ASK},
    )
    r = only(evaluate(run, expect), "clarification")
    assert r.passed is True, [(c.turn, c.reason_code) for c in r.components]
    wrong_first = run_of(
        text="And which item should I refund?",
        outputs=("And which item should I refund?", "What is your order number?"),
        follow_ups=("A12",),
    )
    bad = only(evaluate(wrong_first, expect), "clarification")
    assert bad.passed is False and bad.components[0].reason_code == "clarify.slot_missing"


# ---------------------------------------------------------------- order


def edge_expect(*tools: ToolExpect) -> Expect:
    return Expect(tools=tools, order="partial", extra_calls="allow")


def test_a_concurrent_pair_is_same_batch_even_with_a_later_repeat() -> None:
    expect = edge_expect(ToolExpect("a", id="a"), ToolExpect("b", count=(1, 2), after=("a",)))
    pair = (step(1, "a", parallel_group=1), step(2, "b", parallel_group=1))
    alone = only(evaluate(run_of(*pair), expect), "order")
    assert alone.passed is False and alone.reason_code == "order.edge_same_batch"
    repeated = only(evaluate(run_of(*pair, step(3, "b")), expect), "order")
    assert repeated.passed is False and repeated.reason_code == "order.edge_same_batch"


def test_an_edge_is_satisfied_when_the_first_calls_are_in_different_batches() -> None:
    expect = edge_expect(ToolExpect("a", id="a"), ToolExpect("b", count=(1, 2), after=("a",)))
    run = run_of(step(1, "a"), step(2, "b", parallel_group=1), step(3, "b", parallel_group=1))
    assert only(evaluate(run, expect), "order").passed is True


def test_exact_order_does_not_require_optional_slots() -> None:
    expect = Expect(tools=(ToolExpect("search", count=(1, 2)), ToolExpect("pay")), order="exact")
    once = only(evaluate(run_of(step(1, "search"), step(2, "pay")), expect), "order")
    assert once.passed is True, once.reason
    twice = run_of(step(1, "search"), step(2, "search"), step(3, "pay"))
    assert only(evaluate(twice, expect), "order").passed is True
    swapped = only(evaluate(run_of(step(1, "pay"), step(2, "search")), expect), "order")
    assert swapped.passed is False and swapped.reason_code == "order.exact"
    split = run_of(step(1, "search"), step(2, "pay"), step(3, "search"))
    assert only(evaluate(split, expect), "order").passed is False
    extra = run_of(step(1, "search"), step(2, "other"), step(3, "pay"))
    assert only(evaluate(extra, expect), "order").passed is False
    assert only(evaluate(run_of(step(1, "search")), expect), "order").passed is False


def test_an_after_edge_on_an_optional_call_that_never_happened_is_not_checked() -> None:
    expect = edge_expect(ToolExpect("a", id="a"), ToolExpect("notify", count=(0, 1), after=("a",)))
    rep = evaluate(run_of(step(1, "a")), expect)
    r = only(rep, "order")
    assert r.status == "skipped" and r.reason_code == "not_applicable"
    assert rep.outcome == "passed"
    happened = only(evaluate(run_of(step(1, "notify"), step(2, "a")), expect), "order")
    assert happened.passed is False and happened.reason_code == "order.edge_violated"


def test_an_after_edge_on_a_missing_required_call_is_still_unverifiable() -> None:
    expect = edge_expect(ToolExpect("a", id="a"), ToolExpect("notify", after=("a",)))
    r = only(evaluate(run_of(step(1, "a")), expect), "order")
    assert r.passed is False and r.reason_code == "order.edge_unverifiable"
    optional_first = edge_expect(
        ToolExpect("a", id="a", count=(0, 1)), ToolExpect("notify", after=("a",))
    )
    skipped = only(evaluate(run_of(step(1, "notify")), optional_first), "order")
    assert skipped.status == "skipped" and skipped.reason_code == "not_applicable"


# ---------------------------------------------------------------- budgets


def test_max_steps_without_model_calls_is_incomplete_unless_already_over() -> None:
    two = (step(1, "a"), step(2, "b"))
    rep = evaluate(run_of(*two), Expect(max_steps=2))
    r = only(rep, "budgets")
    assert r.status == "skipped" and r.reason_code == "data_unavailable"
    assert rep.outcome == "incomplete"
    over = evaluate(run_of(*two, step(3, "c")), Expect(max_steps=2))
    bad = only(over, "budgets")
    assert bad.passed is False and bad.reason_code == "budget.exceeded"
    assert "lower bound" in bad.reason and over.outcome == "failed"
    captured = evaluate(run_of(*two, captured=True), Expect(max_steps=2))
    assert only(captured, "budgets").passed is True and captured.outcome == "passed"


# ---------------------------------------------------------------- plan


def test_convergence_is_unmeasured_for_a_run_with_no_calls() -> None:
    plan = PlanExpect(optimal_tool_calls=3)
    r = only(evaluate(run_of(), Expect(plan=plan)), "plan")
    assert r.sub_scores["convergence"] is None
    rep = evaluate(run_of(), Expect(plan=plan, min_scores={"plan.convergence": 0.9}))
    gated = only(rep, "plan")
    assert gated.passed is not True and gated.reason_code == "data_unavailable"
    assert rep.outcome == "incomplete"
    assert "plan.convergence" not in rep.scores


# ---------------------------------------------------------------- forbidden_calls


def guarded(*steps: Any, tools: tuple[ToolExpect, ...] | None = None) -> Any:
    expect = Expect(
        tools=tools or (ToolExpect("verify", id="verify"),),
        forbidden=(ForbiddenCall("refund", unless_after="verify"),),
        extra_calls="allow",
    )
    rep = evaluate(run_of(*steps), expect)
    bound = [m["call"] for m in only(rep, "tool_selection").details["matched"]]
    return only(rep, "forbidden_calls"), bound


def test_unless_after_counts_an_earlier_prerequisite_the_assignment_did_not_bind() -> None:
    tools = (ToolExpect("lookup"), ToolExpect("verify", id="verify"))
    r, bound = guarded(
        step(1, "verify"), step(2, "refund"), step(3, "lookup"), step(4, "verify"), tools=tools
    )
    assert len([c for c in bound if c in ("call-1", "call-4")]) == 1  # one verify is unbound
    assert r.passed is True, r.reason


def test_unless_after_an_unbound_prerequisite_must_still_qualify() -> None:
    tools = (ToolExpect("lookup"), ToolExpect("verify", id="verify"))
    tail = (step(3, "lookup"), step(4, "verify"))
    failed, _ = guarded(step(1, "verify", error="denied"), step(2, "refund"), *tail, tools=tools)
    assert failed.passed is False and failed.reason_code == "forbidden.called"
    concurrent, _ = guarded(
        step(1, "verify", parallel_group=7), step(2, "refund", parallel_group=7), *tail, tools=tools
    )
    assert concurrent.passed is False
    with_args = (ToolExpect("lookup"), ToolExpect("verify", {"user": "u1"}, id="verify"))
    other_user, _ = guarded(
        step(1, "verify", {"user": "u2"}),
        step(2, "refund"),
        step(3, "lookup"),
        step(4, "verify", {"user": "u1"}),
        tools=with_args,
    )
    assert other_user.passed is False  # the earlier call does not satisfy the prerequisite
    later_only, _ = guarded(step(1, "refund"), step(2, "verify"))
    assert later_only.passed is False
