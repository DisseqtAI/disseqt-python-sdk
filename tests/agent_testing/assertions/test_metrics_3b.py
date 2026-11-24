"""Phase 3b metrics: routing, loops, clarification, refusal, scope, intent, plan."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    ExpectConfigError,
    JudgeVerdict,
    evaluate,
)
from disseqt_agent_testing.assertions.predicates import register_predicate, unregister_predicate
from disseqt_agent_testing.models import (
    ClarifyExpect,
    Decision,
    Expect,
    LoopExpect,
    PlanExpect,
    RefuseExpect,
    RouteExpect,
    ScopeExpect,
    ToolInfo,
)

from .helpers import run_of, step


def only(rep: Any, name: str) -> Any:
    got = rep.check(name)
    assert got is not None, [c.name for c in rep.checks]
    return got


class StaticJudge:
    def __init__(self, label: str, status: str = "ok") -> None:
        self.label, self.status = label, status

    def judge(self, task: Any) -> JudgeVerdict:
        return JudgeVerdict(self.status, self.label)  # type: ignore[arg-type]

    async def ajudge(self, task: Any) -> JudgeVerdict:
        return self.judge(task)


# ---------------------------------------------------------------- routing

OFFER = (
    ToolInfo("billing", "subagent", "write"),
    ToolInfo("support", "subagent", "write"),
)


def routed(*targets: str, offered: tuple[ToolInfo, ...] = OFFER) -> Any:
    steps = [step(i + 1, t, kind="subagent") for i, t in enumerate(targets)]
    return run_of(*steps, offered=offered)


def routes(path: tuple[str, ...], mode: str = "in_order") -> Expect:
    return Expect(routes=RouteExpect(path=path, mode=mode))  # type: ignore[arg-type]


def test_r01_routing_by_sub_agent_in_every_mode() -> None:
    run = routed("billing", "support")
    for path, mode, ok in [
        (("billing", "support"), "exact", True), (("billing",), "exact", False),
        (("billing", "support"), "in_order", True), (("support", "billing"), "in_order", False),
        (("billing",), "first", True), (("support",), "first", False),
        (("support", "nothing"), "any", True), (("nothing",), "any", False),
    ]:  # fmt: skip
        got = only(evaluate(run, routes(path, mode)), "routing")
        assert got.passed is ok, (path, mode)
    assert only(evaluate(run_of(offered=OFFER), routes((), "none")), "routing").passed
    assert only(evaluate(run, routes((), "none")), "routing").passed is False  # it did route


def test_a_bare_tuple_means_in_order() -> None:
    assert only(evaluate(routed("billing"), Expect(routes=("billing",))), "routing").passed


def test_r02_nothing_routable_offered_is_unverifiable() -> None:
    rep = evaluate(run_of(text="x"), routes(("billing",)))
    assert only(rep, "routing").reason_code == "data_unavailable" and rep.outcome == "incomplete"


def test_r03_transfer_to_tools_and_the_naming_rule() -> None:
    offered = (ToolInfo("transfer_to_billing_agent", "tool", "write"),)
    run = run_of(step(1, "transfer_to_billing_agent"), offered=offered)
    assert only(evaluate(run, routes(("Billing Agent",))), "routing").passed
    assert only(evaluate(run, routes(("billing",))), "routing").passed


def test_r04_transfer_to_agent_with_an_enum_and_an_invalid_target() -> None:
    schema = {"type": "object", "properties": {"agent_name": {"enum": ["Billing", "Support"]}}}
    offered = (ToolInfo("transfer_to_agent", "tool", "write", schema=schema),)
    good = run_of(step(1, "transfer_to_agent", {"agent_name": "Billing"}), offered=offered)
    assert only(evaluate(good, routes(("billing",))), "routing").passed
    bad = run_of(step(1, "transfer_to_agent", {"agent_name": "Hacker"}), offered=offered)
    r = only(evaluate(bad, routes(("billing",))), "routing")
    assert r.vetoed and r.reason_code == "route.invalid_target" and "not offered" in r.reason


def test_a_sub_agent_that_was_not_offered_is_an_invalid_route() -> None:
    r = only(evaluate(routed("ghost"), routes(("ghost",))), "routing")
    assert r.vetoed and r.reason_code == "route.invalid_target"


# ---------------------------------------------------------------- loops


def loops(**kw: Any) -> Expect:
    return Expect(loops=LoopExpect(**kw))


def seq(*names: str, **per: Any) -> Any:
    return run_of(*(step(i + 1, n, per.get(n, {})) for i, n in enumerate(names)))


def test_l01_too_many_identical_calls() -> None:
    r = only(
        evaluate(seq("a", "b", "a", "c", "a", "d", "a"), loops(max_consecutive_identical=None)),
        "loops",
    )
    assert r.passed is False and r.reason_code == "loop.identical" and "a x4 (limit 3)" in r.reason


def test_l02_l03_argument_hashing() -> None:
    same = run_of(
        step(1, "a", {"x": 1, "y": 2}),
        step(2, "a", {"y": 2, "x": 1}),
        step(3, "a", {"x": 1, "y": 2}),
    )
    assert (
        only(
            evaluate(same, loops(max_identical_calls=2, max_consecutive_identical=None)), "loops"
        ).passed
        is False
    )
    diff = run_of(step(1, "a", {"x": 1}), step(2, "a", {"x": "1"}), step(3, "a", {"x": 1.5}))
    assert only(evaluate(diff, loops(max_identical_calls=1)), "loops").passed


def test_consecutive_runs() -> None:
    r = only(evaluate(seq("a", "a", "a", "b"), loops(max_identical_calls=None)), "loops")
    assert r.reason_code == "loop.consecutive"
    assert only(evaluate(seq("a", "a", "b"), loops(max_identical_calls=None)), "loops").passed


def test_l04_cycles_of_period_two_and_not_aperiodic_mixes() -> None:
    cyc = only(
        evaluate(
            seq("a", "b", "a", "b"), loops(max_identical_calls=None, max_consecutive_identical=None)
        ),
        "loops",
    )
    assert cyc.reason_code == "loop.cycle" and "['a', 'b']" in cyc.reason
    assert only(
        evaluate(
            seq("a", "b", "a", "c"), loops(max_identical_calls=None, max_consecutive_identical=None)
        ),
        "loops",
    ).passed
    off = loops(max_identical_calls=None, max_consecutive_identical=None, detect_cycles=False)
    assert only(evaluate(seq("a", "b", "a", "b"), off), "loops").passed


def test_l05_an_identical_call_right_after_an_error_is_a_retry_not_a_loop() -> None:
    run = run_of(step(1, "a", error="boom"), step(2, "a", error="boom"), step(3, "a"))
    assert only(
        evaluate(run, loops(max_identical_calls=1, max_consecutive_identical=1)), "loops"
    ).passed
    capped = loops(max_identical_calls=1, max_consecutive_identical=1, max_retries_after_error=0)
    assert only(evaluate(run, capped), "loops").passed is False


def test_l06_pagination_is_flagged_with_key_name_by_design() -> None:
    pages = run_of(*(step(i, "list", {"page": i}, effect="read") for i in (1, 2, 3, 4)))
    assert only(evaluate(pages, loops(max_consecutive_identical=None)), "loops").passed
    r = only(evaluate(pages, loops(key="name", max_consecutive_identical=None)), "loops")
    assert r.passed is False and r.reason_code == "loop.identical"


def test_l07_an_empty_trajectory_passes() -> None:
    assert only(evaluate(run_of(text="hi"), loops()), "loops").passed


def test_loops_still_score_a_crashed_run() -> None:
    from .helpers import result_of

    res = result_of(
        run_of(*(step(i, "a") for i in range(1, 6)), text=None, outputs=()), status="crashed"
    )
    assert only(evaluate(res, loops()), "loops").passed is False


# ---------------------------------------------------------------- clarification


def ask_run(text: str, *steps: Any) -> Any:
    return run_of(*steps, text=text)


def test_c01_gates_in_order() -> None:
    expect = Expect(
        decision=Decision.ASK, clarify=ClarifyExpect(must_mention_any=("order number",))
    )
    ok = only(evaluate(ask_run("Which order number is this about?"), expect), "clarification")
    assert ok.passed
    blocked = evaluate(ask_run("Your refund is done."), expect)  # decision_match owns this failure
    assert only(blocked, "clarification").reason_code == "blocked_by:decision_match"
    only_clarify = Expect(clarify=ClarifyExpect(must_mention_any=("order number",)))
    answered = only(evaluate(ask_run("Your refund is done."), only_clarify), "clarification")
    assert answered.components[0].reason_code == "clarify.not_asked"
    acted = ask_run("Which order number?", step(1, "issue_refund"))
    # a write call makes the turn a "call", so the classification gate fires first
    assert (
        only(evaluate(acted, Expect(clarify=ClarifyExpect())), "clarification").reason_code
        == "clarify.not_asked"
    )
    read_ok = ask_run("Which order number?", step(1, "lookup", effect="read"))
    assert only(evaluate(read_ok, Expect(clarify=ClarifyExpect())), "clarification").passed
    strict = only(
        evaluate(read_ok, Expect(clarify=ClarifyExpect(allow_read_calls_before=False))),
        "clarification",
    )
    assert strict.passed is False and strict.reason_code == "clarify.acted_before_asking"


def test_c02_slot_phrases_match_after_normalization() -> None:
    run = ask_run("Which order_number do you mean?")
    spec = ClarifyExpect(must_mention_all=("order number", "which"))
    assert only(evaluate(run, Expect(clarify=spec)), "clarification").passed
    miss = ClarifyExpect(must_mention_any=("email", "phone"))
    r = only(evaluate(run, Expect(clarify=miss)), "clarification")
    assert r.components[0].reason_code == "clarify.slot_missing"
    r2 = only(
        evaluate(run, Expect(clarify=ClarifyExpect(must_mention_all=("order number", "email")))),
        "clarification",
    )
    assert "email" in r2.components[0].details["missing"]


def test_c03_judged_relevance_without_a_judge_is_incomplete() -> None:
    spec = ClarifyExpect(judge_relevance=True)
    rep = evaluate(ask_run("Which order number?"), Expect(clarify=spec))
    assert (
        rep.outcome == "incomplete"
        and only(rep, "clarification").reason_code == "judge_unavailable"
    )
    ok = evaluate(
        ask_run("Which order number?"),
        Expect(clarify=spec),
        judge=StaticJudge("pass"),
        use_judge=True,
    )
    assert ok.outcome == "passed" and only(ok, "clarification").judged
    bad = evaluate(
        ask_run("Which order number?"),
        Expect(clarify=spec),
        judge=StaticJudge("fail"),
        use_judge=True,
    )
    assert only(bad, "clarification").reason_code == "clarify.irrelevant_question"
    na = evaluate(
        ask_run("Which order number?"),
        Expect(clarify=spec),
        judge=StaticJudge("not_applicable"),
        use_judge=True,
    )
    assert only(na, "clarification").reason_code == "judge_unsure"


def test_c04_max_questions() -> None:
    run = ask_run("Which order? What is the amount? Is it urgent?")
    r = only(evaluate(run, Expect(clarify=ClarifyExpect(max_questions=1))), "clarification")
    assert r.components[0].reason_code == "clarify.too_many_questions"
    assert only(
        evaluate(run, Expect(clarify=ClarifyExpect(max_questions=3))), "clarification"
    ).passed


def test_an_expected_ask_enables_clarification_without_clarify() -> None:
    rep = evaluate(ask_run("Which order?"), Expect(decision=Decision.ASK))
    assert only(rep, "clarification").passed


# ---------------------------------------------------------------- refusal / scope / intent


def test_refusal_gates() -> None:
    refused = ask_run("I'm sorry, I can't help with that request.")
    assert only(evaluate(refused, Expect(refuse=RefuseExpect())), "refusal").passed
    r = only(evaluate(ask_run("Sure, done."), Expect(refuse=RefuseExpect())), "refusal")
    assert r.reason_code == "refuse.not_refused"
    phrase = only(
        evaluate(refused, Expect(refuse=RefuseExpect(must_mention_any=("policy",)))), "refusal"
    )
    assert phrase.reason_code == "refuse.phrase_missing"
    ok = only(
        evaluate(refused, Expect(refuse=RefuseExpect(must_mention_any=("can't help",)))), "refusal"
    )
    assert ok.passed


def test_refusal_alternative_is_judge_only_with_polarity() -> None:
    run = ask_run("I'm sorry, I can't help with that request.")
    wants = Expect(refuse=RefuseExpect(offer_alternative=True))
    assert evaluate(run, wants).outcome == "incomplete"
    assert (
        only(evaluate(run, wants, judge=StaticJudge("fail"), use_judge=True), "refusal").reason_code
        == "refuse.no_alternative"
    )
    assert only(evaluate(run, wants, judge=StaticJudge("pass"), use_judge=True), "refusal").passed
    no_alt = Expect(refuse=RefuseExpect(offer_alternative=False))
    r = only(evaluate(run, no_alt, judge=StaticJudge("pass"), use_judge=True), "refusal")
    assert r.reason_code == "refuse.unexpected_alternative"


def test_c05_scope_confusion_cells() -> None:
    refused = ask_run("I'm sorry, I can't help with that request.")
    answered = ask_run("Here is your answer.")
    cells = {
        (True, "answered"): "TP", (True, "refused"): "FN", (False, "refused"): "TN", (False, "answered"): "FP",
    }  # fmt: skip
    for (in_scope, kind), cell in cells.items():
        run = refused if kind == "refused" else answered
        r = only(evaluate(run, Expect(scope=ScopeExpect(in_scope))), "scope")
        assert r.details["cell"] == cell and r.passed is (cell in ("TP", "TN"))
        if cell == "FN":
            assert r.reason_code == "scope.over_refusal"
        if cell == "FP":
            assert r.reason_code == "scope.under_refusal"


def test_scope_empty_output_fails() -> None:
    r = only(evaluate(run_of(text=None, outputs=(None,)), Expect(scope=ScopeExpect(True))), "scope")
    assert r.reason_code == "decision.empty"


def test_c08_intent_from_call_arguments_results_answer_and_route() -> None:
    run = run_of(step(1, "classify", {"label": "refund"}, result={"intent": "Refund"}), text="done")
    for source, want, ok in [
        ("call:classify.args.label", "refund", True), ("call:classify.args.label", "cancel", False),
        ("call:classify.result.intent", "refund", True),  # compared case-insensitively
    ]:  # fmt: skip
        r = only(evaluate(run, Expect(intent=want, intent_source=source)), "intent")
        assert r.passed is ok, (source, want)
    ans = run_of(text=None, outputs=({"intent": "refund"},))
    ans = run_of(text=None, outputs=({"intent": "refund"},))
    from disseqt_agent_testing.models import AgentRun

    with_answer = AgentRun(
        steps=(),
        input="x",
        final_answer={"intent": "refund"},
        final_text='{"intent": "refund"}',
        turn_outputs=({"intent": "refund"},),
    )
    assert only(
        evaluate(with_answer, Expect(intent="refund", intent_source="answer.intent")), "intent"
    ).passed
    assert ans is not None
    routed_run = routed("billing")
    assert only(
        evaluate(routed_run, Expect(intent="Billing", intent_source="route")), "intent"
    ).passed


def test_c08_an_unresolvable_intent_source_fails_as_unresolved() -> None:
    r = only(
        evaluate(run_of(), Expect(intent="refund", intent_source="call:nothing.args.x")), "intent"
    )
    assert r.reason_code == "intent.unresolved" and r.passed is False
    mism = only(
        evaluate(
            run_of(step(1, "c", {"x": "a"})), Expect(intent="b", intent_source="call:c.args.x")
        ),
        "intent",
    )
    assert mism.reason_code == "intent.mismatch"


def test_intent_without_a_source_needs_a_judge() -> None:
    run = run_of(text="hi", input="I want my money back")
    expect = Expect(intent="refund", intent_labels=("refund", "cancel"))
    assert evaluate(run, expect).outcome == "incomplete"
    assert evaluate(run, expect, judge=StaticJudge("refund"), use_judge=True).outcome == "passed"
    wrong = evaluate(run, expect, judge=StaticJudge("cancel"), use_judge=True)
    assert only(wrong, "intent").reason_code == "intent.mismatch"


# ---------------------------------------------------------------- plan


def test_p01_constraints_micro_and_macro() -> None:
    register_predicate("no_deletes", lambda run: all(s["name"] != "delete" for s in run["steps"]))
    register_predicate("short", lambda run: (len(run["steps"]) <= 1, "too many steps"))
    try:
        run = run_of(step(1, "a"), step(2, "b"))
        r = only(
            evaluate(run, Expect(plan=PlanExpect(constraints=("no_deletes", "short")))), "plan"
        )
        assert r.passed is False and r.reason_code == "plan.constraint_failed"
        assert r.sub_scores["micro"] == 0.5 and r.sub_scores["macro"] == 0.0
        assert "short failed too many steps" in r.reason
        ok = only(evaluate(run, Expect(plan=PlanExpect(constraints=("no_deletes",)))), "plan")
        assert ok.passed and ok.sub_scores["macro"] == 1.0
    finally:
        unregister_predicate("no_deletes")
        unregister_predicate("short")


def test_p02_a_raising_constraint_is_an_instrument_error() -> None:
    register_predicate("explodes", lambda run: 1 / 0)
    try:
        rep = evaluate(run_of(), Expect(plan=PlanExpect(constraints=("explodes",))))
        assert only(rep, "plan").status == "error" and rep.outcome == "incomplete"
    finally:
        unregister_predicate("explodes")


def test_p03_convergence_is_informational_unless_thresholded() -> None:
    run = run_of(*(step(i, "a") for i in (1, 2, 3, 4)))
    plan = PlanExpect(optimal_tool_calls=2)
    r = only(evaluate(run, Expect(plan=plan, extra_calls="allow")), "plan")
    assert r.passed and r.sub_scores["convergence"] == 0.5
    gated = evaluate(run, Expect(plan=plan, min_scores={"convergence": 0.9}))
    assert only(gated, "plan").passed is False and gated.outcome == "failed"
    with pytest.raises(ExpectConfigError, match="optimal_tool_calls"):
        evaluate(run, Expect(plan=PlanExpect(), min_scores={"plan.convergence": 0.5}))


def test_unknown_plan_constraint_is_a_config_error() -> None:
    with pytest.raises(ExpectConfigError, match="unknown predicate"):
        evaluate(run_of(), Expect(plan=PlanExpect(constraints=("nope",))))


def test_judged_expectations_without_a_judge_are_skipped_never_passed() -> None:
    from disseqt_agent_testing.models import AnswerQualityExpect, PlanExpect

    for expect, name in (
        (Expect(answer_quality=AnswerQualityExpect("ref")), "answer_quality"),
        (Expect(grounded=True), "groundedness"),
        (Expect(plan=PlanExpect(judge_quality=True)), "plan"),
    ):
        report = evaluate(run_of(text="an answer"), expect)
        check = only(report, name)
        assert check.status == "skipped" and check.passed is None, name
        assert report.outcome == "incomplete", name


def test_first_mode_compares_only_the_first_observed_route_to_the_first_listed() -> None:
    run = routed("billing", "support")
    assert only(evaluate(run, routes(("billing", "support"), "first")), "routing").passed
    assert only(evaluate(run, routes(("support", "billing"), "first")), "routing").passed is False


def test_unknown_predicates_are_caught_even_when_the_matcher_would_never_run() -> None:
    from disseqt_agent_testing.models import Predicate, ToolExpect

    expect = Expect(tools=(ToolExpect("never_called", {"a": Predicate("ghost")}),))
    with pytest.raises(ExpectConfigError, match="unknown predicate 'ghost'"):
        evaluate(run_of(text="no calls at all"), expect)
