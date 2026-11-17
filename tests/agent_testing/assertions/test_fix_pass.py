"""Regression tests for the assertion-engine review fixes (fail-open paths and small hardening)."""

from typing import Any

from disseqt_agent_testing.assertions import evaluate
from disseqt_agent_testing.assertions.matching import MatchContext, match_value
from disseqt_agent_testing.assertions.metrics.base import MetricResult, rollup
from disseqt_agent_testing.models import (
    Decision,
    Expect,
    ForbiddenCall,
    Predicate,
    Regex,
    Semantic,
    ToolExpect,
)

from .helpers import run_of, step


def _only(rep: Any, name: str) -> Any:
    got = rep.check(name)
    assert got is not None
    return got


# ------------------------------------------------------------ 1. rollup is not fail-open


def test_rollup_unmeasurable_child_blocks_a_pass_but_benign_skips_do_not() -> None:
    ok = MetricResult("a", status="ok", value=1.0, passed=True)
    bad = MetricResult("b", status="ok", value=0.0, passed=False)
    unavailable = MetricResult("c", status="skipped", reason_code="data_unavailable")
    unsure = MetricResult("u", status="skipped", reason_code="judge_unsure")
    benign = MetricResult("n", status="skipped", reason_code="not_applicable")
    err = MetricResult("e", status="error", reason_code="metric_error")
    assert rollup([ok, unavailable])[0::3] == ("skipped", "data_unavailable")
    assert rollup([ok, unsure])[0::3] == ("skipped", "judge_unsure")
    assert rollup([ok, benign])[0::2] == ("ok", True)
    assert rollup([ok, bad, unavailable])[0::2] == ("ok", False)  # failure dominates
    assert rollup([ok, err, unavailable])[0] == "error"


def test_budgets_with_an_uncaptured_token_budget_is_incomplete_not_passed() -> None:
    rep = evaluate(run_of(step(1, "t")), Expect(max_steps=5, max_tokens=10))
    r = _only(rep, "budgets")
    assert r.status == "skipped" and r.reason_code == "data_unavailable"
    assert rep.outcome == "incomplete"


def test_in_order_plus_same_step_without_groups_is_incomplete() -> None:
    tools = (ToolExpect("a", id="a"), ToolExpect("b", same_step_as="a"))
    run = run_of(step(1, "a"), step(2, "b"))
    rep = evaluate(run, Expect(tools=tools, order="in_order"))
    assert _only(rep, "order").status == "skipped"
    assert rep.outcome == "incomplete"


# ------------------------------------------------------------ 2. answer matchers tri-state


def _boom(value: Any) -> bool:
    raise RuntimeError("boom")


def test_answer_excludes_with_a_raising_predicate_is_an_error_not_a_pass() -> None:
    rep = evaluate(
        run_of(text="hello"), Expect(answer_excludes=(Predicate("boom"),)),
        predicates={"boom": _boom},
    )  # fmt: skip
    r = _only(rep, "answer")
    assert r.status == "error" and r.reason_code == "predicate_error"
    assert rep.outcome == "incomplete"


def test_answer_excludes_and_contains_semantic_without_a_judge_are_incomplete() -> None:
    for expect in (
        Expect(answer_excludes=(Semantic("a refund"),)),
        Expect(answer_contains=(Semantic("a refund"),)),
    ):
        rep = evaluate(run_of(text="hello"), expect)
        r = _only(rep, "answer")
        assert r.status == "skipped" and r.reason_code == "judge_unavailable", expect
        assert rep.outcome == "incomplete"


def test_answer_a_real_failure_still_dominates_an_undecidable_matcher() -> None:
    rep = evaluate(
        run_of(text="hello"), Expect(answer_contains=("zzz",), answer_excludes=(Semantic("x"),))
    )
    r = _only(rep, "answer")
    assert r.passed is False and r.reason_code == "answer.missing_text"


# ------------------------------------------------------------ 3. forbidden unless_after


def _guarded(prereq_step: Any, hit_step: Any) -> Any:
    expect = Expect(
        decision=Decision.CALL,
        tools=(ToolExpect("verify", id="verify"),),
        forbidden=(ForbiddenCall("refund", unless_after="verify"),),
        extra_calls="allow",
    )
    return _only(evaluate(run_of(prereq_step, hit_step), expect), "forbidden_calls")


def test_unless_after_a_failed_prerequisite_does_not_exempt() -> None:
    r = _guarded(step(1, "verify", error="denied"), step(2, "refund"))
    assert r.passed is False and r.reason_code == "forbidden.called"
    assert _guarded(step(1, "verify"), step(2, "refund")).passed is True


def test_unless_after_a_concurrent_prerequisite_does_not_exempt() -> None:
    same = _guarded(step(1, "verify", parallel_group=7), step(2, "refund", parallel_group=7))
    assert same.passed is False
    other = _guarded(step(1, "verify", parallel_group=7), step(2, "refund", parallel_group=8))
    assert other.passed is True
    none_groups = _guarded(step(1, "verify"), step(2, "refund"))
    assert none_groups.passed is True  # parallel_group None counts as distinct


# ------------------------------------------------------------ 4. thresholds


def test_min_scores_cannot_clear_an_extra_calls_forbid_failure() -> None:
    expect = Expect(
        tools=(ToolExpect("a"),), extra_calls="forbid", min_scores={"tool_selection": 0.0}
    )
    rep = evaluate(run_of(step(1, "a"), step(2, "other")), expect)
    r = _only(rep, "tool_selection")
    assert r.vetoed and r.passed is False and rep.outcome == "failed"


def test_a_sub_score_threshold_on_an_absent_sub_score_is_incomplete() -> None:
    expect = Expect(
        tools=(ToolExpect("a", count=(0, 1)),),
        extra_calls="allow",
        min_scores={"tool_selection.precision": 0.9},
    )
    rep = evaluate(run_of(), expect)
    r = _only(rep, "tool_selection")
    assert r.status == "skipped" and r.reason_code == "data_unavailable"
    assert rep.outcome == "incomplete"


def test_an_unmeasurable_threshold_cannot_turn_a_failed_check_into_a_pass() -> None:
    """Found in the fix-pass re-review: with a base failure (an unexpected call) and a threshold on
    a sub-score that is not measured (recall, no required tools), the threshold step used to
    overwrite the failure with a pass. A threshold that cannot be evaluated must leave a failure
    a failure."""
    run = run_of(step(1, "a"))
    plain = evaluate(run, Expect(allowed_tools=("a",)))
    assert plain.outcome == "failed", "precondition: the base check fails"
    thr = evaluate(run, Expect(allowed_tools=("a",), min_scores={"tool_recall": 1.0}))
    assert thr.outcome == "failed", thr.outcome


# ------------------------------------------------------------ 5. regex bound


def test_regex_search_is_bounded_on_huge_text() -> None:
    import time

    text = "a" * 100_000 + "X" + "a" * 50_000
    t0 = time.perf_counter()
    got = match_value(Regex(r"X"), text, MatchContext(), ())[0]
    assert got.status == "ok" and got.ok is False  # beyond the cap: not searched
    assert "truncated" in str(dict(got.details))
    assert time.perf_counter() - t0 < 2
    head = match_value(Regex(r"^a+"), "a" * 200_000, MatchContext(), ())[0]
    assert head.ok is True


# ------------------------------------------------------------ 6. intent resolution


def test_intent_null_value_does_not_match_none_and_later_calls_are_considered() -> None:
    run = run_of(
        step(1, "classify", {"label": None}), step(2, "classify", {"label": "refund"}), text="x"
    )
    r = _only(
        evaluate(run, Expect(intent="refund", intent_source="call:classify.args.label")), "intent"
    )
    assert r.passed is True
    only_null = run_of(step(1, "classify", {"label": None}), text="x")
    r2 = _only(
        evaluate(only_null, Expect(intent="None", intent_source="call:classify.args.label")),
        "intent",
    )
    assert r2.passed is False and r2.reason_code == "intent.unresolved"


# ------------------------------------------------------------ 7. uncalibrated warning text


def test_uncalibrated_warning_names_the_judge_kind_and_label() -> None:
    from disseqt_agent_testing.assertions import JudgeVerdict
    from disseqt_agent_testing.models import AnswerQualityExpect

    class Advisory:
        def judge(self, task: Any) -> JudgeVerdict:
            return JudgeVerdict("ok", "pass", gating="advisory")

    rep = evaluate(
        run_of(text="hello"),
        Expect(answer_quality=AnswerQualityExpect("hello")),
        judge=Advisory(),
        use_judge=True,
    )
    text = " ".join(rep.warnings)
    assert "uncalibrated" in text and "?" not in text, rep.warnings
    assert "answer_quality" in text and "pass" in text
