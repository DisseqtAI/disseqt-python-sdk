"""answer_quality, groundedness and plan.judge_quality through ``evaluate`` and the real engine."""

from typing import Any

import pytest

from disseqt_agent_testing.assertions import EvalConfig, evaluate
from disseqt_agent_testing.models import AnswerQualityExpect, Expect, GroundednessExpect, PlanExpect

from ..assertions.helpers import run_of, step
from .helpers import engine as _engine
from .test_service_and_quality import ONE, _answer, _ground, _plan

pytestmark = pytest.mark.filterwarnings("ignore::disseqt_agent_testing.judges.api.JudgeWarning")
CFG = EvalConfig(use_judge=True)


def run_with(script: Any, expect: Expect, run: Any = None, **options: Any) -> Any:
    options.setdefault("require_calibration", "off")
    config = options.pop("config", CFG)
    with _engine(script, kinds=ONE, **options) as (eng, model):
        report = evaluate(run or run_of(text="Refund issued"), expect, judge=eng, config=config)
    return report, model


def chk(report: Any, name: str) -> Any:
    got = report.check(name)
    assert got is not None, [c.name for c in report.checks]
    return got


def test_answer_quality_prepends_the_reference_as_criterion_one() -> None:
    expect = Expect(answer_quality=AnswerQualityExpect("A refund of 10", criteria=("polite",)))
    report, model = run_with([_answer([True, True])], expect)
    assert chk(report, "answer_quality").passed is True and chk(report, "answer_quality").judged
    user = model.calls[0].messages[-1]["content"]
    assert "A refund of 10" in user and "polite" in user


def test_answer_quality_accept_levels() -> None:
    strict = Expect(answer_quality=AnswerQualityExpect("ref", criteria=("polite",)))
    lenient = Expect(answer_quality=AnswerQualityExpect("ref", criteria=("polite",),
                                                        accept="partially_correct"))  # fmt: skip
    miss_extra = _answer([True, False])
    miss_reference = _answer([False, True])
    r, _ = run_with([miss_extra], strict)
    assert chk(r, "answer_quality").reason_code == "answer_quality.below_bar"
    assert chk(run_with([miss_extra], lenient)[0], "answer_quality").passed is True
    assert chk(run_with([miss_reference], lenient)[0], "answer_quality").passed is False


def test_answer_quality_no_output_fails_without_a_judge_call() -> None:
    report, model = run_with([], Expect(answer_quality=AnswerQualityExpect("ref")),
                             run_of(text=None))  # fmt: skip
    assert chk(report, "answer_quality").reason_code == "answer_quality.no_output"
    assert model.calls == []


def test_groundedness_outcomes() -> None:
    run = run_of(step(1, "get_order", {"id": "A12"}, effect="read", result={"total": 10}),
                 text="Your order total is 10")  # fmt: skip
    ok, model = run_with([_ground(["supported"])], Expect(grounded=True), run)
    assert chk(ok, "groundedness").passed is True
    assert '"total": 10' in model.calls[0].messages[-1]["content"] or "total" in str(model.calls[0])
    bad, _ = run_with([_ground(["unsupported"])], Expect(grounded=True), run)
    assert chk(bad, "groundedness").reason_code == "groundedness.unsupported"
    contra, _ = run_with([_ground(["contradicted"])], Expect(grounded=True), run)
    assert chk(contra, "groundedness").reason_code == "groundedness.contradicted"
    none, _ = run_with([_ground([])], Expect(grounded=True), run)
    assert chk(none, "groundedness").passed is None  # nothing to check is never a pass


def test_groundedness_allow_unsupported_never_allows_contradictions() -> None:
    loose = Expect(grounded=GroundednessExpect(allow_unsupported=True))
    assert chk(run_with([_ground(["unsupported"])], loose)[0], "groundedness").passed is True
    assert chk(run_with([_ground(["contradicted"])], loose)[0], "groundedness").passed is False


def test_groundedness_sources_filter_the_tool_results_shown_to_the_judge() -> None:
    run = run_of(step(1, "get_order", {}, effect="read", result={"secret": "AAA"}),
                 step(2, "search", {}, effect="read", result={"hit": "BBB"}), text="x")  # fmt: skip
    expect = Expect(grounded=GroundednessExpect(sources=("search",)))
    _, model = run_with([_ground(["supported"])], expect, run)
    seen = model.calls[0].messages[-1]["content"]
    assert "BBB" in seen and "AAA" not in seen


def test_plan_judge_quality_runs_only_when_constraints_pass() -> None:
    run = run_of(step(1, "get_order", {"id": "A12"}, effect="read"), text="ok")
    expect = Expect(plan=PlanExpect(judge_quality=True))
    sound, _ = run_with([_plan([])], expect, run)
    assert chk(sound, "plan").passed is True and chk(sound, "plan").judged
    flawed, _ = run_with([_plan([("missing_step", 2)])], expect, run)
    c = chk(flawed, "plan")
    assert c.reason_code == "plan_quality.flawed" and c.passed is False
    assert c.details["flaws"] == ({"code": "missing_step", "step": 2},)
    failing = Expect(plan=PlanExpect(judge_quality=True, constraints=("nope",)))
    with pytest.raises(Exception, match="nope"):
        run_with([_plan([])], failing, run)


def test_uncalibrated_judges_are_advisory_for_every_new_kind() -> None:
    for expect, name, script in (
        (Expect(answer_quality=AnswerQualityExpect("ref")), "answer_quality", _answer([False])),
        (Expect(grounded=True), "groundedness", _ground(["unsupported"])),
        (Expect(plan=PlanExpect(judge_quality=True)), "plan", _plan([("other", 1)])),
    ):
        run = run_of(step(1, "t", {}, effect="read"), text="x")
        report, _ = run_with([script], expect, run, require_calibration="builtin_or_project")
        c = chk(report, name)
        assert c.status == "skipped" and c.reason_code == "judge_uncalibrated", name
        assert report.outcome != "failed", name  # advisory verdicts never fail a case


def test_triage_is_advisory_opt_in_and_never_changes_the_outcome() -> None:
    from disseqt_agent_testing.judges.triage import disable_triage, enable_triage
    from disseqt_agent_testing.models import Decision

    failing = Expect(decision=Decision.ASK)  # the run answers, so decision_match fails
    plain, _ = run_with([], failing)
    assert plain.outcome == "failed" and plain.triage == ()
    import dataclasses

    on = dataclasses.replace(CFG, triage=True)
    enable_triage()
    try:
        report, model = run_with([{"reasoning": "no question asked", "label": "missing_clarification"}],
                                 failing, config=on)  # fmt: skip
        assert report.outcome == "failed"
        assert [t.label for t in report.triage] == ["missing_clarification"]
        assert report.triage[0].evidence and any(
            c["metric"] == "triage" for c in report.judge_calls
        )
        passing, _ = run_with([], Expect(), config=on)  # nothing failed: no judge call
        assert passing.triage == ()
        broken, _ = run_with([{"reasoning": "x", "label": "bogus"}] * 6, failing, config=on)
        assert broken.outcome == "failed" and broken.triage == ()
    finally:
        disable_triage()
