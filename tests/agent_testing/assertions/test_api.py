"""Composition and API: gating, thresholds, errors, report schema, determinism, extensibility."""

import asyncio
import json
import pickle
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    AgentAssertionError,
    AgentIncompleteError,
    BaseMetric,
    DecisionReport,
    EvalConfig,
    ExpectConfigError,
    MetricResult,
    aevaluate,
    assert_agent,
    check,
    evaluate,
    get_metric,
    list_metrics,
    register_code,
    register_metric,
)
from disseqt_agent_testing.assertions.codes import code_class, known_codes, unregister_code
from disseqt_agent_testing.assertions.config import EvalOptions, config_field_names
from disseqt_agent_testing.assertions.metrics import unregister_metric
from disseqt_agent_testing.assertions.metrics.base import Memo, rollup
from disseqt_agent_testing.assertions.render import render
from disseqt_agent_testing.models import Decision, Expect, ForbiddenCall, ToolExpect

from .helpers import result_of, run_of, step

TWO_TOOLS = Expect(
    decision=Decision.CALL,
    tools=(ToolExpect("a", {"x": 1}, id="a"), ToolExpect("b", {"y": 2}, id="b")),
)


def test_g01_a_failed_decision_with_no_calls_blocks_the_tool_checks() -> None:
    rep = evaluate(run_of(text="Which one?"), TWO_TOOLS)
    assert rep.outcome == "failed"
    blocked = [c for c in rep.checks if c.reason_code == "blocked_by:decision_match"]
    assert {c.name for c in blocked} == {"tool_selection", "tool_args"}


def test_g02_a_threshold_lets_a_partly_wrong_call_pass() -> None:
    run = run_of(step(1, "a", {"x": 1}), step(2, "b", {"y": 9}))
    strict = evaluate(run, Expect(tools=TWO_TOOLS.tools))
    assert strict.outcome == "failed"
    relaxed = evaluate(run, Expect(tools=TWO_TOOLS.tools, min_scores={"arg_score": 0.5}))
    assert relaxed.outcome == "passed"
    assert relaxed.check("tool_args").details["thresholds"]["subs"] == {"arg_score": 0.5}  # type: ignore[union-attr]


def test_g03_thresholds_on_yes_no_checks_and_unknown_keys_are_config_errors() -> None:
    with pytest.raises(ExpectConfigError, match="no threshold"):
        evaluate(run_of(step(1, "a")), Expect(max_steps=3, min_scores={"budgets": 0.5}))
    with pytest.raises(ExpectConfigError) as err:
        evaluate(run_of(), Expect(min_scores={"tool_recal": 0.5}))
    assert "tool_selection.recall" in str(err.value) and "tool_f1" in str(err.value)


def test_a_threshold_for_a_check_that_is_not_enabled_is_an_error_not_a_silent_pass() -> None:
    with pytest.raises(ExpectConfigError, match="does not enable that check"):
        evaluate(run_of(), Expect(min_scores={"tool_args": 0.5}))


def test_g04_a_veto_cannot_be_cleared_by_a_threshold() -> None:
    expect = Expect(
        tools=(ToolExpect("a", count=1),), extra_calls="allow", min_scores={"tool_selection": 0.0}
    )
    rep = evaluate(run_of(step(1, "a"), step(2, "a")), expect)
    assert rep.outcome == "failed" and rep.check("tool_selection").vetoed  # type: ignore[union-attr]


def test_g05_strict_fails_a_score_below_one() -> None:
    run = run_of(step(1, "a", {"x": 1}), step(2, "b", {"y": 9}))
    expect = Expect(tools=TWO_TOOLS.tools, min_scores={"arg_score": 0.5})
    assert evaluate(run, expect).outcome == "passed"
    assert evaluate(run, expect, strict=True).outcome == "failed"


def test_g06_failed_plus_incomplete_is_failed() -> None:
    run = run_of(step(1, "delete"))
    expect = Expect(forbidden=(ForbiddenCall("delete"),), max_llm_calls=1)
    assert evaluate(run, expect).outcome == "failed"
    assert evaluate(run_of(step(1, "ok")), expect).outcome == "incomplete"


def test_g07_a_metric_that_raises_is_an_error_and_the_rest_stay_intact() -> None:
    @register_metric("explodes", "1")
    class Explodes(BaseMetric):
        rank = 5

        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return True

        def evaluate(self, ctx: Any) -> MetricResult:
            raise RuntimeError("kaput")

    try:
        rep = evaluate(run_of(step(1, "a")), Expect(max_steps=3))
    finally:
        unregister_metric("explodes")
    bad = rep.check("explodes")
    assert bad is not None and bad.status == "error" and bad.reason_code == "metric_error"
    assert "RuntimeError: kaput" in bad.reason
    assert rep.check("budgets").passed  # type: ignore[union-attr]
    assert rep.outcome == "incomplete"


def test_config_errors_and_base_exceptions_propagate() -> None:
    @register_metric("interrupts", "1")
    class Interrupts(BaseMetric):
        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return True

        def evaluate(self, ctx: Any) -> MetricResult:
            raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            evaluate(run_of(), Expect(max_steps=1))
    finally:
        unregister_metric("interrupts")


# ---------------------------------------------------------------- harness gate


def test_a_crashed_run_fails_with_the_exact_first_line_and_still_scores_vetoes() -> None:
    run = run_of(step(1, "delete_all"), text=None, outputs=())
    res = result_of(run, status="crashed")
    expect = Expect(forbidden=(ForbiddenCall("delete_*"),), tools=(ToolExpect("x"),))
    rep = evaluate(res, expect)
    text = rep.render()
    assert text.splitlines()[0] == "agent run did not finish: status=crashed (RuntimeError: boom)"
    assert rep.check("forbidden_calls").passed is False  # type: ignore[union-attr]
    assert rep.check("tool_selection").reason_code == "blocked_by:run_status"  # type: ignore[union-attr]
    assert rep.outcome == "failed"


def test_violations_fail_a_finished_run_and_checks_are_still_scored() -> None:
    from disseqt_agent_testing.models import Violation

    res = result_of(
        run_of(step(1, "a")),
        violations=(Violation("unexpected_call", "get_order(B9) matched no response"),),
    )
    rep = evaluate(res, Expect(tools=(ToolExpect("a"),)))
    lines = rep.render().splitlines()
    assert (
        lines[0] == "1 harness violation(s):" and lines[1] == "  get_order(B9) matched no response"
    )
    assert rep.outcome == "failed" and rep.check("tool_selection").passed  # type: ignore[union-attr]


def test_a_bare_agent_run_is_treated_as_finished() -> None:
    assert evaluate(run_of(step(1, "a")), Expect(tools=(ToolExpect("a"),))).outcome == "passed"


def test_t66_a_timed_out_run_is_not_a_pass() -> None:
    rep = evaluate(result_of(run_of(), status="timed_out"), Expect(max_steps=3))
    assert rep.outcome == "failed" and rep.render().startswith(
        "agent run did not finish: status=timed_out"
    )


# ---------------------------------------------------------------- assert_agent and errors


def test_assert_agent_returns_the_report_or_raises() -> None:
    run = run_of(step(1, "a"))
    assert assert_agent(run, Expect(tools=(ToolExpect("a"),))).outcome == "passed"
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(run, Expect(tools=(ToolExpect("zzz"),)), case_id="my_case")
    assert str(err.value).startswith("assert_agent FAILED for case 'my_case' (1 failed")
    assert err.value.report.outcome == "failed"


def test_incomplete_raises_the_subclass_with_the_could_not_verify_line() -> None:
    with pytest.raises(AgentIncompleteError) as err:
        assert_agent(run_of(step(1, "a")), Expect(max_llm_calls=2))
    first = str(err.value).splitlines()[0]
    assert first == "could not verify: budgets (data_unavailable: model calls not captured)"
    assert isinstance(err.value, AgentAssertionError)


def test_x01_a_report_has_no_truth_value() -> None:
    rep = evaluate(run_of(), Expect(max_steps=1))
    with pytest.raises(TypeError, match="use report.passed"):
        bool(rep)
    assert rep.passed is True


def test_x02_errors_survive_pickling() -> None:
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(run_of(), Expect(tools=(ToolExpect("zzz"),)))
    clone = pickle.loads(pickle.dumps(err.value))
    assert str(clone) == str(err.value) and clone.report == err.value.report


def test_x03_the_report_round_trips_and_ignores_unknown_keys() -> None:
    rep = evaluate(
        run_of(step(1, "a", {"x": 1})), Expect(tools=(ToolExpect("a", {"x": 1}),), max_steps=3)
    )
    data = rep.to_dict()
    assert DecisionReport.from_dict(data) == rep
    data["from_the_future"] = {"x": 1}
    assert DecisionReport.from_dict(data) == rep
    assert json.loads(json.dumps(data))["schema_version"] == "1"


def test_x05_assert_agent_hides_its_frames_from_the_traceback() -> None:
    import inspect

    from disseqt_agent_testing.assertions import api

    src = inspect.getsource(api.assert_agent)
    assert "__tracebackhide__ = True" in src and "__tracebackhide__ = True" in inspect.getsource(
        api.raise_for_report
    )


def test_x07_no_ansi_in_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(run_of(), Expect(tools=(ToolExpect("zzz"),)))
    assert "\x1b[" not in str(err.value)


def test_x04_line_budget_truncation_and_ci_override(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = (ToolExpect("t", {f"k{i}": i for i in range(30)}),)
    rep = evaluate(run_of(step(1, "t", {f"k{i}": -1 for i in range(30)})), Expect(tools=tools))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DISSEQT_REPORT_MAX_LINES", raising=False)
    short = rep.render(max_lines=8).splitlines()
    assert (
        len(short) == 8
        and short[-1].startswith("... ")
        and "DISSEQT_REPORT_MAX_LINES=0" in short[-1]
    )
    assert len(rep.render(max_lines=0).splitlines()) > 8
    monkeypatch.setenv("CI", "1")
    assert "more lines" not in rep.render()
    monkeypatch.delenv("CI")
    monkeypatch.setenv("DISSEQT_REPORT_MAX_LINES", "5")
    assert len(rep.render().splitlines()) == 5


def test_every_failing_check_renders_a_registered_code() -> None:
    run = run_of(step(1, "w"), step(2, "delete_it"), text="Which one?")
    expect = Expect(
        decision=Decision.CALL,
        tools=(ToolExpect("a", {"x": 1}), ToolExpect("w", {"q": 2})),
        forbidden=(ForbiddenCall("delete_*"),),
        max_steps=1,
        answer_contains=("zzz",),
        order="in_order",
    )
    rep = evaluate(run, expect)
    text = rep.render(max_lines=0)
    for c in rep.checks:
        if c.status == "ok" and c.passed is False:
            assert (
                c.reason_code in known_codes()
                or c.reason_code.split(":")[0] + ":*" in known_codes()
            )
            assert c.reason_code in text


# ---------------------------------------------------------------- async, threads, determinism


def test_aevaluate_equals_evaluate() -> None:
    run = run_of(step(1, "a", {"x": 1}), step(2, "b", {"y": 2}))
    assert asyncio.run(aevaluate(run, TWO_TOOLS)) == evaluate(run, TWO_TOOLS)
    bad = run_of(text="Which one?")
    assert asyncio.run(aevaluate(bad, TWO_TOOLS)) == evaluate(bad, TWO_TOOLS)


def test_check_is_evaluate() -> None:
    assert check is evaluate


def test_x06_16_threads_by_50_evaluations_are_identical() -> None:
    run = run_of(step(1, "a", {"x": 1}), step(2, "b", {"y": 9}))
    expect = Expect(tools=TWO_TOOLS.tools, decision=Decision.CALL)
    out: list[str] = []
    lock = threading.Lock()

    def go() -> None:
        for _ in range(50):
            text = json.dumps(evaluate(run, expect).to_dict(), sort_keys=True)
            with lock:
                out.append(text)

    threads = [threading.Thread(target=go) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(out) == 800 and len(set(out)) == 1


def test_inputs_are_never_mutated() -> None:
    run = run_of(step(1, "a", {"x": [1, 2]}))
    expect = Expect(tools=(ToolExpect("a", {"x": [1, 2]}),))
    before = (run, expect)
    evaluate(run, expect)
    assert (run, expect) == before


# ---------------------------------------------------------------- extensibility (E-01 ... E-10)


def test_e01_a_registered_metric_is_planned_rendered_and_reported_with_no_core_edit() -> None:
    @register_metric("shouty", "7")
    class Shouty(BaseMetric):
        rank = 999

        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return "shout" in expect.extra

        def evaluate(self, ctx: Any) -> MetricResult:
            loud = (ctx.run.final_text or "").isupper()
            return self.ok(
                ctx, 1.0 if loud else 0.0, loud, reason_code=None if loud else "shout.quiet"
            )

        def render_lines(self, result: MetricResult) -> tuple[str, ...]:
            return (f"shouty    custom line {result.passed}",)

    register_code("shout.quiet", "check_failure", "the answer was not loud")
    try:
        assert evaluate(run_of(), Expect(max_steps=1)).check("shouty") is None  # not applicable
        rep = evaluate(run_of(text="hi"), Expect(extra={"shout": True}))
        got = rep.check("shouty")
        assert got is not None and got.version == "7" and got.passed is False
        assert "shouty    custom line False" in rep.render() and rep.outcome == "failed"
        assert "shouty" in [m.name for m in list_metrics()] and get_metric("shouty").version == "7"
    finally:
        unregister_metric("shouty")
        unregister_code("shout.quiet")
    assert "shouty" not in [m.name for m in list_metrics()]


def test_duplicate_metric_names_raise_unless_overridden() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_metric("budgets", "9")
        class Dup(BaseMetric):
            pass

    assert get_metric("budgets").version == "1"


def test_e05_unknown_codes_and_outcomes_degrade_gracefully() -> None:
    class Odd(BaseMetric):
        rank = 5

        def evaluate(self, ctx: Any) -> MetricResult:
            return self.skip(ctx, "mystery_code", "who knows")

    @register_metric("odd", "1")
    class Registered(Odd):
        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return True

    try:
        rep = evaluate(run_of(), Expect(max_steps=1))
    finally:
        unregister_metric("odd")
    assert code_class("mystery_code") == "unverifiable" and rep.outcome == "incomplete"
    assert "mystery_code" in rep.render()
    assert render({**rep.to_dict(), "outcome": "something_new"}).startswith(
        "assert_agent INCOMPLETE"
    )


def test_e06_options_typeddict_matches_config_fields() -> None:
    assert set(EvalOptions.__annotations__) == set(config_field_names())


def test_unknown_option_lists_valid_ones() -> None:
    with pytest.raises(TypeError, match="unknown option 'strictt'; valid options"):
        evaluate(run_of(), Expect(max_steps=1), strictt=True)


def test_e07_unknown_judge_kind_passes_through_request_judge() -> None:
    from disseqt_agent_testing.assertions import JudgeTask, JudgeVerdict

    class Fixed:
        def judge(self, task: Any) -> JudgeVerdict:
            return JudgeVerdict("ok", task.kind)

        async def ajudge(self, task: Any) -> JudgeVerdict:
            return self.judge(task)

    from disseqt_agent_testing.assertions.metrics import EvalContext

    ctx = EvalContext(
        None, run_of(), Expect(), "c", EvalConfig(use_judge=True), Fixed(), (), Memo()  # type: ignore[arg-type]
    )
    assert ctx.request_judge(JudgeTask("brand_new_kind")).label == "brand_new_kind"  # type: ignore[union-attr]
    off = EvalContext(None, run_of(), Expect(), "c", EvalConfig(), Fixed(), (), Memo())  # type: ignore[arg-type]
    assert off.request_judge(JudgeTask("x")) is None


def test_e08_new_dataclasses_follow_the_layout_rules() -> None:
    import dataclasses

    from disseqt_agent_testing.assertions import (
        DecisionReport,
        EvalConfig,
        MetricResult,
        RepeatReport,
    )
    from disseqt_agent_testing.assertions.config import Lexicons
    from disseqt_agent_testing.assertions.judge import JudgeTask, JudgeVerdict

    expected = {
        MetricResult: ("name",),
        DecisionReport: ("case_id", "outcome"),
        RepeatReport: ("n",),
        EvalConfig: (),
        Lexicons: (),
        JudgeVerdict: ("status", "label"),
        JudgeTask: ("kind", "prompt_version", "inputs"),
    }
    for cls, positional in expected.items():
        got = tuple(f.name for f in dataclasses.fields(cls) if f.init and not f.kw_only)
        assert got == positional, cls.__name__
    for cls in (MetricResult, DecisionReport, RepeatReport.__mro__[0] and EvalConfig):
        assert "extra" in {f.name for f in dataclasses.fields(cls)}


def test_e09_e10_codes_and_metric_registries_are_restored() -> None:
    before_codes, before_metrics = known_codes(), [m.name for m in list_metrics()]
    register_code("tmp.code", "benign_skip", "x")
    unregister_code("tmp.code")
    with pytest.raises(ValueError, match="severity_class"):
        register_code("bad", "nonsense", "x")
    with pytest.raises(ValueError, match="already registered"):
        register_code("tool.missing", "check_failure", "x")
    assert known_codes() == before_codes and [m.name for m in list_metrics()] == before_metrics


def test_e10_a_registered_metrics_sub_score_becomes_a_valid_min_scores_key() -> None:
    @register_metric("graded", "1")
    class Graded(BaseMetric):
        rank = 5
        sub_scores = ("depth",)
        thresholdable = True

        def applicable(self, expect: Expect, config: EvalConfig) -> bool:
            return False

        def evaluate(self, ctx: Any) -> MetricResult:
            return self.ok(ctx, 0.4, True, details={"scores": {"depth": 0.4}})

    try:
        rep = evaluate(
            run_of(),
            Expect(max_steps=1, min_scores={"graded.depth": 0.9}),
            extra_metrics=("graded",),
        )
        assert rep.check("graded").passed is False and rep.outcome == "failed"  # type: ignore[union-attr]
        with pytest.raises(ExpectConfigError, match="graded.depth"):
            evaluate(
                run_of(),
                Expect(max_steps=1, min_scores={"graded.nope": 0.9}),
                extra_metrics=("graded",),
            )
    finally:
        unregister_metric("graded")


def test_memo_computes_once_and_caches_exceptions() -> None:
    memo, calls = Memo(), []

    def boom() -> None:
        calls.append(1)
        raise RuntimeError("x")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            memo.get_or_compute("k", boom)
    assert calls == [1]
    results: list[int] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(memo.get_or_compute("v", lambda: len(calls) or 5))
        )
        for _ in range(16)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(results)) == 1


def test_metric_result_validation_and_round_trip() -> None:
    with pytest.raises(ValueError, match="needs value=None"):
        MetricResult("x", status="skipped", value=1.0, reason_code="data_unavailable")
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        MetricResult("x", status="ok", value=2.0, passed=True)
    child = MetricResult(
        "c", status="ok", value=0.3333333333, passed=True, details={"k": [1.23456789]}
    )
    parent = MetricResult("p", status="ok", value=1.0, passed=True, components=(child,))
    assert MetricResult.from_dict(parent.to_dict()) == parent
    assert child.value == 0.333333 and child.to_dict()["details"] == {"k": [1.234568]}


def test_rollup_rules() -> None:
    """Deliberately flipped: ``rollup([ok, skipped data_unavailable])`` used to be ``("ok", True)``,
    which let a check pass while one of its sub-checks could not be measured (fail-open). Now an
    unmeasurable child makes the parent skipped with that code; failures still dominate and
    benign skips (not_applicable) still do not block."""
    ok = MetricResult("a", status="ok", value=1.0, passed=True)
    bad = MetricResult("b", status="ok", value=0.0, passed=False)
    skip = MetricResult("c", status="skipped", reason_code="data_unavailable")
    err = MetricResult("d", status="error", reason_code="metric_error")
    assert rollup([ok, skip])[0::3] == ("skipped", "data_unavailable")
    benign = MetricResult("n", status="skipped", reason_code="not_applicable")
    assert rollup([ok, benign])[0::2] == ("ok", True)
    assert rollup([bad, skip])[0::2] == ("ok", False)
    assert rollup([ok, bad])[0::2] == ("ok", False)
    assert rollup([skip, skip])[0] == "skipped"
    assert rollup([ok, err])[0] == "error" and rollup([bad, err])[0::2] == ("ok", False)


@dataclass
class _Unused:
    pass


def test_a_tight_line_budget_shows_the_failing_check_first() -> None:
    run = run_of(step(1, "t", {"a": 2}), text="done")
    expect = Expect(
        decision=Decision.CALL, tools=(ToolExpect("t", {"a": 1}),), max_steps=5, extra_calls="allow"
    )
    rep = evaluate(run, expect)
    text = rep.render(max_lines=4)
    assert "$.a" in text and "decision   turn 0" not in text
    full = rep.render(max_lines=0)
    assert full.index("decision   turn 0") < full.index("$.a")  # normal order is by rank
