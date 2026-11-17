"""A plan check that fails for a non-score reason (constraint failed, judge found flaws) must stay
failed when a ``plan.convergence`` threshold is also set and the score meets it.

Regression: ``apply_threshold`` recomputes ``passed`` from the thresholds alone and ignores the
metric's own failure, so a flawed plan passed whenever ``min_scores={"plan.convergence": ...}``
was set and the call count was close to optimal."""

from typing import Any

from disseqt_agent_testing.assertions import EvalConfig, evaluate
from disseqt_agent_testing.harness import build_test_orchestrator, stub
from disseqt_agent_testing.judges import FakeServiceTransport, KindConfig, ServiceModel, make_judge
from disseqt_agent_testing.models import Expect, PlanExpect


def _result() -> Any:
    def factory(tools: Any) -> Any:
        class A:
            def run(self, text: str) -> str:
                tools["get_order"]("A12")
                tools["issue_refund"](order_id="A12")
                return "done"

        return A()

    fakes = {"get_order": stub({"ok": True}), "issue_refund": stub({"ok": True})}
    return build_test_orchestrator(factory, fakes).run("Refund A12")


def _judge(flaws: list[dict[str, Any]]) -> Any:
    reply = {"reasoning": "r", "flaws": flaws, "enough_data": True}
    return make_judge(
        ServiceModel("m", transport=FakeServiceTransport(handler=lambda p: reply)),
        require_calibration="off",
        kinds={"plan_quality": KindConfig(samples=1)},
        cache_mode="off",
    )


def test_judge_flaws_fail_even_when_convergence_threshold_is_met() -> None:
    judge = _judge([{"code": "missing_step", "step": 1}])
    try:
        report = evaluate(
            _result(),
            Expect(
                plan=PlanExpect(optimal_tool_calls=2, judge_quality=True),
                min_scores={"plan.convergence": 0.7},
            ),
            judge=judge,
            config=EvalConfig(use_judge=True),
        )
    finally:
        judge.close()
    assert report.outcome == "failed"


def test_sound_plan_still_passes_with_the_threshold() -> None:
    judge = _judge([])
    try:
        report = evaluate(
            _result(),
            Expect(
                plan=PlanExpect(optimal_tool_calls=2, judge_quality=True),
                min_scores={"plan.convergence": 0.7},
            ),
            judge=judge,
            config=EvalConfig(use_judge=True),
        )
    finally:
        judge.close()
    assert report.outcome == "passed"


def test_failed_constraint_fails_even_when_convergence_threshold_is_met() -> None:
    report = evaluate(
        _result(),
        Expect(
            plan=PlanExpect(optimal_tool_calls=2, constraints=("never_true",)),
            min_scores={"plan.convergence": 0.7},
        ),
        config=EvalConfig(predicates={"never_true": lambda run: (False, "nope")}),
    )
    assert report.outcome == "failed"
