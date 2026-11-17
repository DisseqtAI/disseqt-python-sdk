"""``EvalConfig.triage`` must be the switch for root-cause triage.

Regression: the hook ignored the flag (it ran whenever ``enable_triage()`` had been called and
``use_judge`` was on) and ``triage=True`` alone did nothing unless the caller also knew to call
``enable_triage()``. Plan 04 (A5): triage runs "only when outcome is failed and
``EvalConfig.triage=True``"."""

from typing import Any

from disseqt_agent_testing.assertions import EvalConfig, evaluate
from disseqt_agent_testing.harness import build_test_orchestrator, stub
from disseqt_agent_testing.judges import FakeJudgeModel, KindConfig, make_judge
from disseqt_agent_testing.judges.triage import disable_triage, enable_triage
from disseqt_agent_testing.models import Expect, ForbiddenCall

HINT = {"reasoning": "called a forbidden tool", "label": "wrong_tool"}
EXPECT = Expect(forbidden=(ForbiddenCall("delete_account"),))


def _failing_run() -> Any:
    def factory(tools: Any) -> Any:
        class A:
            def run(self, text: str) -> str:
                tools["delete_account"]("u1")
                return "ok"

        return A()

    return build_test_orchestrator(factory, {"delete_account": stub({"ok": True})}).run("x")


def _evaluate(triage: bool) -> Any:
    judge = make_judge(
        FakeJudgeModel({"root_cause": [HINT] * 4}),
        require_calibration="off",
        kinds={"root_cause": KindConfig(samples=1)},
        cache_mode="off",
    )
    try:
        return evaluate(
            _failing_run(), EXPECT, judge=judge, config=EvalConfig(use_judge=True, triage=triage)
        )
    finally:
        judge.close()


def test_triage_stays_off_when_the_flag_is_off_even_with_the_hook_registered() -> None:
    enable_triage()
    try:
        report = _evaluate(triage=False)
    finally:
        disable_triage()
    assert report.outcome == "failed" and report.triage == ()


def test_triage_flag_alone_is_enough() -> None:
    disable_triage()  # nothing registered by hand: make_judge must do it
    try:
        report = _evaluate(triage=True)
    finally:
        disable_triage()
    assert report.outcome == "failed"
    assert [t.label for t in report.triage] == ["wrong_tool"]
