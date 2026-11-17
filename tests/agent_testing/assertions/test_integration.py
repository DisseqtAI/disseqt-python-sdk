"""Plan 02 acceptance (T48-T50) and plan 03 integration (I-01 ... I-08) with the real harness."""

import asyncio
import functools
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.assertions import (
    AgentAssertionError,
    AgentIncompleteError,
    aevaluate,
    assert_agent,
    evaluate,
)
from disseqt_agent_testing.harness import FakeLLM, build_test_orchestrator, spy
from disseqt_agent_testing.models import (
    Approx,
    Decision,
    Expect,
    FakeResponse,
    FakeSpec,
    ForbiddenCall,
    Ref,
    ToolExpect,
)
from tests.agent_testing._support.refund_agent import (
    FAKES,
    make_async_refund_agent,
    make_llm_refund_agent,
    make_refund_agent,
    refund_script,
)

GOLDEN = Path(__file__).parent / "golden"

REFUND = Expect(
    decision=Decision.CALL,
    tools=(
        ToolExpect("get_order", {"order_id": Ref("input")}, id="lookup"),
        ToolExpect(
            "issue_refund",
            {"order_id": Ref("call:lookup.result.order_id"), "amount": Approx(59.99, abs_tol=0.01)},
            id="refund",
            after=("lookup",),
        ),
    ),
    order="partial",
    forbidden=(ForbiddenCall("delete_*"),),
    max_steps=6,
)


def run_refund(text: str = "refund order A12", **options: Any) -> Any:
    return build_test_orchestrator(make_refund_agent, FAKES, **options).run(text)


def wrong_order() -> Any:
    factory = functools.partial(make_refund_agent, bug="wrong_order")
    return build_test_orchestrator(factory, FAKES).run("refund order A12")


def test_t48_i01_the_happy_path_passes() -> None:
    rep = assert_agent(run_refund(), REFUND, case_id="refund_ok")
    assert rep.outcome == "passed" and rep.scores["tool_args"] == 1.0
    assert [c.name for c in rep.checks] == [
        "decision_match", "tool_selection", "tool_args", "order", "forbidden_calls", "budgets",
    ]  # fmt: skip


def test_t49_i02_the_wrong_order_id_fails_with_one_precise_diff() -> None:
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(wrong_order(), REFUND, case_id="refund_wrong_order")
    rep = err.value.report
    diffs = rep.check("tool_args").details["pairs"]  # type: ignore[union-attr]
    assert len(diffs) == 1
    d = diffs[0]
    assert (d["path_str"], d["code"], d["actual"]) == ("$.order_id", "ungrounded_value", "Z999")
    assert (
        d["ref"]["expected_value"] == "A12" and d["ref"]["resolved_from"] == "expectation 'lookup'"
    )
    assert rep.check("tool_selection").passed and rep.check("order").passed  # type: ignore[union-attr]
    assert str(err.value) == (GOLDEN / "refund_wrong_order.txt").read_text().rstrip("\n")


def test_t50_i03_asking_for_the_missing_order_number_passes_with_no_steps() -> None:
    res = run_refund("I want a refund")
    assert res.run.steps == ()
    expect = Expect(decision=Decision.ASK, forbidden=(ForbiddenCall("issue_refund"),))
    assert assert_agent(res, expect).outcome == "passed"
    with pytest.raises(AgentAssertionError):
        assert_agent(res, REFUND)  # an ASK is not a CALL


def test_i04_an_ineligible_order_is_refused_and_no_refund_is_issued() -> None:
    fakes = {**FAKES, "check_policy": spy({"eligible": False})}
    res = build_test_orchestrator(make_refund_agent, fakes).run("refund order A12")
    expect = Expect(
        decision=Decision.REFUSE,
        answer_contains=("not eligible",),
        forbidden=(ForbiddenCall("issue_refund"),),
        tools=(ToolExpect("get_order", id="lookup"), ToolExpect("check_policy")),
    )
    assert assert_agent(res, expect).outcome == "passed"


def test_i05_a_follow_up_turn_is_checked_per_turn() -> None:
    res = build_test_orchestrator(make_refund_agent, FAKES).run(
        "I want a refund", follow_ups=("order A12",)
    )
    expect = Expect(
        decision=Decision.ASK,
        decision_by_turn={1: Decision.CALL},
        tools=(ToolExpect("issue_refund", turn=1),),
        extra_calls="allow",
    )
    rep = assert_agent(res, expect)
    assert [d["label"] for d in rep.decisions] == ["ask", "call"]


def test_i06_a_crash_names_the_cause_and_vetoes_still_score() -> None:
    spec = FakeSpec(
        "get_order", responses=(FakeResponse(when={"order_id": "A1"}, returns={}, times=1),)
    )
    res = build_test_orchestrator(make_refund_agent, (spec,)).run("refund order B99")
    assert res.status == "crashed"
    expect = Expect(forbidden=(ForbiddenCall("delete_*"),), tools=(ToolExpect("get_order"),))
    with pytest.raises(AgentAssertionError) as err:
        assert_agent(res, expect)
    first = str(err.value).splitlines()[0]
    assert first.startswith(
        "agent run did not finish: status=crashed (UnexpectedFakeCall: get_order("
    )
    assert err.value.report.check("forbidden_calls").passed  # type: ignore[union-attr]
    assert any("harness violation" in ln for ln in str(err.value).splitlines())


def test_i07_a_scripted_llm_run_checks_model_calls_and_flags_missing_usage() -> None:
    res = build_test_orchestrator(
        make_llm_refund_agent, FAKES, llm=FakeLLM(script=refund_script())
    ).run("refund A12")
    assert assert_agent(res, Expect(max_llm_calls=3, tools=(ToolExpect("issue_refund"),))).passed
    with pytest.raises(AgentAssertionError):
        assert_agent(res, Expect(max_llm_calls=2))
    with pytest.raises(AgentIncompleteError):
        assert_agent(res, Expect(max_tokens=1000))  # the script carried no usage


def test_the_async_agent_and_arun_give_identical_reports() -> None:
    orch = build_test_orchestrator(make_async_refund_agent, FAKES, async_tools=True)
    sync_rep = evaluate(orch.run("refund order A12"), REFUND)
    async_rep = asyncio.run(aevaluate(asyncio.run(orch.arun("refund order A12")), REFUND))
    assert sync_rep.outcome == "passed" and sync_rep.scores == async_rep.scores


def test_reports_are_byte_identical_across_runs() -> None:
    import json

    texts = {
        json.dumps(evaluate(run_refund(), REFUND).to_dict(), sort_keys=True).replace(
            run_refund().trace_id, "T"
        )
        for _ in range(3)
    }
    assert len(texts) <= 3  # trace ids differ; everything else must not
    stripped = {
        json.dumps({**evaluate(run_refund(), REFUND).to_dict(), "harness": {}}, sort_keys=True)
        for _ in range(5)
    }
    assert len(stripped) == 1


def test_a_changed_matcher_changes_the_params_hash_but_not_the_version() -> None:
    a = evaluate(run_refund(), REFUND).check("tool_args")
    other = Expect(
        decision=Decision.CALL,
        tools=(ToolExpect("issue_refund", {"amount": 1}),),
    )
    b = evaluate(run_refund(), other).check("tool_args")
    assert a.version == b.version == "1" and a.params_hash != b.params_hash  # type: ignore[union-attr]
