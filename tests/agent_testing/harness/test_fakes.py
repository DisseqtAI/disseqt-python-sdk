"""Fake dispatch: T01-T16, T51 (rules, defaults, errors, signatures, handlers)."""

import inspect
from typing import Any

import pytest

from disseqt_agent_testing.harness import FakeToolError, HarnessConfigError, spy, stub
from disseqt_agent_testing.harness.errors import UnexpectedFakeCall
from disseqt_agent_testing.models import FakeResponse, FakeSpec, ModelValidationError, Ref

from .helpers import run_body

SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}, "note": {"type": "string"}},
    "required": ["order_id"],
}


def test_t01_stub_records_a_full_step() -> None:
    r = run_body({"t": stub({"ok": 1})}, lambda tools: tools["t"](order_id="A1"))
    step = r.run.steps[0]
    assert r.status == "finished"
    assert (step.kind, step.effect, step.call_id, step.seq, step.turn) == (
        "tool", "write", "call-1", 1, 0,
    )  # fmt: skip
    assert step.parallel_group is None
    assert dict(step.args) == {"order_id": "A1"} and dict(step.result) == {"ok": 1}


def test_t02_bare_dict_in_the_mapping_is_a_write_stub() -> None:
    r = run_body({"t": {"ok": 1}}, lambda tools: tools["t"](order_id="A1"))
    assert r.run.steps[0].effect == "write" and dict(r.run.steps[0].result) == {"ok": 1}


def test_t03_spy_returns_none_and_is_a_read() -> None:
    seen: list[Any] = []
    r = run_body({"t": spy()}, lambda tools: seen.append(tools["t"]()))
    assert seen == [None] and r.run.steps[0].effect == "read"


def test_t04_first_matching_rule_wins() -> None:
    spec = FakeSpec(
        "t",
        responses=(
            FakeResponse(when={"x": 1}, returns="first"),
            FakeResponse(when={"x": 1}, returns="second"),
        ),
    )
    seen: list[Any] = []
    run_body((spec,), lambda tools: seen.append(tools["t"](x=1)))
    assert seen == ["first"]


def test_t05_times_limits_a_rule_then_falls_through() -> None:
    spec = FakeSpec("t", responses=(FakeResponse(returns="a", times=1), FakeResponse(returns="b")))
    seen: list[Any] = []
    run_body((spec,), lambda tools: [seen.append(tools["t"]()) for _ in range(3)])
    assert seen == ["a", "b", "b"]


def test_t06_default_is_used_when_nothing_matches() -> None:
    spec = FakeSpec(
        "t",
        responses=(FakeResponse(when={"x": 1}, returns="hit"),),
        default=FakeResponse(returns="dflt"),
    )
    seen: list[Any] = []
    run_body((spec,), lambda tools: seen.append(tools["t"](x=2)))
    assert seen == ["dflt"]


def _strict_spec() -> FakeSpec:
    return FakeSpec(
        "get_order",
        schema=SCHEMA,
        responses=(FakeResponse(when={"order_id": "A1"}, returns=1, times=1),),
    )


def test_t07_unmatched_call_crashes_with_exact_message() -> None:
    r = run_body((_strict_spec(),), lambda tools: tools["get_order"](order_id="B9"))
    assert r.status == "crashed" and r.error is not None
    assert r.error.message.startswith(
        'UnexpectedFakeCall: get_order({"order_id": "B9"}) matched no response and has no '
        'default; rules: #0 when={"order_id": "A1"} remaining=1'
    )
    assert r.run.steps[0].error and r.run.steps[0].error.startswith("UnexpectedFakeCall")
    assert [v.code for v in r.violations] == ["unexpected_call"]


def test_t08_swallowed_unmatched_call_is_still_a_violation() -> None:
    def body(tools: Any) -> str:
        try:
            tools["get_order"](order_id="B9")
        except UnexpectedFakeCall:
            pass
        return "done"

    r = run_body((_strict_spec(),), body)
    assert r.status == "finished" and [v.code for v in r.violations] == ["unexpected_call"]


def test_t09_scripted_error_reaches_the_agent() -> None:
    spec = FakeSpec("t", default=FakeResponse(error="order service down"))
    seen: list[Any] = []

    def body(tools: Any) -> None:
        try:
            tools["t"]()
        except FakeToolError as exc:
            seen.append(str(exc))

    r = run_body((spec,), body)
    assert seen == ["order service down"]
    assert r.run.steps[0].error == "FakeToolError: order service down"
    assert r.run.steps[0].result is None


def test_t10_returned_values_are_fresh_copies() -> None:
    spec = FakeSpec("t", default=FakeResponse(returns={"items": [1, 2]}))
    seen: list[Any] = []

    def body(tools: Any) -> None:
        first = tools["t"]()
        first["items"].append(99)
        seen.append(tools["t"]())

    run_body((spec,), body)
    assert seen == [{"items": [1, 2]}]


def test_t11_positional_without_schema_uses_arg_n() -> None:
    r = run_body({"t": stub(1)}, lambda tools: tools["t"]("A1"))
    assert dict(r.run.steps[0].args) == {"arg0": "A1"}


def test_t12_positional_with_schema_binds_by_name_and_drops_unset_optionals() -> None:
    spec = FakeSpec("t", schema=SCHEMA, default=FakeResponse(returns=1))
    r = run_body((spec,), lambda tools: tools["t"]("A1"))
    assert dict(r.run.steps[0].args) == {"order_id": "A1"}


def test_t13_wrong_keyword_is_a_typeerror_in_the_agent() -> None:
    spec = FakeSpec("get_order", schema=SCHEMA, default=FakeResponse(returns=1))
    r = run_body((spec,), lambda tools: tools["get_order"](id="x"))
    assert r.status == "crashed"
    assert r.run.steps[0].error == "TypeError: get_order() got an unexpected keyword argument 'id'"


def test_t14_subagent_with_one_string() -> None:
    spec = FakeSpec("billing", kind="subagent", default=FakeResponse(returns="ok"))

    def factory(subagents: Any) -> Any:
        return lambda text: subagents["billing"]("please refund")

    from disseqt_agent_testing.harness import build_test_orchestrator

    r = build_test_orchestrator(factory, (spec,)).run("go")
    step = r.run.steps[0]
    assert (step.kind, dict(step.args)) == ("subagent", {"input": "please refund"})


def test_t15_handler_result_is_returned_and_exceptions_are_recorded_and_propagated() -> None:
    spec = FakeSpec("t")
    seen: list[Any] = []
    r = run_body(
        (spec,), lambda tools: seen.append(tools["t"](x=2)), handlers={"t": lambda x: x * 3}
    )
    assert seen == [6] and r.run.steps[0].result == 6

    def boom(**_: Any) -> None:
        raise ValueError("handler broke")

    r = run_body((spec,), lambda tools: tools["t"](), handlers={"t": boom})
    assert r.status == "crashed" and r.run.steps[0].error == "ValueError: handler broke"


def test_t16_unused_rules_are_reported_when_asked() -> None:
    spec = FakeSpec("issue_refund", responses=(FakeResponse(returns=1, times=1),))
    r = run_body((spec,), lambda tools: None, assert_all_called=True)
    assert [v.code for v in r.violations] == ["unused_response"]
    assert r.violations[0].message == (
        "fake 'issue_refund' response #0 declared times=1 but was used 0 times"
    )
    assert run_body((spec,), lambda tools: None).violations == ()  # off by default


def test_t51_unserializable_results_are_captured_lossily() -> None:
    r = run_body({"t": spy()}, lambda tools: tools["t"](obj=object()))
    step = r.run.steps[0]
    assert step.lossy is True and "object" in str(dict(step.args)["obj"])


def test_fake_callables_are_introspectable_functions() -> None:
    spec = FakeSpec("get_order", schema=SCHEMA, default=FakeResponse(returns=1))
    seen: dict[str, Any] = {}

    def body(tools: Any) -> None:
        fn = tools["get_order"]
        seen["name"] = fn.__name__
        seen["sig"] = inspect.signature(fn)

    run_body((spec,), body)
    assert seen["name"] == "get_order"
    assert list(seen["sig"].parameters) == ["order_id", "note"]
    assert seen["sig"].parameters["note"].default is not inspect.Parameter.empty
    assert seen["sig"].parameters["order_id"].default is inspect.Parameter.empty


def test_t56_setup_mistakes_are_config_errors() -> None:
    from disseqt_agent_testing.harness import build_test_orchestrator

    a = FakeSpec("a")
    with pytest.raises(HarnessConfigError, match="duplicate fake name 'a'"):
        build_test_orchestrator(lambda tools: None, (a, a))
    with pytest.raises(HarnessConfigError, match="fakes key 'b' does not match FakeSpec.name 'a'"):
        build_test_orchestrator(lambda tools: None, {"b": a})
    with pytest.raises(HarnessConfigError, match="sets both default and handler"):
        build_test_orchestrator(
            lambda tools: None,
            (FakeSpec("a", default=FakeResponse(returns=1)),),
            handlers={"a": lambda: 1},
        )
    with pytest.raises(ModelValidationError):  # plan-01 error, raised before the harness
        FakeResponse(when={"x": Ref("steps[0].result")}, returns=1)
