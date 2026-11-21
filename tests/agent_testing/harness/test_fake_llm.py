"""FakeLLM and its per-run client: T53, T54, T70-T74."""

import asyncio
import dataclasses
import threading
from typing import Any

import pytest

from disseqt_agent_testing.harness import (
    FakeLLM,
    FakeReply,
    HarnessConfigError,
    ScriptExhaustedError,
    build_test_orchestrator,
)
from disseqt_agent_testing.models import (
    Message,
    MsgToolCall,
    StepEvent,
    Usage,
    from_dict,
    to_dict,
)
from tests.agent_testing._support.refund_agent import FAKES, make_llm_refund_agent, refund_script


def llm_run(llm: FakeLLM, text: str = "refund A12") -> Any:
    return build_test_orchestrator(make_llm_refund_agent, FAKES, llm=llm).run(text)


def test_t53_model_steps_interleave_with_tool_steps() -> None:
    r = llm_run(FakeLLM(script=refund_script()))
    assert r.status == "finished" and r.run.model_steps_captured is True
    assert [(s.seq, s.kind) for s in r.run.steps] == [
        (1, "model"), (2, "tool"), (3, "model"), (4, "tool"), (5, "model"),
    ]  # fmt: skip
    assert r.run.final_text == "Refunded 59.99 for A12."
    assert not any(w.startswith("model calls not captured") for w in r.warnings)


def test_t54_script_exhaustion_message_and_crash() -> None:
    r = llm_run(FakeLLM(script=(Message("assistant", "only one"),)), "hello")
    assert r.status == "finished"  # the agent got its single reply and stopped
    two = build_test_orchestrator(
        lambda llm: (
            lambda t: [llm.complete([Message("user", t)]), llm.complete([Message("user", t)])]
        ),
        (),
        llm=FakeLLM(script=("one",)),
    ).run("question")
    assert two.status == "crashed" and two.error is not None
    assert two.error.message == (
        "ScriptExhaustedError: FakeLLM script exhausted after 1 responses; "
        "call #2 received 1 messages (last user: 'question')"
    )
    assert two.run.steps[-1].error and two.run.steps[-1].error.startswith("ScriptExhaustedError")


def test_exhaustion_can_repeat_the_last_reply() -> None:
    r = build_test_orchestrator(
        lambda llm: (lambda t: [llm.complete([Message("user", t)]).content for _ in range(3)]),
        (),
        llm=FakeLLM(script=("a", "b"), on_exhaust="repeat_last"),
    ).run("q")
    assert r.run.final_answer == ("a", "b", "b")


def test_script_exhausted_error_is_catchable_by_type() -> None:
    assert issubclass(ScriptExhaustedError, Exception)


def test_t70_script_roles_are_validated_and_strings_become_assistant_messages() -> None:
    with pytest.raises(HarnessConfigError) as err:
        FakeLLM(script=("ok", Message("tool", "x", tool_call_id="c1")))
    assert (
        str(err.value)
        == "FakeLLM script entry #2 has role 'tool'; script entries must be assistant messages"
    )
    with pytest.raises(HarnessConfigError, match="has role 'user'"):
        FakeLLM(script=(Message("user", "x"),))
    assert FakeLLM(script=("hi",)).script[0].message == Message("assistant", "hi")
    with pytest.raises(HarnessConfigError, match="exactly one of script or handler"):
        FakeLLM()
    with pytest.raises(HarnessConfigError, match="exactly one of script or handler"):
        FakeLLM(script=("x",), handler=lambda m, t: "x")


def test_t71_parallel_calls_in_one_assistant_message() -> None:
    parallel = Message(
        "assistant",
        tool_calls=(
            MsgToolCall("p1", "get_order", arguments={"order_id": "A12"}),
            MsgToolCall("p2", "check_policy", arguments={"order_id": "A12", "reason": "damaged"}),
        ),
    )
    r = llm_run(FakeLLM(script=(parallel, Message("assistant", "done"))))
    kinds = [(s.kind, s.name) for s in r.run.steps]
    assert kinds == [
        ("model", "fake-llm"),
        ("tool", "get_order"),
        ("tool", "check_policy"),
        ("model", "fake-llm"),
    ]
    reply = r.run.steps[0].result
    assert [c["id"] for c in reply["tool_calls"]] == ["p1", "p2"]  # type: ignore[index]


def test_t72_a_tool_history_is_accepted_and_recorded_verbatim() -> None:
    history = [
        Message("user", "refund A12"),
        Message(
            "assistant", tool_calls=(MsgToolCall("c1", "get_order", arguments={"order_id": "A12"}),)
        ),
        Message("tool", "{}", tool_call_id="c1"),
    ]
    box: dict[str, Any] = {}

    def factory(llm: Any) -> Any:
        def agent(text: str) -> str:
            box["client"] = llm
            return llm.complete(history).content

        return agent

    r = build_test_orchestrator(factory, (), llm=FakeLLM(script=("ok",))).run("go")
    assert r.status == "finished" and tuple(history) == box["client"].requests[-1]
    assert r.run.steps[0].args["messages"][2]["tool_call_id"] == "c1"


def test_t73_a_case_file_script_with_a_missing_call_id_gets_stable_ids() -> None:
    raw = {
        "role": "assistant",
        "tool_calls": [{"name": "get_order", "arguments": {"order_id": "A12"}}, {"name": "x"}],
    }
    first = from_dict(Message, raw, fill_ids=True)
    second = from_dict(Message, raw, fill_ids=True)
    assert [c.id for c in first.tool_calls] == ["call_1", "call_2"] and first == second


def test_t74_usage_flows_to_the_model_step() -> None:
    usage = Usage(input_tokens=10, output_tokens=5)
    r = build_test_orchestrator(
        lambda llm: (lambda t: [llm.complete([Message("user", t)]) for _ in range(2)]),
        (),
        llm=FakeLLM(script=(FakeReply(Message("assistant", "ok"), usage=usage), "no usage")),
    ).run("q")
    first, second = r.run.steps
    assert first.usage == usage and second.usage is None
    assert r.run.usage.input_tokens == 10 and r.run.usage.output_tokens == 5


def test_handler_receives_messages_and_tool_names_and_may_return_str() -> None:
    seen: list[Any] = []

    def handler(messages: Any, tools: Any) -> str:
        seen.append((messages, tools))
        return "handled"

    r = build_test_orchestrator(
        lambda llm, tools: (lambda t: llm.complete([Message("user", t)], list(tools)).content),
        {"a": {}, "b": {}},
        llm=FakeLLM(handler=handler),
    ).run("q")
    assert r.run.final_text == "handled" and seen[0][1] == ("a", "b")


def test_async_client_and_async_handler() -> None:
    async def handler(messages: Any, tools: Any) -> str:
        await asyncio.sleep(0)
        return "async handled"

    async def factory(llm: Any) -> Any:
        async def agent(t: str) -> str:
            return (await llm.acomplete([Message("user", t)])).content

        return agent

    r = build_test_orchestrator(factory, (), llm=FakeLLM(handler=handler)).run("q")
    assert r.run.final_text == "async handled"


def test_the_client_is_fresh_for_every_run() -> None:
    orch = build_test_orchestrator(
        make_llm_refund_agent, FAKES, llm=FakeLLM(script=refund_script())
    )
    assert orch.run("refund A12").status == "finished"
    assert orch.run("refund A12").status == "finished"  # the script restarts


def test_the_script_is_a_frozen_value() -> None:
    llm = FakeLLM(script=refund_script())
    assert isinstance(llm.script, tuple)
    assert from_dict(Message, to_dict(llm.script[0].message)) == llm.script[0].message
    with pytest.raises(dataclasses.FrozenInstanceError):
        llm.name = "x"  # type: ignore[misc]


def test_a_model_step_gets_the_reply_of_its_own_position_when_threads_interleave() -> None:
    first_started, second_done = threading.Event(), threading.Event()

    def hold_the_first_call(event: StepEvent) -> None:
        if event.phase == "start" and event.seq == 1:
            first_started.set()
            second_done.wait(0.2)

    def factory(llm: Any) -> Any:
        def late_call() -> None:
            first_started.wait(5)
            llm.complete(["b"])
            second_done.set()

        def agent(text: str) -> None:
            other = threading.Thread(target=late_call)
            other.start()
            llm.complete(["a"])
            other.join()

        return agent

    orch = build_test_orchestrator(factory, llm=FakeLLM(script=("first", "second")))
    orch.add_listener(hold_the_first_call)
    r = orch.run("go")
    assert [(s.seq, s.result["content"]) for s in r.run.steps] == [(1, "first"), (2, "second")]
