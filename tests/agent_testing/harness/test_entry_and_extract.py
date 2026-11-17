"""Entry-point resolution and answer extraction: T27-T31, T36, T37 and the extraction rules."""

import asyncio
import dataclasses
import sys
import types
from typing import Any

import pytest

from disseqt_agent_testing.harness import HarnessConfigError, build_test_orchestrator
from disseqt_agent_testing.harness.extract import extract_answer, final_text
from disseqt_agent_testing.models import Message


def run(factory: Any, **options: Any) -> Any:
    return build_test_orchestrator(factory, {"t": {}}, **options).run("go")


def test_t27_candidate_precedence() -> None:
    class Both:
        def run(self, text: str) -> str:
            return "run"

        def __call__(self, text: str) -> str:
            return "call"

    class AsyncFirst:
        async def arun(self, text: str) -> str:
            return "arun"

        def run(self, text: str) -> str:
            return "run"

    assert run(lambda tools: Both()).run.final_text == "run"
    assert run(lambda tools: AsyncFirst()).run.final_text == "arun"


def test_t28_explicit_entry_by_name_by_callable_and_missing() -> None:
    class Agent:
        def handle(self, text: str) -> str:
            return f"handled {text}"

        def go(self, text: str) -> str:
            return f"went {text}"

    assert run(lambda tools: Agent(), entry="handle").run.final_text == "handled go"
    assert run(lambda tools: Agent(), entry=lambda a, t: a.go(t)).run.final_text == "went go"
    with pytest.raises(HarnessConfigError, match="entry='nope' not found on Agent"):
        run(lambda tools: Agent(), entry="nope")


def test_t29_no_candidate_gives_the_exact_message() -> None:
    class Empty:
        pass

    with pytest.raises(HarnessConfigError) as err:
        run(lambda tools: Empty())
    assert str(err.value) == (
        "could not find an entry point on Empty: tried "
        "arun, run, ainvoke, invoke, __call__. Pass entry='method_name' or "
        "entry=callable(agent, input)."
    )


def test_t30_framework_objects_without_an_adapter_are_refused() -> None:
    module = types.ModuleType("langgraph")
    module.__file__ = "/x/site-packages/langgraph/__init__.py"
    sub = types.ModuleType("langgraph.graph")
    cls = type("CompiledGraph", (), {"__module__": "langgraph.graph", "invoke": lambda s, t: "x"})
    sys.modules["langgraph"], sys.modules["langgraph.graph"] = module, sub
    try:
        with pytest.raises(HarnessConfigError, match="looks like a langgraph object"):
            run(lambda tools: cls())
        # an explicit entry opts in
        assert run(lambda tools: cls(), entry="invoke").status == "finished"
    finally:
        del sys.modules["langgraph"], sys.modules["langgraph.graph"]


def test_a_projects_own_package_named_like_a_framework_is_not_a_framework() -> None:
    module = types.ModuleType("agents")
    module.__file__ = "/home/me/project/agents/__init__.py"
    cls = type("Mine", (), {"__module__": "agents", "run": lambda s, t: "mine"})
    sys.modules["agents"] = module
    try:
        assert run(lambda tools: cls()).run.final_text == "mine"
    finally:
        del sys.modules["agents"]


def test_t31_a_class_is_refused() -> None:
    class Agent:
        def run(self, text: str) -> str:
            return "x"

    with pytest.raises(HarnessConfigError, match="agent_factory returned a class \\(Agent\\)"):
        run(lambda tools: Agent)


def test_async_factory_and_entry() -> None:
    async def factory(tools: Any) -> Any:
        async def agent(text: str) -> str:
            return "async " + text

        return agent

    assert run(factory).run.final_text == "async go"


@dataclasses.dataclass
class Answer:
    value: int
    tag: str = "t"


class Ai:
    role = "ai"

    def __init__(self, content: Any) -> None:
        self.content = content


def test_t36_extraction_precedence() -> None:
    assert extract_answer("plain").answer == "plain"
    assert extract_answer(Message("assistant", "from message")).answer == "from message"
    state = {"messages": [{"role": "user", "content": "q"}, Ai("final")]}
    assert extract_answer(state).answer == "final"
    assert extract_answer({"output": {"k": 1}}).answer == {"k": 1}
    assert extract_answer(Answer(3)).answer == {"value": 3, "tag": "t"}
    none = extract_answer(None)
    assert none.answer is None and none.warnings == ("agent returned None",)
    assert (
        extract_answer(Ai([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])).answer
        == "ab"
    )
    assert extract_answer({"foo": 1}).answer == {"foo": 1}


def test_unserializable_answers_become_lossy_text() -> None:
    out = extract_answer(object())
    assert isinstance(out.answer, str) and out.warnings == (
        "final answer is not JSON-serializable; stored lossy repr",
    )


def test_final_text_of_structured_answers_is_canonical_json() -> None:
    assert final_text(extract_answer({"b": 1, "a": 2}).answer) == '{"a": 2, "b": 1}'
    assert final_text(None) is None


def test_t37_a_failing_user_extract_is_a_warning_not_a_crash() -> None:
    def extract(raw: Any) -> Any:
        raise KeyError("x")

    r = run(lambda tools: (lambda t: "ok"), extract=extract)
    assert r.status == "finished" and r.run.final_answer is None
    assert "extract failed: KeyError: 'x'" in r.warnings


def test_user_extract_wins_over_everything() -> None:
    r = run(lambda tools: (lambda t: {"output": "no"}), extract=lambda raw: "yes")
    assert r.run.final_text == "yes"


def test_generators_and_awaitables_are_drained_to_the_last_item() -> None:
    def gen(tools: Any) -> Any:
        def agent(text: str) -> Any:
            yield "partial"
            yield "last"

        return agent

    async def coro(tools: Any) -> Any:
        async def agent(text: str) -> Any:
            await asyncio.sleep(0)
            return "awaited"

        return agent

    def sync_returning_coroutine(tools: Any) -> Any:
        async def inner(text: str) -> str:
            return "inner"

        return lambda t: inner(t)

    assert run(gen).run.final_text == "last"
    assert run(coro).run.final_text == "awaited"
    assert run(sync_returning_coroutine).run.final_text == "inner"
