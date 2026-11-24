"""Factory injection: T17-T26 (what is passed, and the exact error texts)."""

import functools
from collections.abc import Callable
from typing import Any

import pytest

from disseqt_agent_testing.harness import FakeLLM, HarnessConfigError, build_test_orchestrator
from tests.agent_testing._support.refund_agent import FAKES, make_refund_agent

from .helpers import refund


def build(factory: Callable[..., Any], **options: Any) -> Any:
    return build_test_orchestrator(factory, FAKES, **options)


def captured(factory_params: str) -> dict[str, Any]:
    box: dict[str, Any] = {}
    ns: dict[str, Any] = {"box": box}
    exec(  # noqa: S102 - building factories with specific signatures
        f"def factory({factory_params}):\n    box.update(locals())\n    return lambda t: 'ok'\n", ns
    )
    build(
        ns["factory"],
        **(
            {"llm": FakeLLM(script=("x",))}
            if ("llm" in factory_params or "**" in factory_params)
            else {}
        ),
    ).run("go")
    return box


def test_t17_tools_dict_has_sorted_keys() -> None:
    box = captured("tools")
    assert list(box["tools"]) == ["check_policy", "get_order", "issue_refund"]
    assert all(callable(v) for v in box["tools"].values())


def test_t18_each_fake_by_name() -> None:
    box = captured("get_order, issue_refund, check_policy")
    assert all(callable(box[n]) for n in ("get_order", "issue_refund", "check_policy"))


def test_t19_var_keyword_receives_slots_only() -> None:
    box = captured("**kw")["kw"]
    assert set(box) == {"tools", "subagents", "llm"}


def test_t20_unknown_required_parameter() -> None:
    with pytest.raises(HarnessConfigError) as err:
        build(lambda tools, db: None)
    assert str(err.value) == (
        "factory parameter 'db' has no default and is not injectable. "
        "Factory parameters: ['tools', 'db']. "
        "Injectable names: ['llm', 'subagents', 'tools', 'check_policy', 'get_order', "
        "'issue_refund']."
    )


def test_t21_positional_only_parameter() -> None:
    def factory(tools: Any, /) -> None: ...

    with pytest.raises(HarnessConfigError, match="is positional-only and has no default"):
        build(factory)


def test_t22_a_fake_the_factory_cannot_receive() -> None:
    with pytest.raises(HarnessConfigError) as err:
        build(lambda get_order: None)
    assert str(err.value).startswith(
        "fake 'check_policy' (kind=tool) cannot be injected: the factory declares neither "
        "'tools', 'check_policy' nor **kwargs."
    )


def test_t23_list_shape_in_name_order() -> None:
    def factory(tools: "list[Callable[..., Any]]") -> Any:
        box["tools"] = tools
        return lambda t: "ok"

    box: dict[str, Any] = {}
    build(factory).run("go")
    assert isinstance(box["tools"], list) and len(box["tools"]) == 3
    assert [t.__name__ for t in box["tools"]] == ["check_policy", "get_order", "issue_refund"]


def test_t24_framework_tool_annotation_needs_an_adapter() -> None:
    def factory(tools: "list[BaseTool]") -> None: ...  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(HarnessConfigError, match="this version supports dict\\[str, Callable\\]"):
        build(factory)


def test_t25_llm_parameter_without_a_fake_llm() -> None:
    with pytest.raises(HarnessConfigError, match="no FakeLLM was configured"):
        build(lambda tools, llm: None)


def test_llm_configured_but_not_declared() -> None:
    with pytest.raises(HarnessConfigError, match="neither 'llm' nor \\*\\*kwargs"):
        build(lambda tools: None, llm=FakeLLM(script=("x",)))


def test_t26_partial_keyword_is_not_injected() -> None:
    factory = functools.partial(make_refund_agent, bug="wrong_order")
    assert build(factory).run("refund order A12").run.final_text == "Refunded 59.99 for Z999."


def test_defaults_are_left_alone() -> None:
    box: dict[str, Any] = {}

    def factory(tools: Any, retries: int = 3) -> Any:
        box["retries"] = retries
        return lambda t: "ok"

    build(factory).run("go")
    assert box["retries"] == 3


def test_class_factories_are_supported() -> None:
    class Agent:
        def __init__(self, tools: Any) -> None:
            self.tools = tools

        def run(self, text: str) -> str:
            return str(sorted(self.tools))

    assert refund().run("x").status == "finished"
    assert build(Agent).run("go").run.final_text == "['check_policy', 'get_order', 'issue_refund']"
