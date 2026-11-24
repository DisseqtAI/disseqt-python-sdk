"""Extensibility guards (plan 02 section 6a): T59-T69. New parameters must never break callers."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing import harness
from disseqt_agent_testing.harness import (
    AgentInput,
    BaseAdapter,
    FakeLLM,
    FakeReply,
    KindSpec,
    OrchestratorConfig,
    OrchestratorOptions,
    RunOptions,
    SlotSpec,
    build_test_orchestrator,
    register_adapter,
    register_fake_kind,
    register_slot,
    unregister_adapter,
    unregister_fake_kind,
    unregister_slot,
)
from disseqt_agent_testing.harness.config import config_field_names
from disseqt_agent_testing.harness.registry import ADAPTERS, FAKE_KIND_SPECS, SLOTS
from disseqt_agent_testing.models import (
    AgentRun,
    FakeResponse,
    FrozenDict,
    RunResult,
    StepEvent,
    from_dict,
    to_dict,
)

from .helpers import refund

GOLDEN = Path(__file__).parent / "golden" / "harness_run_v1.json"


def test_t59_options_typeddict_matches_the_config_fields() -> None:
    assert set(OrchestratorOptions.__annotations__) == set(config_field_names())
    assert {f.name for f in dataclasses.fields(OrchestratorConfig)} == set(config_field_names())


def test_t60_unknown_option_lists_the_valid_ones() -> None:
    with pytest.raises(TypeError) as err:
        build_test_orchestrator(lambda tools: None, (), timeout=5)  # type: ignore[call-arg]
    text = str(err.value)
    assert text.startswith("unknown option 'timeout'; valid options: [")
    assert "timeout_s" in text and "assert_all_called" in text


def test_t61_keyword_options_win_over_the_config_object() -> None:
    orch = build_test_orchestrator(
        lambda tools: None, (), config=OrchestratorConfig(timeout_s=5.0), timeout_s=7.0
    )
    assert orch.config.timeout_s == 7.0
    assert (
        build_test_orchestrator(
            lambda tools: None, (), config=OrchestratorConfig(timeout_s=5.0)
        ).config.timeout_s
        == 5.0
    )


def test_t62_run_options_apply_per_run_and_land_in_extra() -> None:
    import time

    orch = build_test_orchestrator(lambda tools: (lambda t: time.sleep(5)), (), timeout_s=60)
    r = orch.run("go", options=RunOptions(timeout_s=0.2, extra={"acme.k": 1}))
    assert r.status == "timed_out" and r.run.extra["acme.k"] == 1


def test_t63_a_registered_slot_is_injected_and_removed_again() -> None:
    before = dict(SLOTS)
    register_slot(SlotSpec("memory", lambda run, shape: {"k": 1}))
    try:
        r = build_test_orchestrator(lambda memory: (lambda t: memory["k"]), ()).run("go")
        assert r.run.final_answer == 1
        with pytest.raises(harness.HarnessConfigError, match="already registered"):
            register_slot(SlotSpec("memory", lambda run, shape: 1))
    finally:
        unregister_slot("memory")
    assert SLOTS == before


def test_built_in_fake_kinds_match_the_model_constants() -> None:
    from disseqt_agent_testing.models import FAKE_KINDS

    assert set(FAKE_KINDS) == set(FAKE_KIND_SPECS)


def test_t64_a_new_fake_kind_is_one_constant_plus_one_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from disseqt_agent_testing.models import FakeSpec, run, test_case

    # FAKE_KINDS is imported by value in the two modules that validate it (FakeSpec, ToolInfo).
    for module in (test_case, run):
        monkeypatch.setattr(module, "FAKE_KINDS", (*module.FAKE_KINDS, "mcp"))
    register_fake_kind(KindSpec("mcp", "tools", "tool"))
    try:
        spec = FakeSpec("lookup", kind="mcp", effect="read", default=FakeResponse(returns=1))  # type: ignore[arg-type]
        r = build_test_orchestrator(lambda tools: (lambda t: tools["lookup"]()), (spec,)).run("go")
        assert r.status == "finished"
        assert (r.run.steps[0].kind, r.run.steps[0].name) == ("tool", "lookup")  # mapped step kind
    finally:
        unregister_fake_kind("mcp")
    assert "mcp" not in FAKE_KIND_SPECS


def test_t65_a_minimal_adapter_works_through_the_base_defaults() -> None:
    class Handle:
        def go(self, text: str) -> str:
            return "adapted " + text

    class MyAdapter(BaseAdapter):
        name = "mine"

        def matches(self, product: Any) -> bool:
            return isinstance(product, Handle)

        def bind_entry(self, product: Any, entry: Any) -> Any:
            from disseqt_agent_testing.harness import BoundEntry

            return BoundEntry(product.go, False, "go")

    adapter = MyAdapter()
    register_adapter(adapter)
    try:
        r = build_test_orchestrator(
            lambda tools: Handle(),
            (),
        ).run("x")
        assert r.run.final_text == "adapted x"
    finally:
        unregister_adapter(adapter)
    assert adapter not in ADAPTERS
    assert adapter.shape_input(AgentInput("hello")) == "hello"  # base default


def test_t66_an_unknown_status_string_is_a_model_error_not_a_silent_pass() -> None:
    from disseqt_agent_testing.models import ModelValidationError

    run = refund().run("refund order A12").run
    with pytest.raises(ModelValidationError, match="status"):
        RunResult("surprising", run)


def test_t67_events_use_only_known_phases_and_the_model_rejects_others() -> None:
    from disseqt_agent_testing.models import ModelValidationError

    events: list[str] = []
    orch = refund()
    orch.add_listener(lambda e: events.append(e.phase))
    assert orch.run("refund order A12").status == "finished"
    assert set(events) == {"start", "end"}
    # Deviation from the plan text: a synthetic unknown phase cannot be built (StepEvent validates
    # its phase), so a new phase needs a model change, which the golden files then cover.
    with pytest.raises(ModelValidationError, match="StepEvent.phase"):
        StepEvent("heartbeat", "t", 1, "call-1", "tool", "x", 0)


HARNESS_DATACLASSES = {
    OrchestratorConfig: (),
    RunOptions: (),
    AgentInput: ("text",),
    FakeReply: ("message",),
    FakeLLM: (),
    SlotSpec: ("name", "builder", "via_kwargs", "kinds", "optional"),
}


@pytest.mark.parametrize("cls", list(HARNESS_DATACLASSES), ids=lambda c: c.__name__)
def test_t68_layout_only_identity_fields_are_positional(cls: type) -> None:
    positional = tuple(f.name for f in dataclasses.fields(cls) if f.init and not f.kw_only)
    assert positional == HARNESS_DATACLASSES[cls]


@pytest.mark.parametrize(
    "cls",
    [OrchestratorConfig, RunOptions, AgentInput, FakeReply, FakeLLM],
    ids=lambda c: c.__name__,
)
def test_t68_new_value_objects_have_an_extra_bag(cls: type) -> None:
    assert "extra" in {f.name for f in dataclasses.fields(cls)}


def test_t68_value_objects_are_frozen() -> None:
    for obj in (
        OrchestratorConfig(),
        RunOptions(),
        AgentInput("x"),
        FakeReply(
            harness.FakeReply.__annotations__
            and __import__("disseqt_agent_testing").models.Message("assistant", "x")
        ),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.extra = {}  # type: ignore[misc,attr-defined]


def deterministic_run() -> RunResult:
    ticks = iter(range(0, 10**12, 1_000_000))
    orch = refund(clock=lambda: next(ticks), id_factory=lambda: "golden-trace-id")
    result = orch.run("refund order A12")
    # started_at is wall-clock; pin it so the file is reproducible
    run = dataclasses.replace(result.run, started_at="2026-01-01T00:00:00.000Z")
    return dataclasses.replace(result, run=run, duration_ms=0.0, setup_ms=0.0)


def test_t69_golden_run_file_loads_and_round_trips() -> None:
    data = json.loads(GOLDEN.read_text())
    loaded = from_dict(RunResult, data)
    assert to_dict(loaded) == data  # old content preserved, in order
    assert isinstance(loaded.run, AgentRun) and loaded.status == "finished"
    assert loaded.run.final_text == "Refunded 59.99 for A12."


def test_t69_the_harness_still_produces_what_the_golden_file_holds() -> None:
    fresh = to_dict(deterministic_run())
    stored = json.loads(GOLDEN.read_text())
    assert fresh == stored, "re-generate harness_run_v1.json only on a deliberate schema change"


def test_run_extra_is_frozen_and_a_mapping() -> None:
    r = refund().run("refund order A12", options=RunOptions(extra={"a": {"b": 1}}))
    assert isinstance(r.run.extra, FrozenDict)
