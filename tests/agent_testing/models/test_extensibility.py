"""Guards for "add a parameter later" (plans/01, section 3a)."""

import copy
import dataclasses
import pickle
from dataclasses import KW_ONLY, dataclass

import pytest

import disseqt_agent_testing
from disseqt_agent_testing import models
from disseqt_agent_testing.models import (
    FrozenDict,
    Message,
    ModelValidationError,
    Step,
    _frozen,
    check_supported_hints,
    from_dict,
    run,
    to_dict,
)
from disseqt_agent_testing.models.matchers import Matcher

from .factories import every_model_instance, rich_case, rich_result

# Identity fields (the only positional ones) per class. Everything else must be keyword-only so a
# new defaulted field can be added anywhere without breaking callers.
POSITIONAL = {
    "MsgToolCall": ("id", "name"),
    "Message": ("role", "content"),
    "ToolExpect": ("name", "args"),
    "ForbiddenCall": ("name", "args"),
    "FakeResponse": (),
    "FakeSpec": ("name",),
    "Expect": (),
    "AgentTestCase": ("id", "input"),
    "Verdict": ("dimension", "passed", "score", "reason"),
    "Usage": (),
    "Step": ("seq", "kind", "name"),
    "ErrorInfo": ("type", "module", "message"),
    "Violation": ("code", "message"),
    "ToolInfo": ("name", "kind", "effect"),
    "AgentRun": (),
    "RunResult": ("status", "run"),
    "StepEvent": ("phase", "trace_id", "seq", "call_id", "kind", "name", "turn"),
    "Matcher": (),
    "Eq": ("value",),
    "OneOf": ("values",),
    "AnyValue": (),
    "Absent": (),
    "Regex": ("pattern",),
    "Approx": ("value",),
    "Ref": ("source",),
    "Optional_": ("inner",),
    "Normalized": ("values",),
    "TypeIs": ("types",),
    "Unordered": ("items",),
    "Contains": ("items",),
    "DictMatch": ("fields",),
    "DateEq": ("value",),
    "Predicate": ("name",),
    "Semantic": ("reference",),
    "RouteExpect": (),
    "LoopExpect": (),
    "PlanExpect": (),
    "ClarifyExpect": (),
    "RefuseExpect": (),
    "ScopeExpect": ("in_scope",),
    "AnswerQualityExpect": ("reference",),
    "GroundednessExpect": (),
}

WITH_EXTRA = {
    "AgentTestCase", "Expect", "ToolExpect", "ForbiddenCall", "FakeSpec", "FakeResponse",
    "Step", "AgentRun", "RunResult", "ToolInfo", "Verdict", "StepEvent",
    "RouteExpect", "LoopExpect", "PlanExpect", "ClarifyExpect", "RefuseExpect", "ScopeExpect",
    "AnswerQualityExpect", "GroundednessExpect",
}  # fmt: skip


def model_classes() -> list[type]:
    return [
        value
        for value in vars(models).values()
        if isinstance(value, type) and dataclasses.is_dataclass(value)
    ]


def test_every_model_class_is_covered_by_the_layout_table():
    names = {cls.__name__ for cls in model_classes()}
    assert names == set(POSITIONAL), names ^ set(POSITIONAL)


@pytest.mark.parametrize("cls", model_classes(), ids=lambda c: c.__name__)
def test_only_identity_fields_are_positional(cls):
    positional = tuple(f.name for f in dataclasses.fields(cls) if f.init and not f.kw_only)
    assert positional == POSITIONAL[cls.__name__]


@pytest.mark.parametrize("cls", model_classes(), ids=lambda c: c.__name__)
def test_every_model_is_frozen_and_slotted(cls):
    assert cls.__dataclass_params__.frozen is True
    assert "__slots__" in vars(cls)


@pytest.mark.parametrize("cls", model_classes(), ids=lambda c: c.__name__)
def test_every_field_uses_a_supported_hint_shape(cls):
    assert check_supported_hints(cls) == []


def test_a_new_defaulted_field_needs_no_serde_change():
    @dataclass(slots=True, frozen=True)
    class Probe:
        a: str
        _: KW_ONLY
        b: int = 0
        added_later: tuple[str, ...] = ()
        extra_map: "dict[str, int]" = dataclasses.field(default_factory=dict)

    probe = Probe("x", added_later=("y",))
    assert from_dict(Probe, to_dict(probe)) == probe
    assert from_dict(Probe, {"a": "x"}) == Probe("x")  # old data without the new field still loads


def test_extra_bags_exist_exactly_where_planned():
    names = {
        cls.__name__
        for cls in model_classes()
        if "extra" in {f.name for f in dataclasses.fields(cls)}
    }
    assert names == WITH_EXTRA


def minimal(cls_name: str) -> dict:
    return {
        "AgentTestCase": lambda **kw: models.AgentTestCase("c", "hi", **kw),
        "Expect": lambda **kw: models.Expect(**kw),
        "ToolExpect": lambda **kw: models.ToolExpect("x", **kw),
        "ForbiddenCall": lambda **kw: models.ForbiddenCall("x", **kw),
        "FakeSpec": lambda **kw: models.FakeSpec("x", **kw),
        "FakeResponse": lambda **kw: models.FakeResponse(**kw),
        "Step": lambda **kw: Step(1, "model", "m", **kw),
        "AgentRun": lambda **kw: models.AgentRun(**kw),
        "RunResult": lambda **kw: models.RunResult("finished", models.AgentRun(), **kw),
        "ToolInfo": lambda **kw: models.ToolInfo("x", "tool", "read", **kw),
        "Verdict": lambda **kw: models.Verdict("d", True, 1.0, "r", **kw),
        "StepEvent": lambda **kw: models.StepEvent("start", "t", 1, "call-1", "tool", "x", 0, **kw),
        "RouteExpect": lambda **kw: models.RouteExpect(mode="exact", **kw),
        "LoopExpect": lambda **kw: models.LoopExpect(**kw),
        "PlanExpect": lambda **kw: models.PlanExpect(**kw),
        "ClarifyExpect": lambda **kw: models.ClarifyExpect(**kw),
        "RefuseExpect": lambda **kw: models.RefuseExpect(**kw),
        "ScopeExpect": lambda **kw: models.ScopeExpect(True, **kw),
        "AnswerQualityExpect": lambda **kw: models.AnswerQualityExpect("r", **kw),
        "GroundednessExpect": lambda **kw: models.GroundednessExpect(**kw),
    }[cls_name]


@pytest.mark.parametrize("cls_name", sorted(WITH_EXTRA))
def test_extra_bag_behaviour(cls_name):
    make = minimal(cls_name)
    item = make(extra={"acme.ticket": "T-1", "nested": {"a": [1, None]}})
    assert item.extra == {"acme.ticket": "T-1", "nested": {"a": (1, None)}}
    assert isinstance(item.extra, FrozenDict)
    assert make().extra is not make().extra  # no shared default container
    with pytest.raises(ModelValidationError, match=r"\.extra"):
        make(extra={"bad": float("nan")})
    with pytest.raises(ModelValidationError, match=r"\.extra"):
        make(extra=[1])
    assert from_dict(type(item), to_dict(item)) == item


def test_extra_round_trips_through_the_unknown_policy():
    data = to_dict(rich_case())
    data["owner"] = "me"
    case = from_dict(type(rich_case()), data, unknown="extra")
    assert case.extra["owner"] == "me" and case.extra["acme.ticket"] == "T-1"


def test_public_api_snapshot():
    assert sorted(models.__all__) == [
        "Absent",
        "AgentRun",
        "AgentTestCase",
        "AnswerQualityExpect",
        "AnyValue",
        "Approx",
        "ClarifyExpect",
        "Contains",
        "DateEq",
        "Decision",
        "DictMatch",
        "EFFECTS",
        "Effect",
        "Eq",
        "ErrorInfo",
        "Expect",
        "FAKE_KINDS",
        "FakeKind",
        "FakeResponse",
        "FakeSpec",
        "ForbiddenCall",
        "FrozenDict",
        "GroundednessExpect",
        "JSONValue",
        "LoopExpect",
        "Matcher",
        "Message",
        "ModelValidationError",
        "MsgToolCall",
        "Normalized",
        "OneOf",
        "Optional_",
        "PlanExpect",
        "Predicate",
        "RUN_STATUSES",
        "Ref",
        "RefuseExpect",
        "Regex",
        "Role",
        "RouteExpect",
        "RunResult",
        "STEP_KINDS",
        "STEP_KIND_TO_SPAN_KIND",
        "ScopeExpect",
        "Semantic",
        "Step",
        "StepEvent",
        "ToolExpect",
        "ToolInfo",
        "TypeIs",
        "Unordered",
        "Usage",
        "VIOLATION_CODES",
        "Verdict",
        "Violation",
        "capture_json",
        "check_supported_hints",
        "freeze_json",
        "from_dict",
        "parse_ref_source",
        "register_matcher",
        "register_migration",
        "thaw",
        "to_dict",
        "validate_tool_correlation",
    ]
    assert disseqt_agent_testing.__all__ == models.__all__
    for name in models.__all__:
        assert hasattr(models, name) and hasattr(disseqt_agent_testing, name), name


def test_step_kinds_come_from_one_constant(monkeypatch):
    with pytest.raises(ModelValidationError, match="Step.kind"):
        Step(1, "rag", "x")
    monkeypatch.setattr(run, "STEP_KINDS", (*run.STEP_KINDS, "rag"))
    assert Step(1, "rag", "retriever").kind == "rag"  # adding a kind needs no validator edit


def test_models_import_nothing_outside_the_standard_library_and_this_package():
    import ast
    import sys
    from pathlib import Path

    package_dir = Path(models.__file__).parent
    allowed_third_party = {"typing_extensions"}
    forbidden = {"pydantic", "disseqt_sdk", "disseqt_agentic_sdk", "requests", "wrapt", "yaml"}
    stdlib = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else None
    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                assert name not in forbidden, f"{path.name} imports {name}"
                if stdlib is not None:
                    assert name in stdlib or name in allowed_third_party, f"{path.name}: {name}"


@pytest.mark.parametrize("obj", every_model_instance(), ids=lambda o: type(o).__name__)
def test_models_survive_copy_and_pickle(obj):
    assert copy.deepcopy(obj) == obj
    assert pickle.loads(pickle.dumps(obj)) == obj


def test_two_instances_never_share_default_containers():
    assert models.Expect().min_scores is not models.Expect().min_scores
    assert models.AgentRun().state is not models.AgentRun().state
    assert isinstance(Message("user", "x"), Matcher) is False
    assert _frozen.FrozenDict() is not _frozen.FrozenDict()


def test_rich_instances_are_hashable():
    hash(rich_case())
    hash(rich_result())
