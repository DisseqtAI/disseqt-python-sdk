"""Run models: what happened when an agent was run.

All frozen. The harness (plan 02) records steps privately and builds one ``AgentRun`` and one
``RunResult`` at the end; nothing outside the recorder ever holds a mutable trajectory.
"""

from dataclasses import KW_ONLY, dataclass, field
from typing import ClassVar

from ._constants import EFFECTS, FAKE_KINDS, Effect
from ._frozen import FrozenDict, JSONObject, JSONValue, freeze_json, freeze_mapping
from ._validate import (
    as_tuple,
    check_bool,
    check_choice,
    check_int,
    check_number,
    check_opt_int,
    check_opt_number,
    check_opt_str,
    check_str,
    fail,
    set_field,
    str_tuple,
)
from .message import Message

__all__ = [
    "AgentRun",
    "Effect",
    "ErrorInfo",
    "RunResult",
    "RUN_STATUSES",
    "Step",
    "STEP_KIND_TO_SPAN_KIND",
    "STEP_KINDS",
    "StepEvent",
    "ToolInfo",
    "Usage",
    "VIOLATION_CODES",
    "Violation",
]

# One place to add a step kind; validation derives from these constants.
STEP_KINDS: tuple[str, ...] = ("tool", "subagent", "model")
# Maps a step kind to the matching ``disseqt_agentic_sdk.SpanKind`` member name (checked by a
# test), so the recorder can mirror steps to spans later without coupling run data to span names.
STEP_KIND_TO_SPAN_KIND: dict[str, str] = {
    "tool": "TOOL_EXEC",
    "subagent": "AGENT_EXEC",
    "model": "MODEL_EXEC",
}
# "aborted": the harness stopped a runaway agent (see harness max_recorded_steps)
RUN_STATUSES: tuple[str, ...] = ("finished", "crashed", "setup_failed", "timed_out", "aborted")
VIOLATION_CODES: tuple[str, ...] = (
    "unexpected_call",
    "unused_response",
    "listener_error",
    "steps_cap_exceeded",
)
_CALL_STEP_KINDS = ("tool", "subagent")


@dataclass(slots=True, frozen=True)
class Usage:
    """Token, latency and cost figures. Fields that are unknown stay at 0."""

    _: KW_ONLY
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        check_int(self.input_tokens, "Usage.input_tokens", minimum=0)
        check_int(self.output_tokens, "Usage.output_tokens", minimum=0)
        set_field(self, "latency_s", check_number(self.latency_s, "Usage.latency_s", minimum=0))
        set_field(self, "cost_usd", check_number(self.cost_usd, "Usage.cost_usd", minimum=0))


@dataclass(slots=True, frozen=True)
class Step:
    """One recorded call: a tool, a sub-agent or a model call."""

    seq: int
    kind: str
    name: str
    _: KW_ONLY
    call_id: str = ""
    provider_call_id: str | None = None
    args: JSONObject = field(default_factory=FrozenDict)
    result: JSONValue = None
    error: str | None = None
    effect: Effect | None = None
    ts: float = 0.0
    duration_ms: float | None = None
    turn: int = 0
    parallel_group: int | None = None
    usage: Usage | None = None
    lossy: bool = False
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_int(self.seq, "Step.seq", minimum=1)
        check_choice(self.kind, "Step.kind", STEP_KINDS)
        check_str(self.name, "Step.name")
        call_id = self.call_id or f"call-{self.seq}"
        set_field(self, "call_id", check_str(call_id, "Step.call_id"))
        check_opt_str(self.provider_call_id, "Step.provider_call_id")
        set_field(self, "args", freeze_mapping(self.args, "Step.args"))
        set_field(self, "result", freeze_json(self.result, "Step.result"))
        check_opt_str(self.error, "Step.error")
        if self.kind in _CALL_STEP_KINDS:
            if self.effect is None:
                fail("Step.effect", f"is required for {self.kind!r} steps")
            check_choice(self.effect, "Step.effect", EFFECTS)
        elif self.effect is not None:
            fail("Step.effect", f"must be None for {self.kind!r} steps (they have no effect)")
        set_field(self, "ts", check_number(self.ts, "Step.ts", minimum=0))
        set_field(
            self,
            "duration_ms",
            check_opt_number(self.duration_ms, "Step.duration_ms", minimum=0),
        )
        check_int(self.turn, "Step.turn", minimum=0)
        check_opt_int(self.parallel_group, "Step.parallel_group")
        if self.usage is not None and not isinstance(self.usage, Usage):
            fail("Step.usage", "must be a Usage")
        check_bool(self.lossy, "Step.lossy")
        set_field(self, "extra", freeze_mapping(self.extra, "Step.extra"))


@dataclass(slots=True, frozen=True)
class ErrorInfo:
    """An exception from a run. ``message`` is ``"<Type>: <str(exc)>"``."""

    type: str
    module: str
    message: str
    _: KW_ONLY
    traceback: str | None = None

    def __post_init__(self) -> None:
        check_str(self.type, "ErrorInfo.type")
        check_str(self.module, "ErrorInfo.module", non_empty=False)
        check_str(self.message, "ErrorInfo.message")
        check_opt_str(self.traceback, "ErrorInfo.traceback")


@dataclass(slots=True, frozen=True)
class Violation:
    """A harness-detected problem the agent may have swallowed (for example an unmatched call)."""

    code: str
    message: str
    _: KW_ONLY
    fake: str | None = None
    call_id: str | None = None

    def __post_init__(self) -> None:
        check_choice(self.code, "Violation.code", VIOLATION_CODES)
        check_str(self.message, "Violation.message")
        check_opt_str(self.fake, "Violation.fake")
        check_opt_str(self.call_id, "Violation.call_id")


@dataclass(slots=True, frozen=True)
class ToolInfo:
    """A tool or sub-agent the agent was offered during a run."""

    name: str
    kind: str
    effect: Effect
    _: KW_ONLY
    description: str = ""
    schema: JSONObject | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_str(self.name, "ToolInfo.name")
        check_choice(self.kind, "ToolInfo.kind", FAKE_KINDS)
        check_choice(self.effect, "ToolInfo.effect", EFFECTS)
        check_str(self.description, "ToolInfo.description", non_empty=False)
        if self.schema is not None:
            set_field(self, "schema", freeze_mapping(self.schema, "ToolInfo.schema"))
        set_field(self, "extra", freeze_mapping(self.extra, "ToolInfo.extra"))


@dataclass(slots=True, frozen=True)
class AgentRun:
    """Everything recorded about one agent run. A versioned document (``SCHEMA_VERSION``)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    _: KW_ONLY
    case_id: str = ""
    input: "str | tuple[Message, ...]" = ""
    follow_ups: tuple[str, ...] = ()
    steps: tuple[Step, ...] = ()
    final_answer: JSONValue = None
    final_text: str | None = None
    turn_outputs: tuple[JSONValue, ...] = ()
    messages: tuple[Message, ...] = ()
    available_tools: tuple[ToolInfo, ...] = ()
    model_steps_captured: bool = False
    started_at: str = ""
    state: JSONObject = field(default_factory=FrozenDict)  # reserved: sandbox snapshot, later
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_str(self.case_id, "AgentRun.case_id", non_empty=False)
        if not isinstance(self.input, str):
            items = as_tuple(self.input, "AgentRun.input")
            for index, item in enumerate(items):
                if not isinstance(item, Message):
                    fail(f"AgentRun.input[{index}]", "must be a Message")
            set_field(self, "input", items)
        set_field(self, "follow_ups", str_tuple(self.follow_ups, "AgentRun.follow_ups"))
        steps = as_tuple(self.steps, "AgentRun.steps")
        for index, step in enumerate(steps):
            if not isinstance(step, Step):
                fail(f"AgentRun.steps[{index}]", "must be a Step")
        seqs = [step.seq for step in steps]
        if any(later <= earlier for earlier, later in zip(seqs, seqs[1:], strict=False)):
            fail("AgentRun.steps", f"must be sorted by strictly increasing seq, got {seqs}")
        set_field(self, "steps", steps)
        set_field(self, "final_answer", freeze_json(self.final_answer, "AgentRun.final_answer"))
        check_opt_str(self.final_text, "AgentRun.final_text")
        outputs = as_tuple(self.turn_outputs, "AgentRun.turn_outputs")
        set_field(
            self,
            "turn_outputs",
            tuple(freeze_json(o, f"AgentRun.turn_outputs[{i}]") for i, o in enumerate(outputs)),
        )
        messages = as_tuple(self.messages, "AgentRun.messages")
        for index, message in enumerate(messages):
            if not isinstance(message, Message):
                fail(f"AgentRun.messages[{index}]", "must be a Message")
        set_field(self, "messages", messages)
        tools = as_tuple(self.available_tools, "AgentRun.available_tools")
        for index, tool in enumerate(tools):
            if not isinstance(tool, ToolInfo):
                fail(f"AgentRun.available_tools[{index}]", "must be a ToolInfo")
        set_field(self, "available_tools", tools)
        check_bool(self.model_steps_captured, "AgentRun.model_steps_captured")
        check_str(self.started_at, "AgentRun.started_at", non_empty=False)
        set_field(self, "state", freeze_mapping(self.state, "AgentRun.state"))
        if not isinstance(self.usage, Usage):
            fail("AgentRun.usage", "must be a Usage")
        check_opt_str(self.error, "AgentRun.error")
        set_field(self, "extra", freeze_mapping(self.extra, "AgentRun.extra"))

    def tool_steps(self) -> tuple[Step, ...]:
        """Tool and sub-agent steps (model steps excluded), in order."""
        return tuple(step for step in self.steps if step.kind in _CALL_STEP_KINDS)

    def tool_names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.tool_steps())


@dataclass(slots=True, frozen=True)
class RunResult:
    """The outcome of ``Orchestrator.run()``. A versioned document (``SCHEMA_VERSION``)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    status: str
    run: AgentRun
    _: KW_ONLY
    error: ErrorInfo | None = None
    violations: tuple[Violation, ...] = ()
    warnings: tuple[str, ...] = ()
    trace_id: str = ""
    duration_ms: float = 0.0
    setup_ms: float = 0.0
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_choice(self.status, "RunResult.status", RUN_STATUSES)
        if not isinstance(self.run, AgentRun):
            fail("RunResult.run", "must be an AgentRun")
        needs_error = self.status in ("crashed", "setup_failed")
        if needs_error and self.error is None:
            fail("RunResult.error", f"is required when status is {self.status!r}")
        if not needs_error and self.error is not None:
            fail("RunResult.error", f"must be None when status is {self.status!r}")
        if self.error is not None and not isinstance(self.error, ErrorInfo):
            fail("RunResult.error", "must be an ErrorInfo")
        violations = as_tuple(self.violations, "RunResult.violations")
        for index, violation in enumerate(violations):
            if not isinstance(violation, Violation):
                fail(f"RunResult.violations[{index}]", "must be a Violation")
        set_field(self, "violations", violations)
        set_field(self, "warnings", str_tuple(self.warnings, "RunResult.warnings"))
        check_str(self.trace_id, "RunResult.trace_id", non_empty=False)
        set_field(
            self, "duration_ms", check_number(self.duration_ms, "RunResult.duration_ms", minimum=0)
        )
        set_field(self, "setup_ms", check_number(self.setup_ms, "RunResult.setup_ms", minimum=0))
        set_field(self, "extra", freeze_mapping(self.extra, "RunResult.extra"))


@dataclass(slots=True, frozen=True)
class StepEvent:
    """Payload handed to step listeners (not a saved document)."""

    phase: str
    trace_id: str
    seq: int
    call_id: str
    kind: str
    name: str
    turn: int
    _: KW_ONLY
    args: JSONObject = field(default_factory=FrozenDict)
    result: JSONValue = None
    error: str | None = None
    duration_ms: float | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_choice(self.phase, "StepEvent.phase", ("start", "end"))
        check_str(self.trace_id, "StepEvent.trace_id", non_empty=False)
        check_int(self.seq, "StepEvent.seq", minimum=1)
        check_str(self.call_id, "StepEvent.call_id")
        check_choice(self.kind, "StepEvent.kind", STEP_KINDS)
        check_str(self.name, "StepEvent.name")
        check_int(self.turn, "StepEvent.turn", minimum=0)
        set_field(self, "args", freeze_mapping(self.args, "StepEvent.args"))
        set_field(self, "result", freeze_json(self.result, "StepEvent.result"))
        check_opt_str(self.error, "StepEvent.error")
        set_field(
            self,
            "duration_ms",
            check_opt_number(self.duration_ms, "StepEvent.duration_ms", minimum=0),
        )
        set_field(self, "extra", freeze_mapping(self.extra, "StepEvent.extra"))
