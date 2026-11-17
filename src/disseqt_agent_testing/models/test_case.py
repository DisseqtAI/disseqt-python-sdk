"""Test case models: what should happen. Pure data, validated on construction.

``AgentTestCase`` is the top-level document; ``Expect`` says what must be true; ``ToolExpect`` and
``ForbiddenCall`` describe tool calls; ``FakeSpec`` / ``FakeResponse`` describe the fake tools and
sub-agents the agent sees. Phase 1B fields (routing, loops, intent, clarification, ...) are added
later as new defaulted fields (see the extensibility contract in plans/01).
"""

import re
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from graphlib import CycleError, TopologicalSorter
from typing import Any, ClassVar, Literal

from ._constants import EFFECTS, FAKE_KINDS, Effect, FakeKind
from ._frozen import (
    FrozenDict,
    JSONObject,
    JSONValue,
    MatcherMap,
    MatcherOrStr,
    freeze_json,
    freeze_mapping,
)
from ._validate import (
    as_tuple,
    check_bool,
    check_choice,
    check_int,
    check_name,
    check_number,
    check_opt_int,
    check_opt_number,
    check_opt_str,
    check_str,
    fail,
    set_field,
    str_tuple,
)
from .matchers import (
    PREDICATE_NAME_PATTERN,
    Matcher,
    Semantic,
    matcher_map,
    reject_undecidable,
    walk_matchers,
)
from .message import Message

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MAX_CASE_ID = 128
MAX_LATENCY_S = 3600.0  # one scripted response never sleeps longer than an hour
CALL_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # ToolExpect.id (usable in a Ref)
TAG_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Tag names that would collide with pytest or Disseqt markers (kept in sync with the runner).
RESERVED_TAGS: tuple[str, ...] = (
    "skip",
    "skipif",
    "xfail",
    "parametrize",
    "usefixtures",
    "filterwarnings",
    "agent",
    "disseqt_repeat",
    "xdist_group",
    "asyncio",
    "timeout",
    "flaky",
)
ORDERS: tuple[str, ...] = ("any", "in_order", "exact", "partial")
EXTRA_CALL_POLICIES: tuple[str, ...] = ("allow", "allow_reads", "forbid")
ARGS_MODES: tuple[str, ...] = ("subset", "exact")
ROUTE_MODES: tuple[str, ...] = ("exact", "in_order", "first", "any", "none")
LOOP_KEYS: tuple[str, ...] = ("name_args", "name")
ACCEPT_LEVELS: tuple[str, ...] = ("correct", "partially_correct")
_PATH = r"(?:\.[A-Za-z0-9_-]+|\[\d+\])"
INTENT_SOURCE_PATTERN = re.compile(
    rf"^(?:route|call:[A-Za-z0-9_*?-]+\.(?:args|result){_PATH}+|answer{_PATH}+)$"
)


class Decision(str, Enum):
    """What the agent did on a turn."""

    CALL = "call"
    ASK = "ask"
    REFUSE = "refuse"
    ANSWER = "answer"


def _is_glob(name: str) -> bool:
    return any(char in name for char in "*?[")


def _decisions(value: object, path: str) -> tuple[Decision, ...]:
    """A single decision or a list of decisions (``None`` means none)."""
    if value is None:
        return ()
    single = isinstance(value, (str, Decision))
    items = (value,) if single else as_tuple(value, path)
    out: list[Decision] = []
    for index, item in enumerate(items):
        try:
            out.append(item if isinstance(item, Decision) else Decision(item))
        except ValueError:
            fail(
                path if single else f"{path}[{index}]",
                f"must be one of {[d.value for d in Decision]}, got {item!r}",
            )
    return tuple(out)


@dataclass(slots=True, frozen=True)
class ToolExpect:
    """One expected tool call."""

    name: str
    args: MatcherMap = field(default_factory=FrozenDict)
    _: KW_ONLY
    args_mode: Literal["subset", "exact"] = "subset"
    count: "int | tuple[int, int | None] | None" = None
    id: str | None = None
    after: tuple[str, ...] = ()
    turn: int | None = None
    defaults: MatcherMap = field(default_factory=FrozenDict)
    ignore: tuple[str, ...] = ()
    same_step_as: str | None = None
    weight: float = 1.0
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_name(self.name, "ToolExpect.name")
        set_field(self, "args", matcher_map(self.args, "ToolExpect.args"))
        check_choice(self.args_mode, "ToolExpect.args_mode", ARGS_MODES)
        set_field(self, "count", _normalize_count(self.count))
        if self.id is not None:
            if not isinstance(self.id, str) or not CALL_LABEL_PATTERN.fullmatch(self.id):
                fail("ToolExpect.id", f"must match {CALL_LABEL_PATTERN.pattern}, got {self.id!r}")
        set_field(self, "after", str_tuple(self.after, "ToolExpect.after"))
        check_opt_int(self.turn, "ToolExpect.turn", minimum=0)
        set_field(self, "defaults", matcher_map(self.defaults, "ToolExpect.defaults"))
        set_field(self, "ignore", str_tuple(self.ignore, "ToolExpect.ignore"))
        if self.same_step_as is not None and (
            not isinstance(self.same_step_as, str)
            or not CALL_LABEL_PATTERN.fullmatch(self.same_step_as)
        ):
            fail("ToolExpect.same_step_as", f"must be a ToolExpect.id, got {self.same_step_as!r}")
        set_field(
            self,
            "weight",
            check_number(self.weight, "ToolExpect.weight", minimum=0, exclusive_minimum=True),
        )
        set_field(self, "extra", freeze_mapping(self.extra, "ToolExpect.extra"))


def _normalize_count(count: object) -> "tuple[int, int | None] | None":
    if count is None:
        return None
    if isinstance(count, bool):
        fail("ToolExpect.count", "must be an int or a (min, max) pair, not a bool")
    if isinstance(count, int):
        if count == 0:
            fail(
                "ToolExpect.count", "must be >= 1 (use Expect.forbidden to say a tool must not run)"
            )
        if count < 0:
            fail("ToolExpect.count", f"must be >= 1, got {count}")
        return (count, count)
    if isinstance(count, (list, tuple)) and len(count) == 2:
        low = check_int(count[0], "ToolExpect.count[0]", minimum=0)
        high = check_opt_int(count[1], "ToolExpect.count[1]")
        if high is not None and high < max(low, 1):
            fail("ToolExpect.count", f"max must be None or >= max(min, 1), got ({low}, {high})")
        return (low, high)
    fail("ToolExpect.count", f"must be an int or a (min, max) pair, got {count!r}")


@dataclass(slots=True, frozen=True)
class ForbiddenCall:
    """A hard veto: one matching call fails the case, whatever the scores."""

    name: str
    args: "MatcherMap | None" = None
    _: KW_ONLY
    unless_after: str | None = None
    reason: str = ""
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_name(self.name, "ForbiddenCall.name")
        if self.args is not None:
            matchers = matcher_map(self.args, "ForbiddenCall.args")
            reject_undecidable(matchers, "ForbiddenCall.args", "ForbiddenCall.args")
            set_field(self, "args", matchers)
        check_opt_str(self.unless_after, "ForbiddenCall.unless_after")
        check_str(self.reason, "ForbiddenCall.reason", non_empty=False)
        set_field(self, "extra", freeze_mapping(self.extra, "ForbiddenCall.extra"))


@dataclass(slots=True, frozen=True)
class FakeResponse:
    """One scripted reply of a fake tool or sub-agent."""

    _: KW_ONLY
    when: MatcherMap = field(default_factory=FrozenDict)
    returns: JSONValue = None
    error: str | None = None
    times: int | None = None
    latency_s: float = 0.0
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        matchers = matcher_map(self.when, "FakeResponse.when")
        reject_undecidable(matchers, "FakeResponse.when", "FakeResponse.when")
        set_field(self, "when", matchers)
        set_field(self, "returns", freeze_json(self.returns, "FakeResponse.returns"))
        check_opt_str(self.error, "FakeResponse.error")
        if self.error is not None and self.returns is not None:
            fail("FakeResponse", "error and returns are mutually exclusive")
        check_opt_int(self.times, "FakeResponse.times", minimum=1)
        set_field(
            self,
            "latency_s",
            check_number(
                self.latency_s, "FakeResponse.latency_s", minimum=0, maximum=MAX_LATENCY_S
            ),
        )
        set_field(self, "extra", freeze_mapping(self.extra, "FakeResponse.extra"))


def _check_schema_shape(schema: Mapping[str, Any]) -> None:
    """``properties`` is a name -> object mapping; ``required`` lists names found there."""
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        fail("FakeSpec.schema.properties", f"must be an object, got {type(properties).__name__}")
    for name, prop in properties.items():
        if not isinstance(prop, Mapping):
            fail(
                f"FakeSpec.schema.properties.{name}",
                f"must be an object, got {type(prop).__name__}",
            )
    required = schema.get("required", ())
    if not isinstance(required, (list, tuple)):
        fail("FakeSpec.schema.required", f"must be a list of names, got {type(required).__name__}")
    for index, name in enumerate(required):
        if not isinstance(name, str):
            fail(f"FakeSpec.schema.required[{index}]", f"must be a string, got {name!r}")
        if name not in properties:
            fail(f"FakeSpec.schema.required[{index}]", f"{name!r} is not in properties")


@dataclass(slots=True, frozen=True)
class FakeSpec:
    """One fake tool or sub-agent that stands in for the real one during the test."""

    name: str
    _: KW_ONLY
    kind: FakeKind = "tool"
    description: str = ""
    effect: Effect = "write"
    schema: "JSONObject | None" = None
    responses: tuple[FakeResponse, ...] = ()
    default: FakeResponse | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_name(self.name, "FakeSpec.name")
        check_choice(self.kind, "FakeSpec.kind", FAKE_KINDS)
        check_str(self.description, "FakeSpec.description", non_empty=False)
        check_choice(self.effect, "FakeSpec.effect", EFFECTS)
        if self.schema is not None:
            schema = freeze_mapping(self.schema, "FakeSpec.schema")
            if "type" in schema and schema["type"] != "object":
                fail("FakeSpec.schema", f"type must be 'object', got {schema['type']!r}")
            _check_schema_shape(schema)
            set_field(self, "schema", schema)
        responses = as_tuple(self.responses, "FakeSpec.responses")
        for index, response in enumerate(responses):
            if not isinstance(response, FakeResponse):
                fail(f"FakeSpec.responses[{index}]", "must be a FakeResponse")
        set_field(self, "responses", responses)
        if self.default is not None and not isinstance(self.default, FakeResponse):
            fail("FakeSpec.default", "must be a FakeResponse")
        set_field(self, "extra", freeze_mapping(self.extra, "FakeSpec.extra"))


@dataclass(slots=True, frozen=True)
class RouteExpect:
    """Expected handoff / sub-agent path. A bare tuple of names means ``in_order``."""

    _: KW_ONLY
    path: tuple[str, ...] = ()
    mode: Literal["exact", "in_order", "first", "any", "none"] = "in_order"
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        set_field(self, "path", str_tuple(self.path, "RouteExpect.path"))
        check_choice(self.mode, "RouteExpect.mode", ROUTE_MODES)
        if self.mode == "none" and self.path:
            fail("RouteExpect.path", "must be empty when mode is 'none'")
        if self.mode in ("in_order", "first", "any") and not self.path:
            fail("RouteExpect.path", f"must not be empty when mode is {self.mode!r}")
        set_field(self, "extra", freeze_mapping(self.extra, "RouteExpect.extra"))


@dataclass(slots=True, frozen=True)
class LoopExpect:
    """Loop and cycle limits for the recorded calls."""

    _: KW_ONLY
    max_identical_calls: int | None = 3
    max_consecutive_identical: int | None = 2
    detect_cycles: bool = True
    max_period: int = 4
    min_repeats: int = 2
    key: Literal["name_args", "name"] = "name_args"
    max_retries_after_error: int = 2
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_opt_int(self.max_identical_calls, "LoopExpect.max_identical_calls", minimum=1)
        check_opt_int(
            self.max_consecutive_identical, "LoopExpect.max_consecutive_identical", minimum=1
        )
        check_bool(self.detect_cycles, "LoopExpect.detect_cycles")
        check_int(self.max_period, "LoopExpect.max_period", minimum=2)
        check_int(self.min_repeats, "LoopExpect.min_repeats", minimum=2)
        check_choice(self.key, "LoopExpect.key", LOOP_KEYS)
        check_int(self.max_retries_after_error, "LoopExpect.max_retries_after_error", minimum=0)
        set_field(self, "extra", freeze_mapping(self.extra, "LoopExpect.extra"))


def _predicate_names(value: object, path: str) -> tuple[str, ...]:
    names = str_tuple(value, path)
    for name in names:
        if not PREDICATE_NAME_PATTERN.fullmatch(name):
            fail(
                path,
                f"{name!r} must be a registered predicate name matching {PREDICATE_NAME_PATTERN.pattern}",
            )
    return names


@dataclass(slots=True, frozen=True)
class PlanExpect:
    """Plan constraints (registered predicate names) and convergence."""

    _: KW_ONLY
    optimal_tool_calls: int | None = None
    constraints: tuple[str, ...] = ()
    judge_quality: bool = False
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_opt_int(self.optimal_tool_calls, "PlanExpect.optimal_tool_calls", minimum=1)
        set_field(self, "constraints", _predicate_names(self.constraints, "PlanExpect.constraints"))
        check_bool(self.judge_quality, "PlanExpect.judge_quality")
        set_field(self, "extra", freeze_mapping(self.extra, "PlanExpect.extra"))


@dataclass(slots=True, frozen=True)
class ClarifyExpect:
    """What a clarifying question must do."""

    _: KW_ONLY
    must_mention_any: tuple[str, ...] = ()
    must_mention_all: tuple[str, ...] = ()
    max_questions: int | None = None
    allow_read_calls_before: bool = True
    judge_relevance: bool = False
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        set_field(
            self,
            "must_mention_any",
            str_tuple(self.must_mention_any, "ClarifyExpect.must_mention_any"),
        )
        set_field(
            self,
            "must_mention_all",
            str_tuple(self.must_mention_all, "ClarifyExpect.must_mention_all"),
        )
        check_opt_int(self.max_questions, "ClarifyExpect.max_questions", minimum=1)
        check_bool(self.allow_read_calls_before, "ClarifyExpect.allow_read_calls_before")
        check_bool(self.judge_relevance, "ClarifyExpect.judge_relevance")
        set_field(self, "extra", freeze_mapping(self.extra, "ClarifyExpect.extra"))


@dataclass(slots=True, frozen=True)
class RefuseExpect:
    """What a refusal must do."""

    _: KW_ONLY
    must_mention_any: tuple[str, ...] = ()
    offer_alternative: bool | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        set_field(
            self,
            "must_mention_any",
            str_tuple(self.must_mention_any, "RefuseExpect.must_mention_any"),
        )
        if self.offer_alternative is not None:
            check_bool(self.offer_alternative, "RefuseExpect.offer_alternative")
        set_field(self, "extra", freeze_mapping(self.extra, "RefuseExpect.extra"))


@dataclass(slots=True, frozen=True)
class ScopeExpect:
    """Whether the request is in scope (so an agent refusing it is over- or under-refusing)."""

    in_scope: bool
    _: KW_ONLY
    topics: tuple[str, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_bool(self.in_scope, "ScopeExpect.in_scope")
        set_field(self, "topics", str_tuple(self.topics, "ScopeExpect.topics"))
        set_field(self, "extra", freeze_mapping(self.extra, "ScopeExpect.extra"))


@dataclass(slots=True, frozen=True)
class AnswerQualityExpect:
    """A judged comparison of the final answer with a reference answer."""

    reference: str
    _: KW_ONLY
    accept: Literal["correct", "partially_correct"] = "correct"
    criteria: tuple[str, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        check_str(self.reference, "AnswerQualityExpect.reference")
        check_choice(self.accept, "AnswerQualityExpect.accept", ACCEPT_LEVELS)
        set_field(self, "criteria", str_tuple(self.criteria, "AnswerQualityExpect.criteria"))
        set_field(self, "extra", freeze_mapping(self.extra, "AnswerQualityExpect.extra"))


@dataclass(slots=True, frozen=True)
class GroundednessExpect:
    """A judged check that the final answer's claims are supported by tool results."""

    _: KW_ONLY
    sources: tuple[str, ...] = ("*",)
    include_user_statements: bool = True
    allow_unsupported: bool = False
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        sources = str_tuple(self.sources, "GroundednessExpect.sources")
        if not sources:
            fail("GroundednessExpect.sources", "must not be empty")
        set_field(self, "sources", sources)
        check_bool(self.include_user_statements, "GroundednessExpect.include_user_statements")
        check_bool(self.allow_unsupported, "GroundednessExpect.allow_unsupported")
        set_field(self, "extra", freeze_mapping(self.extra, "GroundednessExpect.extra"))


@dataclass(slots=True, frozen=True)
class Expect:
    """What must be true about the agent's behavior. Everything left unset is not checked."""

    _: KW_ONLY
    decision: "Decision | tuple[Decision, ...]" = ()
    decision_by_turn: Mapping[Any, Any] = field(default_factory=FrozenDict)
    tools: tuple[ToolExpect, ...] = ()
    order: Literal["any", "in_order", "exact", "partial"] = "any"
    extra_calls: Literal["allow", "allow_reads", "forbid"] = "allow_reads"
    allowed_tools: tuple[str, ...] | None = None
    forbidden: tuple[ForbiddenCall, ...] = ()
    max_steps: int | None = None
    max_tool_calls: "int | Mapping[str, int] | None" = None
    max_llm_calls: int | None = None
    max_tokens: int | None = None
    max_latency_s: float | None = None
    answer_contains: tuple[MatcherOrStr, ...] = ()
    answer_excludes: tuple[MatcherOrStr, ...] = ()
    min_scores: Mapping[str, float] = field(default_factory=FrozenDict)
    routes: "RouteExpect | tuple[str, ...] | None" = None
    loops: LoopExpect | None = None
    intent: "str | tuple[str, ...]" = ()
    intent_labels: tuple[str, ...] = ()
    intent_source: str | None = None
    clarify: ClarifyExpect | None = None
    refuse: RefuseExpect | None = None
    scope: ScopeExpect | None = None
    plan: PlanExpect | None = None
    answer_quality: AnswerQualityExpect | None = None
    grounded: "GroundednessExpect | bool | None" = None
    extra: JSONObject = field(default_factory=FrozenDict)

    @property
    def intents(self) -> tuple[str, ...]:
        """Acceptable intent labels, always a tuple."""
        intent = self.intent
        return (intent,) if isinstance(intent, str) else intent

    @property
    def decisions(self) -> tuple[Decision, ...]:
        """Acceptable decisions for turn 0, always a tuple."""
        decision = self.decision
        return decision if isinstance(decision, tuple) else (decision,)

    @property
    def turn_decisions(self) -> "dict[int, tuple[Decision, ...]]":
        """Acceptable decisions per turn (turn 0 from ``decision``, the rest from the map)."""
        out: dict[int, tuple[Decision, ...]] = {}
        if self.decisions:
            out[0] = self.decisions
        for key, value in self.decision_by_turn.items():
            out[int(key)] = value
        return out

    def __post_init__(self) -> None:
        set_field(self, "decision", _decisions(self.decision, "Expect.decision"))
        set_field(self, "decision_by_turn", _normalize_decision_by_turn(self.decision_by_turn))
        tools = as_tuple(self.tools, "Expect.tools")
        for index, tool in enumerate(tools):
            if not isinstance(tool, ToolExpect):
                fail(f"Expect.tools[{index}]", "must be a ToolExpect")
        set_field(self, "tools", tools)
        check_choice(self.order, "Expect.order", ORDERS)
        check_choice(self.extra_calls, "Expect.extra_calls", EXTRA_CALL_POLICIES)
        if self.allowed_tools is not None:
            set_field(self, "allowed_tools", str_tuple(self.allowed_tools, "Expect.allowed_tools"))
        forbidden = as_tuple(self.forbidden, "Expect.forbidden")
        for index, veto in enumerate(forbidden):
            if not isinstance(veto, ForbiddenCall):
                fail(f"Expect.forbidden[{index}]", "must be a ForbiddenCall")
        set_field(self, "forbidden", forbidden)
        check_opt_int(self.max_steps, "Expect.max_steps", minimum=1)
        set_field(self, "max_tool_calls", _normalize_max_tool_calls(self.max_tool_calls))
        check_opt_int(self.max_llm_calls, "Expect.max_llm_calls", minimum=1)
        check_opt_int(self.max_tokens, "Expect.max_tokens", minimum=1)
        set_field(
            self,
            "max_latency_s",
            check_opt_number(
                self.max_latency_s, "Expect.max_latency_s", minimum=0, exclusive_minimum=True
            ),
        )
        set_field(
            self, "answer_contains", _text_or_matchers(self.answer_contains, "answer_contains")
        )
        set_field(
            self, "answer_excludes", _text_or_matchers(self.answer_excludes, "answer_excludes")
        )
        set_field(self, "min_scores", _normalize_min_scores(self.min_scores))
        self._normalize_blocks()
        set_field(self, "extra", freeze_mapping(self.extra, "Expect.extra"))
        self._validate_tool_graph()

    def _normalize_blocks(self) -> None:
        routes = self.routes
        if isinstance(routes, (list, tuple)):
            routes = RouteExpect(path=tuple(routes))
        _check_block("routes", routes, RouteExpect)
        set_field(self, "routes", routes)
        for name, kind in (
            ("loops", LoopExpect),
            ("clarify", ClarifyExpect),
            ("refuse", RefuseExpect),
            ("scope", ScopeExpect),
            ("plan", PlanExpect),
            ("answer_quality", AnswerQualityExpect),
        ):
            _check_block(name, getattr(self, name), kind)
        grounded = self.grounded
        if grounded is True:
            grounded = GroundednessExpect()
        elif grounded is False:
            grounded = None
        _check_block("grounded", grounded, GroundednessExpect)
        set_field(self, "grounded", grounded)
        intents = (self.intent,) if isinstance(self.intent, str) else self.intent
        intents = str_tuple(intents, "Expect.intent")
        set_field(self, "intent", intents)
        labels = str_tuple(self.intent_labels, "Expect.intent_labels")
        set_field(self, "intent_labels", labels)
        if self.intent_source is not None and (
            not isinstance(self.intent_source, str)
            or not INTENT_SOURCE_PATTERN.fullmatch(self.intent_source)
        ):
            fail(
                "Expect.intent_source",
                f"{self.intent_source!r} must be 'route', 'call:<tool>.args.<path>', "
                "'call:<tool>.result.<path>' or 'answer.<path>'",
            )
        if intents and not labels and self.intent_source is None:
            fail("Expect.intent", "needs intent_labels or intent_source")
        if labels:
            missing = [i for i in intents if i not in labels]
            if missing:
                fail("Expect.intent", f"{missing[0]!r} is not in intent_labels")

    def _validate_tool_graph(self) -> None:
        ids = [tool.id for tool in self.tools if tool.id is not None]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            fail("Expect.tools", f"ToolExpect.id values must be unique, repeated: {duplicates}")
        known = set(ids)
        graph: dict[str, set[str]] = {}
        for index, tool in enumerate(self.tools):
            for target in tool.after:
                if target not in known:
                    fail(
                        f"Expect.tools[{index}].after",
                        f"after {target!r} does not match any ToolExpect.id",
                    )
            if tool.same_step_as is not None:
                if tool.same_step_as not in known:
                    fail(
                        f"Expect.tools[{index}].same_step_as",
                        f"{tool.same_step_as!r} does not match any ToolExpect.id",
                    )
                if tool.same_step_as == tool.id:
                    fail(f"Expect.tools[{index}].same_step_as", "cannot refer to itself")
            if tool.id is not None:
                graph[tool.id] = set(tool.after)
        try:
            TopologicalSorter(graph).prepare()
        except CycleError as exc:
            cycle = " -> ".join(str(node) for node in exc.args[1])
            fail("Expect.tools", f"cycle in after: {cycle}")
        if any(tool.after for tool in self.tools) and self.order != "partial":
            fail("Expect.order", "must be 'partial' when any ToolExpect has 'after'")
        if Decision.CALL in self.decisions and not self.tools and not self.allowed_tools:
            fail("Expect.decision", "'call' needs tools or allowed_tools")
        for index, veto in enumerate(self.forbidden):
            if veto.unless_after is not None and veto.unless_after not in known:
                fail(
                    f"Expect.forbidden[{index}].unless_after",
                    f"{veto.unless_after!r} does not match any ToolExpect.id",
                )


def _check_block(name: str, value: object, kind: type) -> None:
    if value is not None and not isinstance(value, kind):
        fail(f"Expect.{name}", f"must be a {kind.__name__} or None, got {type(value).__name__}")


_TURN_KEY = re.compile(r"[0-9]{1,9}")


def _normalize_decision_by_turn(value: object) -> "FrozenDict[str, tuple[Decision, ...]]":
    if value is None:
        return FrozenDict()
    if not isinstance(value, Mapping):
        fail("Expect.decision_by_turn", f"must be an object, got {type(value).__name__}")
    out: dict[str, tuple[Decision, ...]] = {}
    for key, item in value.items():
        if isinstance(key, bool) or not (
            isinstance(key, int) or (isinstance(key, str) and _TURN_KEY.fullmatch(key))
        ):
            fail("Expect.decision_by_turn", f"keys must be turn numbers, got {key!r}")
        turn = int(key)
        if str(turn) in out:
            fail("Expect.decision_by_turn", f"turn {turn} is given twice (keys {key!r} collide)")
        if turn == 0:
            fail("Expect.decision_by_turn[0]", "turn 0 is set by 'decision'")
        decisions = _decisions(item, f"Expect.decision_by_turn[{turn}]")
        if not decisions:
            fail(f"Expect.decision_by_turn[{turn}]", "must list at least one decision")
        out[str(turn)] = decisions
    return FrozenDict(dict(sorted(out.items(), key=lambda kv: int(kv[0]))))


def _normalize_max_tool_calls(value: object) -> "int | FrozenDict[str, int] | None":
    if value is None:
        return None
    if isinstance(value, Mapping):
        out: dict[str, int] = {}
        for key, cap in value.items():
            check_str(key, "Expect.max_tool_calls")
            out[key] = check_int(cap, f"Expect.max_tool_calls[{key!r}]", minimum=1)
        return FrozenDict(out)
    return check_int(value, "Expect.max_tool_calls", minimum=1)


def _text_or_matchers(value: object, name: str) -> tuple[Any, ...]:
    items = as_tuple(value, f"Expect.{name}")
    for index, item in enumerate(items):
        if isinstance(item, str):
            check_str(item, f"Expect.{name}[{index}]")
        elif not isinstance(item, Matcher):
            fail(f"Expect.{name}[{index}]", "must be a string or a matcher")
    return items


def _normalize_min_scores(value: object) -> "FrozenDict[str, float]":
    if value is None:
        return FrozenDict()
    if not isinstance(value, Mapping):
        fail("Expect.min_scores", f"must be an object, got {type(value).__name__}")
    return FrozenDict(
        {
            check_str(key, "Expect.min_scores"): check_number(
                score, f"Expect.min_scores[{key!r}]", minimum=0, maximum=1
            )
            for key, score in value.items()
        }
    )


@dataclass(slots=True, frozen=True)
class AgentTestCase:
    """One complete test. A versioned document (``SCHEMA_VERSION``)."""

    SCHEMA_VERSION: ClassVar[int] = 1

    id: str
    input: "str | tuple[Message, ...]"
    _: KW_ONLY
    follow_ups: tuple[str, ...] = ()
    fakes: tuple[FakeSpec, ...] = ()
    expect: Expect = field(default_factory=Expect)
    tags: tuple[str, ...] = ()
    repeats: int = 1
    pass_k: int | None = None
    pass_k_threshold: float = 1.0
    use_judge: tuple[str, ...] = ()
    extra: JSONObject = field(default_factory=FrozenDict)

    @property
    def messages(self) -> tuple[Message, ...]:
        """The input as a tuple of messages (a string input is stored as one user message)."""
        assert isinstance(self.input, tuple)
        return self.input

    def __post_init__(self) -> None:
        check_str(self.id, "AgentTestCase.id")
        if not ID_PATTERN.fullmatch(self.id) or len(self.id) > MAX_CASE_ID:
            fail("AgentTestCase.id", f"must match {ID_PATTERN.pattern} (max {MAX_CASE_ID} chars)")
        set_field(self, "input", _normalize_input(self.input))
        set_field(self, "follow_ups", str_tuple(self.follow_ups, "AgentTestCase.follow_ups"))
        fakes = as_tuple(self.fakes, "AgentTestCase.fakes")
        for index, fake in enumerate(fakes):
            if not isinstance(fake, FakeSpec):
                fail(f"AgentTestCase.fakes[{index}]", "must be a FakeSpec")
        set_field(self, "fakes", fakes)
        if not isinstance(self.expect, Expect):
            fail("AgentTestCase.expect", "must be an Expect")
        tags = str_tuple(self.tags, "AgentTestCase.tags")
        for tag in tags:
            if not TAG_PATTERN.fullmatch(tag):
                fail("AgentTestCase.tags", f"{tag!r} must match {TAG_PATTERN.pattern}")
            if tag in RESERVED_TAGS:
                fail(
                    "AgentTestCase.tags",
                    f"{tag!r} is a reserved name; reserved: {list(RESERVED_TAGS)}",
                )
        set_field(self, "tags", tags)
        check_int(self.repeats, "AgentTestCase.repeats", minimum=1)
        check_opt_int(self.pass_k, "AgentTestCase.pass_k", minimum=1)
        if self.pass_k is not None and self.pass_k > self.repeats:
            fail(
                "AgentTestCase.pass_k",
                f"pass_k ({self.pass_k}) cannot exceed repeats ({self.repeats})",
            )
        set_field(
            self,
            "pass_k_threshold",
            check_number(
                self.pass_k_threshold,
                "AgentTestCase.pass_k_threshold",
                minimum=0,
                exclusive_minimum=True,
                maximum=1,
            ),
        )
        set_field(self, "use_judge", str_tuple(self.use_judge, "AgentTestCase.use_judge"))
        set_field(self, "extra", freeze_mapping(self.extra, "AgentTestCase.extra"))
        self._validate_fakes()
        self._validate_judge_consistency()

    def _validate_judge_consistency(self) -> None:
        """A case that needs a judge must say so (``use_judge``), so it fails at load, not late."""
        if self.use_judge:
            return
        expect = self.expect
        needs: list[str] = []
        if any(isinstance(m, Semantic) for m in walk_matchers(expect)):
            needs.append("a Semantic matcher")
        if expect.clarify is not None and expect.clarify.judge_relevance:
            needs.append("clarify.judge_relevance")
        if expect.refuse is not None and expect.refuse.offer_alternative is not None:
            needs.append("refuse.offer_alternative")
        if expect.plan is not None and expect.plan.judge_quality:
            needs.append("plan.judge_quality")
        if expect.answer_quality is not None:
            needs.append("answer_quality")
        if expect.grounded is not None:
            needs.append("grounded")
        if expect.intent and expect.intent_source is None:
            needs.append("intent without intent_source")
        if needs:
            fail("AgentTestCase.use_judge", f"must be set because the case uses {', '.join(needs)}")

    def _validate_fakes(self) -> None:
        names = [fake.name for fake in self.fakes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            fail("AgentTestCase.fakes", f"duplicate fake name {duplicates[0]!r}")
        if not self.fakes:
            return  # fakes may be supplied to build_test_orchestrator instead
        known = sorted(names)
        wanted = [(f"Expect.tools[{i}]", t.name) for i, t in enumerate(self.expect.tools)]
        wanted += [(f"Expect.forbidden[{i}]", v.name) for i, v in enumerate(self.expect.forbidden)]
        for where, name in wanted:
            if _is_glob(name):
                if not any(fnmatchcase(fake, name) for fake in names):
                    fail(where, f"glob {name!r} matches no fake; known: {known}")
            elif name not in names:
                fail(where, f"expected tool {name!r} has no fake; known: {known}")
        effects = {fake.name: fake.effect for fake in self.fakes}
        # A turn blocks a write only when EVERY acceptable decision there is a non-call; a turn
        # without a decision is unconstrained. A tool with turn=None needs a block on all turns.
        decisions_by_turn = self.expect.turn_decisions
        blocked_turns = {
            turn
            for turn, decisions in decisions_by_turn.items()
            if all(d is not Decision.CALL for d in decisions)
        }
        all_turns = set(range(len(self.follow_ups) + 1))
        for tool in self.expect.tools:
            if tool.turn is None:
                conflict = all_turns <= blocked_turns
                turn = 0
            else:
                conflict = tool.turn in blocked_turns
                turn = tool.turn
            if not conflict:
                continue
            blocked = decisions_by_turn[turn][0]
            for fake_name, effect in effects.items():
                if effect == "write" and fnmatchcase(fake_name, tool.name):
                    fail(
                        "Expect.decision",
                        f"decision {blocked.value!r} on turn {turn} cannot expect write "
                        f"tool {fake_name!r} (set ToolExpect.turn to a later turn)",
                    )


def _normalize_input(value: object) -> tuple[Message, ...]:
    if isinstance(value, str):
        return (Message("user", value),)
    items = as_tuple(value, "AgentTestCase.input")
    if not items:
        fail("AgentTestCase.input", "must not be empty")
    for index, item in enumerate(items):
        if not isinstance(item, Message):
            fail(f"AgentTestCase.input[{index}]", "must be a Message")
    if items[-1].role != "user":
        fail("AgentTestCase.input", "last input message must be from the user")
    return items
