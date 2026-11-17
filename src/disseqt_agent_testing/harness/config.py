"""Orchestrator options live in one config object so new options never change a signature."""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

from ..models import FrozenDict, Message
from ..models._frozen import JSONObject, freeze_mapping
from ..models._validate import check_bool, check_int, check_opt_number, fail, set_field

if TYPE_CHECKING:
    from .fake_llm import FakeLLM


@dataclass(slots=True, frozen=True)
class OrchestratorConfig:
    """Everything ``build_test_orchestrator`` accepts besides the factory and the fakes."""

    _: KW_ONLY
    entry: "str | Callable[[Any, str], Any] | None" = None
    llm: "FakeLLM | None" = None
    handlers: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    timeout_s: float | None = 60.0
    extract: Callable[[Any], Any] | None = None
    async_tools: bool = False
    assert_all_called: bool = False
    traceback_limit: int = 20
    max_recorded_steps: int = 10_000
    clock: Callable[[], int] | None = None
    id_factory: Callable[[], str] | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if self.entry is not None and not (isinstance(self.entry, str) or callable(self.entry)):
            fail("OrchestratorConfig.entry", "must be a method name or a callable")
        handlers = dict(self.handlers)
        for name, handler in handlers.items():
            if not isinstance(name, str) or not callable(handler):
                fail("OrchestratorConfig.handlers", "must map fake names to callables")
        set_field(self, "handlers", MappingProxyType(handlers))
        set_field(
            self,
            "timeout_s",
            check_opt_number(
                self.timeout_s, "OrchestratorConfig.timeout_s", minimum=0, exclusive_minimum=True
            ),
        )
        for name in ("extract", "clock", "id_factory"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                fail(f"OrchestratorConfig.{name}", "must be callable")
        check_bool(self.async_tools, "OrchestratorConfig.async_tools")
        check_bool(self.assert_all_called, "OrchestratorConfig.assert_all_called")
        check_int(self.traceback_limit, "OrchestratorConfig.traceback_limit", minimum=1)
        check_int(self.max_recorded_steps, "OrchestratorConfig.max_recorded_steps", minimum=1)
        set_field(self, "extra", freeze_mapping(self.extra, "OrchestratorConfig.extra"))


class OrchestratorOptions(TypedDict, total=False):
    """Keyword overrides for ``build_test_orchestrator`` (same keys as ``OrchestratorConfig``)."""

    entry: "str | Callable[[Any, str], Any] | None"
    llm: "FakeLLM | None"
    handlers: Mapping[str, Callable[..., Any]]
    timeout_s: float | None
    extract: Callable[[Any], Any] | None
    async_tools: bool
    assert_all_called: bool
    traceback_limit: int
    max_recorded_steps: int
    clock: Callable[[], int] | None
    id_factory: Callable[[], str] | None
    extra: JSONObject


def config_field_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(OrchestratorConfig))


def resolve_config(
    config: OrchestratorConfig | None, options: Mapping[str, Any]
) -> OrchestratorConfig:
    """Apply keyword overrides to ``config``; an unknown option is an error, never ignored."""
    valid = sorted(config_field_names())
    for key in options:
        if key not in valid:
            raise TypeError(f"unknown option {key!r}; valid options: {valid}")
    base = config if config is not None else OrchestratorConfig()
    return dataclasses.replace(base, **options)


@dataclass(slots=True, frozen=True)
class RunOptions:
    """Per-run overrides (new per-run parameters are new fields here)."""

    _: KW_ONLY
    timeout_s: float | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        set_field(
            self,
            "timeout_s",
            check_opt_number(
                self.timeout_s, "RunOptions.timeout_s", minimum=0, exclusive_minimum=True
            ),
        )
        set_field(self, "extra", freeze_mapping(self.extra, "RunOptions.extra"))


@dataclass(slots=True, frozen=True)
class AgentInput:
    """The normalized input for one turn, handed to adapters (new context = new fields)."""

    text: str
    _: KW_ONLY
    messages: tuple[Message, ...] = ()
    turn: int = 0
    extra: JSONObject = field(default_factory=FrozenDict)
