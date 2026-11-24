"""The test harness: run an agent offline with fake tools and a scripted fake LLM.

    orch = build_test_orchestrator(make_agent, {"get_order": spy({"id": "A1"})})
    result = orch.run("refund order A1")      # always a RunResult; never raises for agent failures

See ``plans/02-build-test-orchestrator.md``.
"""

from .adapters.base import ADAPTER_API_VERSION, Adapter, BaseAdapter
from .adapters.plain import BoundEntry, PlainAdapter
from .config import AgentInput, OrchestratorConfig, OrchestratorOptions, RunOptions
from .errors import (
    FakeToolError,
    HarnessConfigError,
    RunAbandonedError,
    ScriptExhaustedError,
    UnexpectedFakeCall,
)
from .fake_llm import FakeLLM, FakeLLMClient, FakeReply
from .fakes import FakeDraft, spy, stub
from .orchestrator import Orchestrator, build_test_orchestrator
from .recorder import current_recorder
from .registry import (
    KindSpec,
    SlotSpec,
    register_adapter,
    register_fake_kind,
    register_slot,
    unregister_adapter,
    unregister_fake_kind,
    unregister_slot,
)

__all__ = [
    "ADAPTER_API_VERSION",
    "Adapter",
    "AgentInput",
    "BaseAdapter",
    "BoundEntry",
    "FakeDraft",
    "FakeLLM",
    "FakeLLMClient",
    "FakeReply",
    "FakeToolError",
    "HarnessConfigError",
    "KindSpec",
    "Orchestrator",
    "OrchestratorConfig",
    "OrchestratorOptions",
    "PlainAdapter",
    "RunAbandonedError",
    "RunOptions",
    "ScriptExhaustedError",
    "SlotSpec",
    "UnexpectedFakeCall",
    "build_test_orchestrator",
    "current_recorder",
    "register_adapter",
    "register_fake_kind",
    "register_slot",
    "spy",
    "stub",
    "unregister_adapter",
    "unregister_fake_kind",
    "unregister_slot",
]
