"""The adapter protocol (frozen in phase 1a; ``register_adapter`` becomes public in 1b)."""

from typing import Any, Protocol, runtime_checkable

from ..config import AgentInput

ADAPTER_API_VERSION = 1

# Top-level module names of agent frameworks. An agent object from one of them (installed, not the
# user's own package of the same name) cannot be driven by guessing a method: it needs an adapter
# or an explicit ``entry=``.
FRAMEWORK_MODULES: tuple[str, ...] = (
    "langgraph",
    "langchain",
    "langchain_core",
    "agents",
    "pydantic_ai",
    "crewai",
    "google",
    "autogen_agentchat",
    "llama_index",
)


@runtime_checkable
class Adapter(Protocol):
    name: str

    def matches(self, product: Any) -> bool: ...
    def bind_entry(self, product: Any, entry: Any) -> Any: ...  # a BoundEntry
    def shape_input(self, inp: AgentInput) -> Any: ...
    def extract(self, raw: Any) -> tuple[bool, Any]: ...
    def tool_slot(self, param: Any, fakes: Any) -> Any | None: ...
    def wrap_llm(self, client: Any, param: Any) -> Any | None: ...


class BaseAdapter:
    """Safe defaults for every protocol method, so new protocol methods never break adapters."""

    name = "base"

    def matches(self, product: Any) -> bool:
        return False

    def bind_entry(self, product: Any, entry: Any) -> Any:
        raise NotImplementedError

    def shape_input(self, inp: AgentInput) -> Any:
        return inp.text

    def extract(self, raw: Any) -> tuple[bool, Any]:
        return False, None

    def tool_slot(self, param: Any, fakes: Any) -> Any | None:
        return None

    def wrap_llm(self, client: Any, param: Any) -> Any | None:
        return None
