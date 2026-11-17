"""Errors and exact message texts of the harness (snapshot-tested; see plan 02 section 13)."""

from typing import Any


class HarnessConfigError(ValueError):
    """The test itself is mis-configured (not the agent). Always raised, never returned."""


class FakeToolError(Exception):
    """Raised into the agent when a fake response is scripted with ``error=``."""

    def __init__(self, message: str, *, fake: str | None = None, call_id: str | None = None):
        super().__init__(message)
        self.fake = fake
        self.call_id = call_id


class UnexpectedFakeCall(Exception):
    """A fake was called and no rule matched, with no handler and no default."""

    def __init__(self, fake: str, args_json: str, rules: str):
        message = (
            f"{fake}({args_json}) matched no response and has no default; rules: {rules or 'none'}"
        )
        super().__init__(message)
        self.fake = fake
        self.args_json = args_json
        self.rules = rules


class ScriptExhaustedError(Exception):
    """The scripted ``FakeLLM`` has no reply left for this call."""


class RunAbandonedError(Exception):
    """A fake was called after its run already ended (a timed-out agent still running)."""

    def __init__(self, trace_id: str, status: str, fake: str):
        super().__init__(
            f"run {trace_id[:8]}... already ended ({status}); "
            f"fake '{fake}' was called after the run closed"
        )


class RunAbortedError(BaseException):
    """The harness stopped a runaway run (``max_recorded_steps`` exceeded).

    Derives from ``BaseException`` on purpose: an agent's broad ``except Exception`` cannot
    swallow it and keep calling fakes. Never escapes ``Orchestrator.run``; the result has
    status ``"aborted"`` and a ``steps_cap_exceeded`` violation.
    """

    def __init__(self, trace_id: str, max_steps: int):
        super().__init__(
            f"run {trace_id[:8]}... aborted: more than {max_steps} recorded steps "
            "(max_recorded_steps); the agent is probably stuck in a loop"
        )
        self.max_steps = max_steps


def names(items: Any) -> str:
    """Render a list of names the way the messages show them."""
    return str([str(item) for item in items])


E_UNKNOWN_PARAM = (
    "factory parameter '{param}' has no default and is not injectable. "
    "Factory parameters: {params}. Injectable names: {injectable}."
)
E_POSONLY = (
    "factory parameter '{param}' is positional-only and has no default; the harness passes "
    "keywords only. Factory parameters: {params}. Injectable names: {injectable}."
)
E_UNDELIVERABLE = (
    "fake '{fake}' (kind={kind}) cannot be injected: the factory declares neither '{slot}', "
    "'{fake}' nor **kwargs. Factory parameters: {params}. Injectable names: {injectable}."
)
E_LLM_REQUIRED = (
    "factory parameter 'llm' has no default but no FakeLLM was configured; pass llm=FakeLLM(...) "
    "or give 'llm' a default."
)
E_LLM_UNDELIVERABLE = (
    "a FakeLLM was configured but the factory declares neither 'llm' nor **kwargs. "
    "Factory parameters: {params}."
)
E_SHAPE = (
    "parameter '{param}' is annotated {ann}; this version supports dict[str, Callable] (default) "
    "or list[Callable]. Framework tool types need an adapter (phase 1b) or wrap the callables in "
    "your factory."
)
E_DUP = "duplicate fake name '{name}'"
E_KEY = "fakes key '{key}' does not match FakeSpec.name '{name}'"
E_HANDLER_DEFAULT = "fake '{name}' sets both default and handler; choose one"
E_HANDLER_UNKNOWN = "handler given for '{name}' but no fake has that name"
E_DRAFT = "fake drafts need a name: use a mapping {{name: stub(...)}} or FakeSpec(name=...)"
E_ENTRY = (
    "could not find an entry point on {cls}: tried {tried}. Pass entry='method_name' or "
    "entry=callable(agent, input)."
)
E_ENTRY_ATTR = "entry='{entry}' not found on {cls}"
E_FRAMEWORK = (
    "agent of type {type} looks like a {framework} object and no adapter is registered for it. "
    "Pass entry=... (and extract=...) to drive it explicitly, or use the {framework} adapter "
    "(phase 1b)."
)
E_CLASS = "agent_factory returned a class ({cls}); return an instance or a callable"
E_LLM_SCRIPT = "FakeLLM needs exactly one of script or handler"
E_LLM_SCRIPT_ROLE = (
    "FakeLLM script entry #{index} has role '{role}'; script entries must be assistant messages"
)
E_ASYNC_HANDLER = (
    "fake '{name}' has an async handler but was called from sync code; build the orchestrator "
    "with async_tools=True"
)

W_ZOMBIE = "worker thread still running after timeout; later fake calls raise RunAbandonedError"
W_NO_MODEL_CAPTURE = (
    "model calls not captured: no FakeLLM or LLM instrumentation; max_llm_calls and token "
    "metrics will be skipped"
)
W_NONE_ANSWER = "agent returned None"
W_LOSSY_ANSWER = "final answer is not JSON-serializable; stored lossy repr"
