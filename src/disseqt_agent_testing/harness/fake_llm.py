"""A neutral, scripted fake LLM: the only source of MODEL steps in phase 1a.

``FakeLLM`` is the (frozen) script; ``FakeLLMClient`` is the per-run object the agent receives
as ``llm``, bound to that run's recorder. Framework translations (a LangChain chat model, an
OpenAI ``Model``, ...) are phase 1b; they wrap the same script so one script works everywhere.
"""

import inspect
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

from ..models import (
    FrozenDict,
    Message,
    Usage,
    from_dict,
    to_dict,
)
from ..models._frozen import JSONObject, freeze_mapping
from .errors import (
    E_LLM_SCRIPT,
    E_LLM_SCRIPT_ROLE,
    HarnessConfigError,
    ScriptExhaustedError,
)
from .recorder import Recorder


@dataclass(slots=True, frozen=True)
class FakeReply:
    """One scripted reply, optionally with token usage (read by the ``max_tokens`` budget)."""

    message: Message
    _: KW_ONLY
    usage: Usage | None = None
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message) or self.message.role != "assistant":
            raise HarnessConfigError("FakeReply.message must be an assistant Message")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise HarnessConfigError("FakeReply.usage must be a Usage")
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "FakeReply.extra"))


def _as_reply(entry: Any) -> FakeReply:
    if isinstance(entry, FakeReply):
        return entry
    if isinstance(entry, str):
        return FakeReply(Message("assistant", entry))
    if isinstance(entry, Message):
        return FakeReply(entry)
    raise HarnessConfigError(
        f"FakeLLM replies must be str, Message or FakeReply, got {type(entry).__name__}"
    )


@dataclass(slots=True, frozen=True)
class FakeLLM:
    """The script of replies (one per model call) or a handler function, never both."""

    _: KW_ONLY
    script: "tuple[Message | str | FakeReply, ...]" = ()
    handler: Callable[..., Any] | None = None
    on_exhaust: Literal["raise", "repeat_last"] = "raise"
    name: str = "fake-llm"
    extra: JSONObject = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        script = tuple(self.script)
        if bool(script) == (self.handler is not None):
            raise HarnessConfigError(E_LLM_SCRIPT)
        if self.handler is not None and not callable(self.handler):
            raise HarnessConfigError("FakeLLM.handler must be callable")
        replies: list[FakeReply] = []
        for index, entry in enumerate(script, start=1):
            message = entry.message if isinstance(entry, FakeReply) else entry
            if isinstance(message, Message) and message.role != "assistant":
                raise HarnessConfigError(E_LLM_SCRIPT_ROLE.format(index=index, role=message.role))
            replies.append(_as_reply(entry))
        object.__setattr__(self, "script", tuple(replies))
        if self.on_exhaust not in ("raise", "repeat_last"):
            raise HarnessConfigError("FakeLLM.on_exhaust must be 'raise' or 'repeat_last'")
        if not isinstance(self.name, str) or not self.name.strip():
            raise HarnessConfigError("FakeLLM.name must be a non-empty string")
        object.__setattr__(self, "extra", freeze_mapping(self.extra, "FakeLLM.extra"))


def _messages(items: Any) -> tuple[Message, ...]:
    if isinstance(items, (str, Message, Mapping)) or not isinstance(items, Sequence):
        raise HarnessConfigError("llm.complete() needs a sequence of messages")
    out: list[Message] = []
    for item in items:
        if isinstance(item, Message):
            out.append(item)
        elif isinstance(item, str):
            out.append(Message("user", item))
        elif isinstance(item, Mapping):
            out.append(from_dict(Message, item))
        else:
            raise HarnessConfigError(f"unsupported message type {type(item).__name__}")
    return tuple(out)


def _tool_names(tools: Any) -> tuple[str, ...]:
    if tools is None:
        return ()
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, str):
            names.append(tool)
        elif isinstance(tool, Mapping) and isinstance(tool.get("name"), str):
            names.append(tool["name"])
        elif isinstance(getattr(tool, "name", None), str):
            names.append(tool.name)
        else:
            raise HarnessConfigError(f"unsupported tool description {type(tool).__name__}")
    return tuple(names)


class FakeLLMClient:
    """Fresh per run. ``complete`` / ``acomplete`` return the next scripted assistant message."""

    def __init__(self, llm: FakeLLM, recorder: Recorder):
        self._llm = llm
        self._replies = tuple(_as_reply(entry) for entry in llm.script)
        self._rec = recorder
        self._lock = threading.RLock()
        self._calls = 0
        self._requests: list[tuple[Message, ...]] = []
        self._last_request: tuple[Message, ...] = ()
        self._last_reply: Message | None = None

    @property
    def requests(self) -> tuple[tuple[Message, ...], ...]:
        with self._lock:
            return tuple(self._requests)

    @property
    def last_exchange(self) -> tuple[Message, ...]:
        """The last request followed by its reply (empty before the first call)."""
        with self._lock:
            return (*self._last_request, self._last_reply) if self._last_reply else ()

    def _open(
        self, messages: Any, tools: Any
    ) -> tuple[tuple[Message, ...], tuple[str, ...], Any, int]:
        msgs = _messages(messages)
        names = _tool_names(tools)
        args = {"messages": [to_dict(m) for m in msgs], "tools": list(names)}
        # One critical section for the call number (which picks the scripted reply) and the step
        # number, so that among concurrent calls the earlier step always gets the earlier reply.
        with self._lock:
            pending = self._rec.start("model", self._llm.name, args, None)
            self._calls += 1
            self._requests.append(msgs)
            return msgs, names, pending, self._calls

    def _scripted(self, msgs: tuple[Message, ...], calls: int) -> FakeReply:
        script = self._replies
        if calls <= len(script):
            return script[calls - 1]
        if self._llm.on_exhaust == "repeat_last" and script:
            return script[-1]
        last_user = next((m.content for m in reversed(msgs) if m.role == "user"), None)
        tail = f" (last user: {last_user[:60]!r})" if last_user is not None else ""
        raise ScriptExhaustedError(
            f"FakeLLM script exhausted after {len(script)} responses; "
            f"call #{calls} received {len(msgs)} messages{tail}"
        )

    def _close(self, pending: Any, msgs: tuple[Message, ...], reply: FakeReply) -> Message:
        with self._lock:
            self._last_request, self._last_reply = msgs, reply.message
        self._rec.finish(pending, result=to_dict(reply.message), usage=reply.usage)
        return reply.message

    def complete(self, messages: Any, tools: Any = None) -> Message:
        msgs, names, pending, calls = self._open(messages, tools)
        try:
            if self._llm.handler is not None:
                value = self._llm.handler(msgs, names)
                if inspect.isawaitable(value):
                    raise HarnessConfigError(
                        "the FakeLLM handler is async; call acomplete() or use a sync handler"
                    )
                reply = _as_reply(value)
            else:
                reply = self._scripted(msgs, calls)
        except BaseException as exc:
            self._rec.finish(pending, error=f"{type(exc).__name__}: {exc}")
            raise
        return self._close(pending, msgs, reply)

    async def acomplete(self, messages: Any, tools: Any = None) -> Message:
        msgs, names, pending, calls = self._open(messages, tools)
        try:
            if self._llm.handler is not None:
                value = self._llm.handler(msgs, names)
                if inspect.isawaitable(value):
                    value = await value
                reply = _as_reply(value)
            else:
                reply = self._scripted(msgs, calls)
        except BaseException as exc:
            self._rec.finish(pending, error=f"{type(exc).__name__}: {exc}")
            raise
        return self._close(pending, msgs, reply)
