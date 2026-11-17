"""``Message`` and ``MsgToolCall``: one conversation message, with optional tool calls.

Design rule (from the tool-call research): parse tool-call arguments **once, at the boundary**,
into a dict plus a typed failure record. ``MsgToolCall.from_wire`` does that and never raises on
bad model output, so malformed JSON can never survive in history and crash a later conversion.
"""

import json
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, ClassVar, Literal

from ._frozen import FrozenDict, JSONObject, freeze_mapping
from ._validate import check_opt_str, check_str, fail, set_field
from .errors import ModelValidationError

Role = Literal["system", "user", "assistant", "tool"]
ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")


@dataclass(slots=True, frozen=True)
class MsgToolCall:
    """One tool call requested by the model (parallel calls are several entries)."""

    id: str
    name: str
    _: KW_ONLY
    arguments: JSONObject = field(default_factory=FrozenDict)
    raw_arguments: str | None = None
    parse_error: str | None = None

    # JSON form omits fields at their defaults (messages are the most numerous objects).
    SERDE_OMIT_DEFAULTS: ClassVar[bool] = True
    # ``from_dict(fill_ids=True)`` fills a missing id as ``call_<n>`` (see _serde).
    SERDE_FILL_ID: ClassVar[tuple[str, str]] = ("id", "call_{n}")

    def __post_init__(self) -> None:
        check_str(self.id, "MsgToolCall.id")
        check_str(self.name, "MsgToolCall.name")
        set_field(self, "arguments", freeze_mapping(self.arguments, "MsgToolCall.arguments"))
        if self.parse_error is not None:
            check_str(self.parse_error, "MsgToolCall.parse_error")
            if not isinstance(self.raw_arguments, str):
                fail("MsgToolCall.raw_arguments", "must be set when parse_error is set")
            if self.arguments:
                fail("MsgToolCall.arguments", "must be empty when parse_error is set")
        elif self.raw_arguments is not None:
            fail("MsgToolCall.raw_arguments", "is only allowed when parse_error is set")

    @classmethod
    def from_wire(
        cls, id: str, name: str, arguments: "str | dict[str, Any] | None" = None
    ) -> "MsgToolCall":
        """Build a call from provider data. Bad *arguments* never raise.

        ``None`` and ``""`` become ``{}``. A dict passes through. A string is parsed once with
        ``json.loads``; invalid JSON, NaN/Infinity, or JSON that is not an object give empty
        arguments plus ``raw_arguments`` (verbatim) and ``parse_error``. ``id`` and ``name`` are
        still required (adapters synthesize ids before calling this).
        """
        if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
            return cls(id, name)
        if isinstance(arguments, str):
            return cls._from_text(id, name, arguments)
        try:
            return cls(id, name, arguments=arguments)
        except ModelValidationError as exc:
            return cls(
                id, name, raw_arguments=repr(arguments)[:2000], parse_error=_one_line(str(exc))
            )

    @classmethod
    def _from_text(cls, id: str, name: str, text: str) -> "MsgToolCall":
        def reject_constant(constant: str) -> Any:
            raise ValueError(f"{constant} is not valid JSON")

        try:
            parsed = json.loads(text, parse_constant=reject_constant)
        except (ValueError, RecursionError) as exc:
            return cls(id, name, raw_arguments=text, parse_error=_one_line(str(exc)))
        if not isinstance(parsed, dict):
            return cls(
                id,
                name,
                raw_arguments=text,
                parse_error=f"expected a JSON object, got {type(parsed).__name__}",
            )
        try:
            return cls(id, name, arguments=parsed)
        except ModelValidationError as exc:
            return cls(id, name, raw_arguments=text, parse_error=_one_line(str(exc)))


def _one_line(text: str) -> str:
    return " ".join(text.split())[:500] or "invalid arguments"


@dataclass(slots=True, frozen=True)
class Message:
    """One message of a conversation."""

    role: Role
    content: str = ""
    _: KW_ONLY
    tool_calls: tuple[MsgToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    SERDE_OMIT_DEFAULTS: ClassVar[bool] = True
    SERDE_ALWAYS: ClassVar[tuple[str, ...]] = ("role", "content")
    SERDE_NONE_AS_DEFAULT: ClassVar[tuple[str, ...]] = ("content",)  # null loads as ""

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            fail("Message.role", f"must be one of {list(ROLES)}, got {self.role!r}")
        content = "" if self.content is None else self.content
        check_str(content, "Message.content", non_empty=False)
        set_field(self, "content", content)
        if self.role in ("user", "system") and not content.strip():
            fail("Message.content", f"must not be empty for role {self.role!r}")

        if not isinstance(self.tool_calls, (list, tuple)):
            fail(
                "Message.tool_calls",
                f"must be a list or tuple, got {type(self.tool_calls).__name__}",
            )
        calls = tuple(self.tool_calls)
        for index, call in enumerate(calls):
            if not isinstance(call, MsgToolCall):
                fail(f"Message.tool_calls[{index}]", "must be a MsgToolCall")
        if calls and self.role != "assistant":
            fail("Message.tool_calls", f"is only allowed on assistant messages, not {self.role!r}")
        ids = [call.id for call in calls]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            fail(
                "Message.tool_calls",
                f"call ids must be unique within a message, repeated: {duplicates}",
            )
        set_field(self, "tool_calls", calls)

        if self.role == "tool":
            check_str(self.tool_call_id, "Message.tool_call_id")
        elif self.tool_call_id is not None:
            fail("Message.tool_call_id", f"is only allowed on tool messages, not {self.role!r}")
        check_opt_str(self.name, "Message.name")


def validate_tool_correlation(messages: Sequence[Message]) -> None:
    """Check a full history: every tool call is answered, and every tool message answers one.

    Each ``tool`` message must answer an id from the immediately preceding assistant message
    (only tool messages in between), and every call must be answered exactly once. Raises
    ``ModelValidationError`` describing the first problem. Not run on ``FakeLLM`` scripts or on
    ``AgentTestCase.input`` automatically.
    """
    pending: dict[str, int] = {}
    answered: set[str] = set()
    owner = -1

    def check_complete(position: int) -> None:
        if pending:
            fail(
                f"messages[{owner}]",
                f"tool call(s) {list(pending)} were not answered before messages[{position}]",
            )

    for index, message in enumerate(messages):
        if message.role == "tool":
            call_id = message.tool_call_id or ""
            if call_id in answered:
                fail(f"messages[{index}]", f"tool call {call_id!r} is answered more than once")
            if call_id not in pending:
                fail(
                    f"messages[{index}]",
                    f"tool message answers {call_id!r}, which is not a call of the preceding "
                    "assistant message",
                )
            del pending[call_id]
            answered.add(call_id)
            continue
        check_complete(index)
        pending = {}
        answered = set()
        if message.role == "assistant" and message.tool_calls:
            pending = {call.id: index for call in message.tool_calls}
            owner = index
    check_complete(len(messages))
