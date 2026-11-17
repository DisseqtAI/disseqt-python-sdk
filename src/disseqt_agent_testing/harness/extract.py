"""Turn whatever an agent returned into the run's final answer (documented precedence).

1. the user's ``extract(raw)``; 2. the adapter's ``extract`` (phase 1b); 3. a ``str`` as is;
4. a ``Message`` or an object with ``.content`` and ``.role``/``.type`` (text blocks joined);
5. a ``Mapping`` with ``"messages"`` (last assistant/ai item) or a well-known key;
6. a dataclass or pydantic-like model, structure kept; 7. ``None``; 8. anything else (lossy repr).
"""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import JSONValue, Message, capture_json, thaw
from .errors import W_LOSSY_ANSWER, W_NONE_ANSWER

ANSWER_KEYS = ("output", "final_output", "answer", "response", "result", "content", "text")
_ASSISTANT = ("assistant", "ai")


@dataclass(slots=True, frozen=True)
class Extracted:
    answer: JSONValue
    text: str | None
    warnings: tuple[str, ...] = ()


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def _content_text(content: Any) -> Any:
    """A string content as is; a list of blocks joined from their text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            else:
                text = _field(block, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return content


def _is_message_like(obj: Any) -> bool:
    if isinstance(obj, (str, Mapping, Sequence)):
        return False
    return hasattr(obj, "content") and (hasattr(obj, "role") or hasattr(obj, "type"))


def _pick(raw: Any) -> tuple[Any, bool]:
    """Apply rules 3 to 6; ``(value, True)`` if a rule decided, ``(None, False)`` otherwise."""
    if isinstance(raw, str):
        return raw, True
    if isinstance(raw, Message) or _is_message_like(raw):
        return _content_text(raw.content), True
    if isinstance(raw, Mapping):
        messages = raw.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) and messages:
            for item in reversed(messages):
                role = _field(item, "role") or _field(item, "type")
                if role in _ASSISTANT:
                    return _pick_message(item), True
        for key in ANSWER_KEYS:
            if key in raw:
                value = raw[key]
                return (_content_text(value) if key == "content" else value), True
        return raw, True
    if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
        return dataclasses.asdict(raw), True
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json"), True
        except Exception:
            return None, False
    return None, False


def _pick_message(item: Any) -> Any:
    if isinstance(item, Message) or _is_message_like(item):
        return _content_text(item.content)
    if isinstance(item, Mapping) and "content" in item:
        return _content_text(item["content"])
    value, decided = _pick(item)
    return value if decided else item


def final_text(answer: JSONValue) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, str):
        return answer
    return json.dumps(thaw(answer), sort_keys=True, ensure_ascii=False)


def extract_answer(
    raw: Any,
    user_extract: Callable[[Any], Any] | None = None,
    adapter_extract: Callable[[Any], tuple[bool, Any]] | None = None,
) -> Extracted:
    warnings: list[str] = []
    value: Any
    decided = False
    if user_extract is not None:
        try:
            value, decided = user_extract(raw), True
        except Exception as exc:
            return Extracted(None, None, (f"extract failed: {type(exc).__name__}: {exc}",))
    if not decided and adapter_extract is not None:
        decided, value = adapter_extract(raw)
    if not decided:
        value, decided = _pick(raw)
    if not decided:
        value = raw
    if value is None:
        return Extracted(None, None, (W_NONE_ANSWER,))
    answer, lossy = capture_json(value)
    if lossy:
        warnings.append(W_LOSSY_ANSWER)
    return Extracted(answer, final_text(answer), tuple(warnings))
