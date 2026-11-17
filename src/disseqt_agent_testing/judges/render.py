"""Deterministic rendering of judge inputs: redaction, truncation, tag escaping, size caps.

Order of operations for every field: redact secrets, truncate, then escape ``</`` so recorded data
can never close a tag the prompt relies on."""

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..assertions.canonical import canonical_json
from .templates import JudgeTemplate

TEXT_CAP = 4_000
RESULT_CAP = 2_000
MAX_MESSAGES = 20
MAX_CALLS = 40
TOTAL_CAP = 32_000

DEFAULT_REDACT: tuple[str, ...] = (
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",
    r"(?i)(?:api[_-]?key|authorization|secret|password|token)\s*[:=]\s*\S+",
)
REDACTED = "[REDACTED]"


def redact(text: str, patterns: Sequence[str] = ()) -> str:
    for pattern in (*DEFAULT_REDACT, *patterns):
        text = re.sub(pattern, REDACTED, text)
    return text


def truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return f"{text[:cap]}...[truncated {len(text) - cap} chars]"


def escape(text: str) -> str:
    return text.replace("</", "<\\/")


def as_text(value: Any) -> str:
    """Any JSON-ish value as stable text (strings as is, structures as canonical JSON)."""
    if isinstance(value, str):
        return value
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return str(value)


def fmt_conversation(conversation: Any) -> str:
    """``[user t0] text`` lines; the last 20 messages, each capped."""
    items = (
        list(conversation)
        if isinstance(conversation, Sequence) and not isinstance(conversation, str)
        else []
    )
    if isinstance(conversation, str):
        return truncate(conversation, TEXT_CAP)
    omitted = max(0, len(items) - MAX_MESSAGES)
    lines = []
    if omitted:
        lines.append(f"[{omitted} earlier messages omitted]")
    for m in items[-MAX_MESSAGES:]:
        if isinstance(m, Mapping):
            lines.append(
                f"[{m.get('role', '?')} t{m.get('turn', 0)}] {truncate(as_text(m.get('text', '')), TEXT_CAP)}"
            )
        else:
            lines.append(truncate(as_text(m), TEXT_CAP))
    return "\n".join(lines) or "(empty)"


def fmt_tool_calls(calls: Any) -> str:
    """``call-3 issue_refund {"order_id":"A12"} -> {"status":"ok"} (write)``; at most 40 calls."""
    items = list(calls) if isinstance(calls, Sequence) and not isinstance(calls, str) else []
    if not items:
        return "(none)"
    lines = []
    for c in items[:MAX_CALLS]:
        if not isinstance(c, Mapping):
            lines.append(truncate(as_text(c), RESULT_CAP))
            continue
        out = (
            f"-> ERROR {c['error']}"
            if c.get("error")
            else f"-> {truncate(as_text(c.get('result')), RESULT_CAP)}"
        )
        effect = f" ({c['effect']})" if c.get("effect") else ""
        lines.append(
            f"{c.get('call_id', '?')} {c.get('name', '?')} {as_text(c.get('args', {}))} {out}{effect}"
        )
    if len(items) > MAX_CALLS:
        lines.append(f"[{len(items) - MAX_CALLS} calls omitted]")
    return "\n".join(lines)


def fmt_list(values: Any) -> str:
    items = list(values) if isinstance(values, Sequence) and not isinstance(values, str) else []
    return "\n".join(f"- {as_text(v)}" for v in items) if items else "(none)"


def render_messages(
    template: JudgeTemplate,
    blocks: Sequence[tuple[str, str]],
    redact_patterns: Sequence[str] = (),
    total_cap: int = TOTAL_CAP,
) -> list[dict[str, str]]:
    """The system and user messages. The user message is at most ``total_cap`` characters."""
    cleaned = [(tag, truncate(redact(text, redact_patterns), TEXT_CAP * 2)) for tag, text in blocks]

    def build(parts: list[tuple[str, str]]) -> str:
        return "\n\n".join(f"<{tag}>\n{escape(text)}\n</{tag}>" for tag, text in parts)

    user = build(cleaned)
    while len(user) > total_cap:  # shrink the biggest block until it fits
        sizes = [len(t) for _, t in cleaned]
        i = sizes.index(max(sizes))
        over = len(user) - total_cap
        keep = max(200, sizes[i] - over - 40)
        if keep >= sizes[i]:
            break
        cleaned[i] = (cleaned[i][0], truncate(cleaned[i][1], keep))
        user = build(cleaned)
    return [{"role": "system", "content": template.system}, {"role": "user", "content": user}]


_ = json
