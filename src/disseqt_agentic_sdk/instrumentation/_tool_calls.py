"""
Cross-provider tool-call normalization.

LLM providers ship tool calls in four different shapes on their responses:

  * OpenAI / Groq / Mistral / LiteLLM
        `choices[0].message.tool_calls[]` — each entry has
        `id`, `function.name`, `function.arguments` (JSON string).
  * Anthropic
        `content[]` blocks where `type == "tool_use"` — each block has
        `id`, `name`, `input` (parsed dict).
  * Google Gemini
        `candidates[0].content.parts[].function_call` — each entry has
        `name` and `args` (parsed dict); Gemini emits no id, so we
        synthesize one from the part index.
  * Cohere v2
        `message.tool_calls[]` — same as OpenAI shape.

To make dashboards, validators, and log queries consistent, this module
folds all four into a single canonical dict::

    {"id": str, "name": str, "arguments": str}   # arguments is JSON string

`arguments` is always a JSON string (never a dict) so consumers can
inspect it uniformly regardless of provider.
"""

from __future__ import annotations

import json
from typing import Any


def _read(obj: Any, name: str) -> Any:
    """Attribute-or-key read; tolerates dicts and Pydantic models."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _stringify_args(value: Any) -> str:
    """Return ``value`` as a JSON string; pass through if already a string."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def from_openai(tool_calls: Any) -> list[dict[str, str]]:
    """
    Normalize OpenAI-shape tool_calls (also Groq, Mistral, LiteLLM, Cohere v2).

    Input entries have ``id``, ``function.name``, ``function.arguments``.
    Returns [] on falsy input; skips entries missing a name.
    """
    if not tool_calls:
        return []
    out: list[dict[str, str]] = []
    for tc in tool_calls:
        function = _read(tc, "function")
        name = _read(function, "name") if function is not None else _read(tc, "name")
        if not name:
            continue
        args = _read(function, "arguments") if function is not None else _read(tc, "arguments")
        out.append(
            {
                "id": str(_read(tc, "id") or f"call_{len(out)}"),
                "name": str(name),
                "arguments": _stringify_args(args),
            }
        )
    return out


def from_anthropic(content_blocks: Any) -> list[dict[str, str]]:
    """
    Normalize Anthropic content blocks with ``type == "tool_use"``.

    Anthropic returns tool calls interleaved with text blocks in
    ``response.content``. Each tool_use block has ``id``, ``name``, and
    ``input`` (a parsed dict, not a JSON string).
    """
    if not content_blocks:
        return []
    out: list[dict[str, str]] = []
    for block in content_blocks:
        if _read(block, "type") != "tool_use":
            continue
        name = _read(block, "name")
        if not name:
            continue
        out.append(
            {
                "id": str(_read(block, "id") or f"call_{len(out)}"),
                "name": str(name),
                "arguments": _stringify_args(_read(block, "input")),
            }
        )
    return out


def from_gemini(parts: Any) -> list[dict[str, str]]:
    """
    Normalize Gemini ``candidates[0].content.parts[].function_call`` entries.

    Gemini emits no id per call, so we synthesize ``call_<index>``.
    Non-function parts (plain text) are skipped.
    """
    if not parts:
        return []
    out: list[dict[str, str]] = []
    for part in parts:
        fc = _read(part, "function_call")
        if fc is None:
            continue
        name = _read(fc, "name")
        if not name:
            continue
        out.append(
            {
                "id": f"call_{len(out)}",
                "name": str(name),
                "arguments": _stringify_args(_read(fc, "args")),
            }
        )
    return out
