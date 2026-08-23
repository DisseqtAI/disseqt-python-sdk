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

from disseqt_agentic_sdk.instrumentation._utils import (
    read as _read,  # noqa: F401 — re-exported for local callers
)


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

    Recognized shapes, in order:

    1. **Function tool call** (default): entry has ``function.name`` +
       ``function.arguments`` (already a JSON string on OpenAI).
    2. **Custom tool call**: entry has ``type == "custom"`` +
       ``custom.name`` + ``custom.input`` (a free-form string, not
       necessarily JSON). This is OpenAI's currently-shipping "custom
       tool calling" feature — before the fix, `.function` came back
       None, `.name` came back None, and the entry was silently dropped
       (TP-2128 P1 #1.4).
    3. Legacy / permissive fallback: entry has top-level ``name`` +
       ``arguments``.

    Returns [] on falsy input; skips entries missing a name.
    """
    if not tool_calls:
        return []
    out: list[dict[str, str]] = []
    for tc in tool_calls:
        function = _read(tc, "function")
        custom = _read(tc, "custom")

        if function is not None:
            name = _read(function, "name")
            args = _read(function, "arguments")
        elif custom is not None:
            name = _read(custom, "name")
            # OpenAI custom tools use `input` (free-form text), not `arguments`.
            args = _read(custom, "input")
        else:
            name = _read(tc, "name")
            args = _read(tc, "arguments")

        if not name:
            continue
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


def from_gemini(parts: Any, *, response_id: str | None = None) -> list[dict[str, str]]:
    """
    Normalize Gemini ``candidates[0].content.parts[].function_call`` entries.

    IDs are chosen in order:

    1. ``function_call.id`` if the real Gemini `FunctionCall` type has
       it populated (some Gemini deployments do set this — TP-2128 audit
       item #2.9).
    2. ``{response_id}_call_{index}`` when the caller passes ``response_id``.
       This is the fix for TP-2128 P1 #1.3: two Gemini responses within
       the same ``agent_span`` used to synthesize ``call_0`` for both, so
       the Lane-B aggregator's ``setdefault`` merge silently dropped the
       second. Using the response_id as a per-response namespace keeps
       ids unique across responses without needing extra plumbing.
    3. Plain ``call_{index}`` when no response_id is available. Still
       colliding across responses, but the only realistic case that hits
       this branch is a bare-mock test or a partial response object.

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
        real_id = _read(fc, "id")
        # isinstance(str) guards against MagicMock in tests (bare mocks
        # return a truthy fresh MagicMock for any unset attribute, which
        # would otherwise get stringified as "<MagicMock ...>").
        if isinstance(real_id, str) and real_id:
            call_id = real_id
        elif response_id:
            call_id = f"{response_id}_call_{len(out)}"
        else:
            call_id = f"call_{len(out)}"
        out.append(
            {
                "id": call_id,
                "name": str(name),
                "arguments": _stringify_args(_read(fc, "args")),
            }
        )
    return out
