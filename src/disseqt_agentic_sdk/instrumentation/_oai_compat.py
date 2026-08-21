"""
Shared helpers for OpenAI-compatible provider SDKs.

OpenAI, Groq, Mistral, LiteLLM (and many others) return the same
response shape: `choices[0].message.content`, `usage.prompt_tokens`,
`usage.completion_tokens`. Streaming events all carry deltas under
`choices[0].delta`. Instrumentors for those SDKs share this module
instead of copy-pasting 200 lines of attribute-mapping logic.

Provider-specific instrumentors just pass the right `provider`
(AgenticProvider.*) and `system` (GenAISystem.*) values.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.instrumentation._kwargs import (
    KW_MESSAGES,
    KW_MODEL,
    KW_STREAM,
    KW_TOOLS,
)
from disseqt_agentic_sdk.instrumentation._tool_calls import from_openai as _tc_from_openai
from disseqt_agentic_sdk.instrumentation._utils import safe_set, serialize_messages
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes

if TYPE_CHECKING:
    from disseqt_agentic_sdk.span import DisseqtSpan


# ---------------------------------------------------------------------
# Request-side
# ---------------------------------------------------------------------
def set_common_chat_request(
    span: DisseqtSpan,
    kwargs: dict[str, Any],
    *,
    provider: str,
    system: str,
    operation_agentic: str,
    operation_gen_ai: str,
) -> None:
    """
    Populate the span with model + common request params before the wrapped
    call runs. Handles the params every OpenAI-shaped SDK accepts:
    temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
    plus `stream` and `messages`.
    """
    model = kwargs.get(KW_MODEL, "")
    span.set_model_info(model, provider)
    span.set_operation(operation_agentic)

    safe_set(span, GenAIAttributes.SYSTEM, system)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, operation_gen_ai)

    for key, agentic_key, gen_ai_key in (
        ("temperature", AgenticAttributes.REQUEST_TEMPERATURE, GenAIAttributes.REQUEST_TEMPERATURE),
        ("max_tokens", AgenticAttributes.REQUEST_MAX_TOKENS, GenAIAttributes.REQUEST_MAX_TOKENS),
        ("top_p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        (
            "frequency_penalty",
            AgenticAttributes.REQUEST_FREQUENCY_PENALTY,
            GenAIAttributes.REQUEST_FREQUENCY_PENALTY,
        ),
        (
            "presence_penalty",
            AgenticAttributes.REQUEST_PRESENCE_PENALTY,
            GenAIAttributes.REQUEST_PRESENCE_PENALTY,
        ),
    ):
        val = kwargs.get(key)
        if val is not None:
            safe_set(span, agentic_key, val)
            safe_set(span, gen_ai_key, val)

    if KW_STREAM in kwargs:
        safe_set(span, GenAIAttributes.REQUEST_IS_STREAM, bool(kwargs[KW_STREAM]))

    messages = serialize_messages(kwargs.get(KW_MESSAGES))
    if messages:
        span.set_messages(input_messages=messages)
        safe_set(span, GenAIAttributes.PROMPT, messages)

    tools = kwargs.get(KW_TOOLS)
    if tools:
        # Serialize to JSON so downstream consumers see a stable string
        # regardless of whether the caller passed dicts or Pydantic models.
        try:
            tools_json = json.dumps(tools, default=str)
        except (TypeError, ValueError):
            tools_json = str(tools)
        safe_set(span, AgenticAttributes.REQUEST_TOOLS, tools_json)
        safe_set(span, GenAIAttributes.REQUEST_TOOLS, tools_json)


# ---------------------------------------------------------------------
# Response-side (non-streaming)
# ---------------------------------------------------------------------
def set_chat_response(span: DisseqtSpan, response: Any) -> None:
    """
    Populate response id/model/tokens/messages/finish_reason from an
    OpenAI-shaped ChatCompletion. Works for OpenAI, Groq, Mistral,
    LiteLLM, and any provider matching that structure.
    """
    resp_id = read(response, "id")
    resp_model = read(response, "model")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    usage = read(response, "usage")
    if usage is not None:
        prompt_tokens = read(usage, "prompt_tokens") or read(usage, "input_tokens") or 0
        completion_tokens = read(usage, "completion_tokens") or read(usage, "output_tokens") or 0
        span.set_token_usage(prompt_tokens, completion_tokens)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, completion_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, prompt_tokens + completion_tokens)

    choices = read(response, "choices") or []
    output_messages: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    raw_tool_calls: list[Any] = []
    for choice in choices:
        message = read(choice, "message")
        if message is not None:
            role = read(message, "role") or "assistant"
            content = read(message, "content") or ""
            output_messages.append({"role": role, "content": content})
            msg_tool_calls = read(message, "tool_calls")
            if msg_tool_calls:
                raw_tool_calls.extend(msg_tool_calls)
        finish_reason = read(choice, "finish_reason")
        if finish_reason:
            finish_reasons.append(finish_reason)

    if output_messages:
        span.set_messages(output_messages=output_messages)
        safe_set(span, GenAIAttributes.COMPLETION, output_messages)
    if finish_reasons:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, finish_reasons[0])
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reasons)

    tool_calls = _tc_from_openai(raw_tool_calls)
    if tool_calls:
        safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
        safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
        # Populate the single-tool columns from the first call so the
        # backend's enriched-table columns (agentic.tool.name /
        # agentic.tool.call_id / agentic.tool.args) get a value even for
        # dashboards that don't query the JSON array.
        first = tool_calls[0]
        safe_set(span, AgenticAttributes.TOOL_NAME, first["name"])
        safe_set(span, GenAIAttributes.TOOL_NAME, first["name"])
        safe_set(span, AgenticAttributes.TOOL_CALL_ID, first["id"])
        safe_set(span, GenAIAttributes.TOOL_CALL_ID, first["id"])
        safe_set(span, AgenticAttributes.TOOL_ARGS, first["arguments"])
        safe_set(span, GenAIAttributes.TOOL_ARGS, first["arguments"])


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------
class ChatStreamAccumulator:
    """
    Accumulates streamed OpenAI-shaped chat chunks into text + token counts.
    Providers hand each chunk to `absorb(chunk)`; when the stream ends,
    `finalize(span)` writes the aggregated attributes.
    """

    def __init__(self) -> None:
        """Initialize an empty accumulator; call absorb() per chunk."""
        self.buffer: list[str] = []
        self.role: str = "assistant"
        self.finish_reason: str | None = None
        self.model: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        # OpenAI streams tool calls one index at a time, with `function.arguments`
        # arriving as concatenatable text fragments. Merge by `index`.
        self._tool_calls: dict[int, dict[str, Any]] = {}

    def absorb(self, chunk: Any) -> None:
        """
        Fold one streaming chunk into the running totals.

        Reads model/id from the first chunk that carries them, appends any
        content delta to the buffer, and captures token counts and finish
        reason when they appear (typically on the final chunk, or the
        after-last chunk when ``stream_options={"include_usage": True}``).
        """
        self.model = self.model or read(chunk, "model")
        self.response_id = self.response_id or read(chunk, "id")

        # `stream_options={"include_usage": True}` sends usage on a final chunk.
        usage = read(chunk, "usage")
        if usage is not None:
            self.prompt_tokens = (
                read(usage, "prompt_tokens") or read(usage, "input_tokens") or self.prompt_tokens
            )
            self.completion_tokens = (
                read(usage, "completion_tokens")
                or read(usage, "output_tokens")
                or self.completion_tokens
            )

        for choice in read(chunk, "choices") or []:
            delta = read(choice, "delta")
            if delta is not None:
                role = read(delta, "role")
                if role:
                    self.role = role
                content = read(delta, "content")
                if content:
                    self.buffer.append(content)
                self._absorb_tool_call_deltas(read(delta, "tool_calls"))
            finish_reason = read(choice, "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason

    def _absorb_tool_call_deltas(self, deltas: Any) -> None:
        """
        Fold streamed ``delta.tool_calls`` fragments into ``self._tool_calls``.

        OpenAI emits each tool call across multiple chunks: the first chunk
        for a given ``index`` carries ``id``, ``function.name`` and an opening
        ``function.arguments`` fragment; subsequent chunks for the same index
        carry additional ``function.arguments`` text that must be concatenated.
        """
        if not deltas:
            return
        for delta in deltas:
            idx = read(delta, "index")
            if idx is None:
                continue
            slot = self._tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            new_id = read(delta, "id")
            if new_id:
                slot["id"] = new_id
            function = read(delta, "function")
            if function is not None:
                new_name = read(function, "name")
                if new_name:
                    slot["name"] = new_name
                arg_frag = read(function, "arguments")
                if arg_frag:
                    slot["arguments"] = (slot["arguments"] or "") + arg_frag

    def finalize(self, span: DisseqtSpan) -> None:
        """
        Write the aggregated output attributes onto ``span`` at stream end.

        Idempotent-friendly: fields left as None (e.g. missing token counts
        when usage isn't included) are skipped rather than zeroed, so the
        span reflects "unknown" not "0".
        """
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": self.role, "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)
        if self.model:
            safe_set(span, AgenticAttributes.RESPONSE_MODEL, self.model)
            safe_set(span, GenAIAttributes.RESPONSE_MODEL, self.model)
        if self.response_id:
            safe_set(span, AgenticAttributes.RESPONSE_ID, self.response_id)
            safe_set(span, GenAIAttributes.RESPONSE_ID, self.response_id)
        if self.finish_reason:
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, self.finish_reason)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [self.finish_reason])
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            span.set_token_usage(self.prompt_tokens, self.completion_tokens)
            safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, self.prompt_tokens)
            safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, self.completion_tokens)
            safe_set(
                span,
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self.prompt_tokens + self.completion_tokens,
            )

        if self._tool_calls:
            # Flatten index-keyed dict into list ordered by index, drop entries
            # missing a name, then feed through the OpenAI adapter for canonical
            # {id, name, arguments} shape.
            ordered = [
                {
                    "id": slot["id"],
                    "function": {"name": slot["name"], "arguments": slot["arguments"]},
                }
                for _idx, slot in sorted(self._tool_calls.items())
                if slot.get("name")
            ]
            tool_calls = _tc_from_openai(ordered)
            if tool_calls:
                safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
                safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
                first = tool_calls[0]
                safe_set(span, AgenticAttributes.TOOL_NAME, first["name"])
                safe_set(span, GenAIAttributes.TOOL_NAME, first["name"])
                safe_set(span, AgenticAttributes.TOOL_CALL_ID, first["id"])
                safe_set(span, GenAIAttributes.TOOL_CALL_ID, first["id"])
                safe_set(span, AgenticAttributes.TOOL_ARGS, first["arguments"])
                safe_set(span, GenAIAttributes.TOOL_ARGS, first["arguments"])


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def read(obj: Any, name: str) -> Any:
    """
    Read a field from a provider response tolerating shape drift.

    Provider SDKs occasionally shuffle between Pydantic models (attribute
    access) and plain dicts (key access) across minor releases — sometimes
    within the same response tree. This helper unifies both so instrumentors
    don't need per-provider branches. Returns None on missing keys, missing
    attributes, or ``obj is None``; never raises.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
