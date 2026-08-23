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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._kwargs import (
    KW_MESSAGES,
    KW_MODEL,
    KW_STREAM,
    KW_TOOLS,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._tool_calls import from_openai as _tc_from_openai
from disseqt_agentic_sdk.instrumentation._tool_result import (
    _notify_planned_tool_calls,
)
from disseqt_agentic_sdk.instrumentation._utils import (
    open_llm_span,
    safe_call,
    safe_set,
    serialize_messages,
    set_first_tool_call_attrs,
    set_messages_if_capturing,
)
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
        set_messages_if_capturing(span, input_messages=messages)
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
    # Tool calls come from choice 0 only (TP-2128 P2 #2.7). Flattening
    # tool_calls across choices misattributes ownership — the backend
    # validators fire on the AGENT_EXEC span and score plan-coherence
    # against a single planned tool list, so mixing choices in produces
    # false-positive plan divergence when n>1. Choice-0 matches how
    # RESPONSE_FINISH_REASON (singular) and the single-value convenience
    # attrs (TOOL_NAME / TOOL_CALL_ID / TOOL_ARGS) already behave.
    raw_tool_calls: list[Any] = []
    for i, choice in enumerate(choices):
        message = read(choice, "message")
        if message is not None:
            role = read(message, "role") or "assistant"
            # Reasoning-model responses (Mistral magistral-*, others)
            # ship a list of ContentChunks here; normalize to str.
            content = _extract_content_text(read(message, "content")) or ""
            output_messages.append({"role": role, "content": content})
            if i == 0:
                msg_tool_calls = read(message, "tool_calls")
                if msg_tool_calls:
                    raw_tool_calls.extend(msg_tool_calls)
        finish_reason = read(choice, "finish_reason")
        if finish_reason:
            finish_reasons.append(finish_reason)

    if output_messages:
        set_messages_if_capturing(span, output_messages=output_messages)
        safe_set(span, GenAIAttributes.COMPLETION, output_messages)
    if finish_reasons:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, finish_reasons[0])
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reasons)

    tool_calls = _tc_from_openai(raw_tool_calls)
    if tool_calls:
        safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
        _notify_planned_tool_calls(tool_calls)
        safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
        set_first_tool_call_attrs(span, tool_calls)


# ---------------------------------------------------------------------
# Content normalization for structured message content
# ---------------------------------------------------------------------
def _extract_content_text(content: Any) -> str | None:
    """
    Coerce a message-content value into plain text.

    OpenAI's chat completions traditionally return ``str`` for
    ``choices[i].message.content``. Mistral's reasoning-capable models
    (magistral-*, etc.) return a **list of ContentChunk objects** for
    the same field — e.g. ``[TextChunk(text=...), ThinkChunk(thinking=
    [Thinking(...), ...])]``. Feeding a list into ``"".join(...)`` in
    the accumulator crashes with TypeError, and because it happens
    inside ``contextlib.suppress(Exception)`` on the stream finalize
    path, the whole finalize step silently drops — no model, no
    response_id, no tokens, no finish_reason land on the span.
    TP-2128 P1 #1.9.

    We walk the list defensively:
      * ``TextChunk`` → ``.text``
      * ``ThinkChunk`` → concat ``.thinking[i].text`` (thinking blocks
        are themselves objects with a text field)
      * anything else → ``str(chunk)`` fallback so we never lose visibility
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            text = read(chunk, "text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            thinking = read(chunk, "thinking")
            if isinstance(thinking, list):
                for t in thinking:
                    t_text = read(t, "text")
                    if isinstance(t_text, str) and t_text:
                        parts.append(t_text)
                continue
            parts.append(str(chunk))
        return "".join(parts) if parts else None
    return str(content)


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
            # Streaming absorbs choice 0 only (TP-2128 P2 #2.7). The
            # deltas from other choices would collide on the tool-call
            # slot key (each choice restarts index=0) and interleave in
            # the content buffer, producing garbage attributes. Callers
            # using n>1 streaming today were already reading corrupt
            # output; restricting to choice 0 makes the behavior
            # consistent with non-streaming's tool_calls policy.
            # Real OpenAI ChatCompletionChunk.Choice always sets a
            # numeric .index. If a caller (or a MagicMock test) leaves
            # it unset we don't know which choice this chunk carries —
            # be permissive and treat it as choice 0 so a bare-mock
            # test isn't silently skipped.
            choice_idx = read(choice, "index")
            if isinstance(choice_idx, int) and choice_idx != 0:
                continue
            delta = read(choice, "delta")
            if delta is not None:
                role = read(delta, "role")
                if role:
                    self.role = role
                content = _extract_content_text(read(delta, "content"))
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
            set_messages_if_capturing(span, output_messages=msgs)
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
                _notify_planned_tool_calls(tool_calls)
                safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
                set_first_tool_call_attrs(span, tool_calls)


# ---------------------------------------------------------------------
# Wrapper factory
# ---------------------------------------------------------------------
def make_openai_shape_chat_wrappers(
    instrumentor: Any,
    *,
    sync_span_name: str,
    async_span_name: str | None = None,
    provider: str,
    system: str,
    operation_agentic: str,
    operation_gen_ai: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Build the (sync, async) wrapt wrappers for an OpenAI-shape
    ``chat.completions.create``-style call.

    Groq, LiteLLM, and any future provider that ships an OpenAI-
    compatible response used to hand-write two ~35-line wrappers that
    differed only in span_name / provider / system. Rolling that
    skeleton into one factory means fixes like TP-2128 P1 #1.1
    (``except Exception`` → ``except BaseException``) or a future
    span-attribute addition ripple in one place instead of N. See
    audit item TP-2128 P4 #4.4.

    ``async_span_name`` defaults to ``sync_span_name`` — pass a
    distinct value only when the SDK exposes different sync/async
    method names (e.g. LiteLLM's ``completion`` vs ``acompletion``).
    """
    async_span_name = async_span_name or sync_span_name

    def _apply_request_attrs(span: DisseqtSpan, kwargs: dict[str, Any]) -> None:
        safe_call(
            set_common_chat_request,
            span,
            kwargs,
            provider=provider,
            system=system,
            operation_agentic=operation_agentic,
            operation_gen_ai=operation_gen_ai,
        )

    def _make_stream_wrapper(cls: Any, result: Any, scope: Any, span: DisseqtSpan) -> Any:
        state = ChatStreamAccumulator()
        return cls(
            stream=result,
            scope=scope,
            on_chunk=lambda chunk: state.absorb(chunk),
            on_finish=lambda: state.finalize(span),
        )

    def sync_wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, sync_span_name, SpanKind.MODEL_EXEC)
        span = scope.span
        _apply_request_attrs(span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        if kwargs.get(KW_STREAM):
            return _make_stream_wrapper(SyncStreamWrapper, result, scope, span)
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    async def async_wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, async_span_name, SpanKind.MODEL_EXEC)
        span = scope.span
        _apply_request_attrs(span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        if kwargs.get(KW_STREAM):
            return _make_stream_wrapper(AsyncStreamWrapper, result, scope, span)
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return sync_wrapper, async_wrapper


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
