"""
Cohere v2 SDK instrumentor.

The Cohere v2 response shape differs from OpenAI:
  * `response.message.content` is a list of blocks with `.text`
  * `response.usage.tokens.input_tokens` / `.output_tokens`
  * `response.finish_reason` at top level

Streaming events use `event.type`:
  * `content-delta` carries `event.delta.message.content.text`
  * `message-end` carries final `event.delta.usage.tokens.*` + finish_reason
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._kwargs import KW_MESSAGES, KW_MODEL, KW_TOOLS
from disseqt_agentic_sdk.instrumentation._oai_compat import read
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
)
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticOperation,
    AgenticProvider,
    GenAIAttributes,
    GenAIOperation,
    GenAISystem,
)

if TYPE_CHECKING:
    from disseqt_agentic_sdk.span import DisseqtSpan


PROVIDER = AgenticProvider.COHERE
SYSTEM = GenAISystem.COHERE


class CohereInstrumentor(DisseqtInstrumentor):
    package_name = "cohere"
    min_version = "5.11.0"

    def _instrument(self) -> None:
        # v2 sync + async client
        self._wrap("cohere.v2.client", "V2Client.chat", _sync_chat(self))
        self._wrap("cohere.v2.client", "V2Client.chat_stream", _sync_stream(self))
        # AsyncV2Client lives in the same module.
        self._wrap("cohere.v2.client", "AsyncV2Client.chat", _async_chat(self))
        self._wrap("cohere.v2.client", "AsyncV2Client.chat_stream", _async_stream(self))


# ---------------------------------------------------------------------
# Attribute writers
# ---------------------------------------------------------------------
def _set_request_attrs(span: DisseqtSpan, kwargs: dict[str, Any]) -> None:
    model = kwargs.get(KW_MODEL, "")
    span.set_model_info(model, PROVIDER)
    span.set_operation(AgenticOperation.CHAT)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.CHAT)

    for key, agentic_key, gen_ai_key in (
        ("temperature", AgenticAttributes.REQUEST_TEMPERATURE, GenAIAttributes.REQUEST_TEMPERATURE),
        ("max_tokens", AgenticAttributes.REQUEST_MAX_TOKENS, GenAIAttributes.REQUEST_MAX_TOKENS),
        ("p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        ("k", AgenticAttributes.REQUEST_TOP_K, GenAIAttributes.REQUEST_TOP_K),
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

    messages = serialize_messages(kwargs.get(KW_MESSAGES))
    if messages:
        span.set_messages(input_messages=messages)
        safe_set(span, GenAIAttributes.PROMPT, messages)

    tools = kwargs.get(KW_TOOLS)
    if tools:
        try:
            tools_json = json.dumps(tools, default=str)
        except (TypeError, ValueError):
            tools_json = str(tools)
        safe_set(span, AgenticAttributes.REQUEST_TOOLS, tools_json)
        safe_set(span, GenAIAttributes.REQUEST_TOOLS, tools_json)


def _set_response_attrs(span: DisseqtSpan, response: Any) -> None:
    resp_id = read(response, "id")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)

    finish_reason = read(response, "finish_reason")
    if finish_reason:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, finish_reason)
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [finish_reason])

    input_t, output_t = _extract_usage(read(response, "usage"))
    if input_t is not None and output_t is not None:
        span.set_token_usage(input_t, output_t)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, input_t)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, output_t)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, input_t + output_t)

    message = read(response, "message")
    text = _extract_message_text(message)
    if text:
        msgs = [{"role": read(message, "role") or "assistant", "content": text}]
        span.set_messages(output_messages=msgs)
        safe_set(span, GenAIAttributes.COMPLETION, msgs)

    # Cohere v2 tool calls sit on message.tool_calls in OpenAI shape.
    raw_tool_calls = read(message, "tool_calls") if message is not None else None
    tool_calls = _tc_from_openai(raw_tool_calls)
    if tool_calls:
        safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
        _notify_planned_tool_calls(tool_calls)
        safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
        first = tool_calls[0]
        safe_set(span, AgenticAttributes.TOOL_NAME, first["name"])
        safe_set(span, GenAIAttributes.TOOL_NAME, first["name"])
        safe_set(span, AgenticAttributes.TOOL_CALL_ID, first["id"])
        safe_set(span, GenAIAttributes.TOOL_CALL_ID, first["id"])
        safe_set(span, AgenticAttributes.TOOL_ARGS, first["arguments"])
        safe_set(span, GenAIAttributes.TOOL_ARGS, first["arguments"])


def _extract_usage(usage: Any) -> tuple[int | None, int | None]:
    if usage is None:
        return None, None
    tokens = read(usage, "tokens")
    if tokens is None:
        return None, None
    return read(tokens, "input_tokens"), read(tokens, "output_tokens")


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""
    content = read(message, "content") or []
    parts: list[str] = []
    for block in content:
        text = read(block, "text")
        if text:
            parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------
def _sync_chat(instrumentor: CohereInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "cohere.chat", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_request_attrs, span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        safe_call(_set_response_attrs, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _async_chat(instrumentor: CohereInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "cohere.chat", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_request_attrs, span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        safe_call(_set_response_attrs, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------
class _StreamAccumulator:
    """Aggregates Cohere v2 stream events."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.response_id: str | None = None
        self.finish_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def absorb(self, event: Any) -> None:
        etype = read(event, "type")
        if etype == "message-start":
            self.response_id = self.response_id or read(event, "id")
        elif etype == "content-delta":
            delta = read(event, "delta")
            msg = read(delta, "message") if delta is not None else None
            content = read(msg, "content") if msg is not None else None
            text = read(content, "text") if content is not None else None
            if text:
                self.buffer.append(text)
        elif etype == "message-end":
            delta = read(event, "delta")
            if delta is not None:
                finish_reason = read(delta, "finish_reason")
                if finish_reason:
                    self.finish_reason = finish_reason
                usage = read(delta, "usage")
                if usage is not None:
                    inp, out = _extract_usage(usage)
                    if inp is not None:
                        self.input_tokens = inp
                    if out is not None:
                        self.output_tokens = out

    def finalize(self, span: DisseqtSpan) -> None:
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": "assistant", "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)
        if self.response_id:
            safe_set(span, AgenticAttributes.RESPONSE_ID, self.response_id)
            safe_set(span, GenAIAttributes.RESPONSE_ID, self.response_id)
        if self.finish_reason:
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, self.finish_reason)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [self.finish_reason])
        if self.input_tokens is not None and self.output_tokens is not None:
            span.set_token_usage(self.input_tokens, self.output_tokens)
            safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, self.input_tokens)
            safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, self.output_tokens)
            safe_set(
                span,
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self.input_tokens + self.output_tokens,
            )


def _sync_stream(instrumentor: CohereInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "cohere.chat_stream", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_request_attrs, span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = _StreamAccumulator()
        return SyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda evt: state.absorb(evt),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


def _async_stream(instrumentor: CohereInstrumentor) -> Callable[..., Any]:
    # NOTE: this wrapper is intentionally a plain `def`, not `async def`.
    # Cohere's `AsyncV2Client.chat_stream` is an async **generator function**,
    # not a coroutine — calling it returns an async-generator object
    # immediately. Awaiting an async generator raises
    # `TypeError: object async_generator can't be used in 'await' expression`.
    # wrapt calls the returned wrapper as `wrapper(wrapped, instance, args,
    # kwargs)` and passes its return value straight through to the caller,
    # so returning the AsyncStreamWrapper directly (which is itself an
    # async iterator) preserves `async for chunk in client.chat_stream(...)`.
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "cohere.chat_stream", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_request_attrs, span, kwargs)
        try:
            result = wrapped(*args, **kwargs)  # no await — this is already an async generator
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = _StreamAccumulator()
        return AsyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda evt: state.absorb(evt),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper
