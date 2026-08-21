"""
Anthropic patch functions.

Wraps `anthropic.resources.messages.Messages.create` (sync + async). Handles
streaming by wrapping the returned iterator so `message_delta` / `message_stop`
chunks land token counts on the span.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._kwargs import (
    KW_MESSAGES,
    KW_MODEL,
    KW_STREAM,
    KW_SYSTEM,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import (
    open_llm_span,
    safe_set,
    serialize_messages,
)
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticOperation,
    AgenticProvider,
    GenAIAttributes,
    GenAIOperation,
    GenAISystem,
)

if TYPE_CHECKING:
    from disseqt_agentic_sdk.instrumentation.anthropic.instrumentor import AnthropicInstrumentor
    from disseqt_agentic_sdk.span import DisseqtSpan


PROVIDER = AgenticProvider.ANTHROPIC
SYSTEM = GenAISystem.ANTHROPIC


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
        ("top_p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        ("top_k", AgenticAttributes.REQUEST_TOP_K, GenAIAttributes.REQUEST_TOP_K),
    ):
        val = kwargs.get(key)
        if val is not None:
            safe_set(span, agentic_key, val)
            safe_set(span, gen_ai_key, val)

    if KW_STREAM in kwargs:
        safe_set(span, GenAIAttributes.REQUEST_IS_STREAM, bool(kwargs[KW_STREAM]))

    system_prompt = kwargs.get(KW_SYSTEM)
    if system_prompt:
        safe_set(span, AgenticAttributes.SYSTEM_INSTRUCTIONS, system_prompt)

    messages = serialize_messages(kwargs.get(KW_MESSAGES))
    if messages:
        span.set_messages(input_messages=messages)
        safe_set(span, GenAIAttributes.PROMPT, messages)


def _set_response_attrs(span: DisseqtSpan, response: Any) -> None:
    resp_id = _read(response, "id")
    resp_model = _read(response, "model")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    stop_reason = _read(response, "stop_reason")
    if stop_reason:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, stop_reason)
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [stop_reason])

    usage = _read(response, "usage")
    if usage is not None:
        input_tokens = _read(usage, "input_tokens") or 0
        output_tokens = _read(usage, "output_tokens") or 0
        span.set_token_usage(input_tokens, output_tokens)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, input_tokens + output_tokens)

    # Anthropic returns response.content as a list of blocks ({"type":"text","text":...}).
    content_blocks = _read(response, "content") or []
    text_parts: list[str] = []
    for block in content_blocks:
        if _read(block, "type") == "text":
            text_val = _read(block, "text") or ""
            text_parts.append(text_val)
    if text_parts:
        msgs = [{"role": "assistant", "content": "".join(text_parts)}]
        span.set_messages(output_messages=msgs)
        safe_set(span, GenAIAttributes.COMPLETION, msgs)


def messages_create(instrumentor: AnthropicInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "anthropic.messages.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_request_attrs(span, kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get(KW_STREAM):
            state = _StreamAccumulator()
            return SyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        _set_response_attrs(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_messages_create(instrumentor: AnthropicInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "anthropic.messages.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_request_attrs(span, kwargs)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get(KW_STREAM):
            state = _StreamAccumulator()
            return AsyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        _set_response_attrs(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


class _StreamAccumulator:
    """
    Anthropic streams emit typed events: `message_start`, `content_block_start`,
    `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.
    Usage arrives on `message_start` (input tokens) and `message_delta` (output tokens).
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.model: str | None = None
        self.response_id: str | None = None
        self.stop_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def absorb(self, chunk: Any) -> None:
        event_type = _read(chunk, "type")

        if event_type == "message_start":
            message = _read(chunk, "message")
            if message is not None:
                self.model = self.model or _read(message, "model")
                self.response_id = self.response_id or _read(message, "id")
                usage = _read(message, "usage")
                if usage is not None:
                    self.input_tokens = _read(usage, "input_tokens") or self.input_tokens
                    self.output_tokens = _read(usage, "output_tokens") or self.output_tokens

        elif event_type == "content_block_delta":
            delta = _read(chunk, "delta")
            if delta is not None and _read(delta, "type") == "text_delta":
                text_val = _read(delta, "text") or ""
                if text_val:
                    self.buffer.append(text_val)

        elif event_type == "message_delta":
            delta = _read(chunk, "delta")
            if delta is not None:
                stop_reason = _read(delta, "stop_reason")
                if stop_reason:
                    self.stop_reason = stop_reason
            usage = _read(chunk, "usage")
            if usage is not None:
                out_tokens = _read(usage, "output_tokens")
                if out_tokens is not None:
                    self.output_tokens = out_tokens

    def finalize(self, span: DisseqtSpan) -> None:
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": "assistant", "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)
        if self.model:
            safe_set(span, AgenticAttributes.RESPONSE_MODEL, self.model)
            safe_set(span, GenAIAttributes.RESPONSE_MODEL, self.model)
        if self.response_id:
            safe_set(span, AgenticAttributes.RESPONSE_ID, self.response_id)
            safe_set(span, GenAIAttributes.RESPONSE_ID, self.response_id)
        if self.stop_reason:
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, self.stop_reason)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [self.stop_reason])
        if self.input_tokens is not None and self.output_tokens is not None:
            span.set_token_usage(self.input_tokens, self.output_tokens)
            safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, self.input_tokens)
            safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, self.output_tokens)
            safe_set(
                span,
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self.input_tokens + self.output_tokens,
            )


def _read(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
