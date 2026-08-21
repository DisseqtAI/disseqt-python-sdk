"""
Mistral SDK (mistralai v1.x+) instrumentor.

Non-streaming `complete` / `complete_async` return ChatCompletionResponse
in an OpenAI-compatible shape. `stream` / `stream_async` return an iterable
of `CompletionEvent` (which wraps `event.data.choices[0].delta` — same
shape as OpenAI once we drill through the `data` field).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
    read,
    set_chat_response,
    set_common_chat_request,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.semantics import (
    AgenticOperation,
    AgenticProvider,
    GenAIOperation,
    GenAISystem,
)


class MistralInstrumentor(DisseqtInstrumentor):
    # The pip package is `mistralai` in v1.x+.
    package_name = "mistralai"
    min_version = "1.5.0"

    def _instrument(self) -> None:
        # Non-streaming
        self._wrap("mistralai.client.chat", "Chat.complete", _sync_chat(self))
        self._wrap("mistralai.client.chat", "Chat.complete_async", _async_chat(self))
        # Streaming
        self._wrap("mistralai.client.chat", "Chat.stream", _sync_stream(self))
        self._wrap("mistralai.client.chat", "Chat.stream_async", _async_stream(self))


def _sync_chat(instrumentor: MistralInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "mistral.chat.complete", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.MISTRAL_AI,
            system=GenAISystem.MISTRAL_AI,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _async_chat(instrumentor: MistralInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "mistral.chat.complete", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.MISTRAL_AI,
            system=GenAISystem.MISTRAL_AI,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _sync_stream(instrumentor: MistralInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "mistral.chat.stream", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.MISTRAL_AI,
            system=GenAISystem.MISTRAL_AI,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = ChatStreamAccumulator()
        return SyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda evt: state.absorb(_unwrap_event(evt)),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


def _async_stream(instrumentor: MistralInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "mistral.chat.stream", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.MISTRAL_AI,
            system=GenAISystem.MISTRAL_AI,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = ChatStreamAccumulator()
        return AsyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda evt: state.absorb(_unwrap_event(evt)),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


def _unwrap_event(event: Any) -> Any:
    """
    Mistral streams `CompletionEvent(data=CompletionChunk(...))`. Peel the
    `data` field so downstream OpenAI-compat accumulator sees the chunk
    directly.
    """
    data = read(event, "data")
    return data if data is not None else event
