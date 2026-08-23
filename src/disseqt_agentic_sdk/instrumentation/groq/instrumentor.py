"""
Groq SDK instrumentor.

Groq's `chat.completions.create` is OpenAI-shape-compatible, so we reuse
`_oai_compat` helpers wholesale — just point at the groq module path and
tag with the groq provider/system.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._kwargs import KW_STREAM
from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
    set_chat_response,
    set_common_chat_request,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span, safe_call
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.semantics import (
    AgenticOperation,
    AgenticProvider,
    GenAIOperation,
    GenAISystem,
)


class GroqInstrumentor(DisseqtInstrumentor):
    package_name = "groq"
    min_version = "0.11.0"

    def _instrument(self) -> None:
        self._wrap(
            "groq.resources.chat.completions",
            "Completions.create",
            _sync_chat(self),
        )
        self._wrap(
            "groq.resources.chat.completions",
            "AsyncCompletions.create",
            _async_chat(self),
        )


def _sync_chat(instrumentor: GroqInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(
            instrumentor.client, "groq.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        safe_call(
            set_common_chat_request,
            span,
            kwargs,
            provider=AgenticProvider.GROQ,
            system=GenAISystem.GROQ,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        if kwargs.get(KW_STREAM):
            state = ChatStreamAccumulator()
            return SyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _async_chat(instrumentor: GroqInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(
            instrumentor.client, "groq.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        safe_call(
            set_common_chat_request,
            span,
            kwargs,
            provider=AgenticProvider.GROQ,
            system=GenAISystem.GROQ,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        if kwargs.get(KW_STREAM):
            state = ChatStreamAccumulator()
            return AsyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper
