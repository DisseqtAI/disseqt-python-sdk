"""
OpenAI patch functions.

Chat completions and embeddings both use the shared `_oai_compat` helpers.
Legacy text completions have their own thin wrapper because they take
`prompt=` instead of `messages=`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
    read,
    set_chat_response,
    set_common_chat_request,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span, safe_set
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticOperation,
    AgenticProvider,
    GenAIAttributes,
    GenAIOperation,
    GenAISystem,
)

if TYPE_CHECKING:
    from disseqt_agentic_sdk.instrumentation.openai.instrumentor import OpenAIInstrumentor
    from disseqt_agentic_sdk.span import DisseqtSpan


PROVIDER = AgenticProvider.OPENAI
SYSTEM = GenAISystem.OPENAI


# ---------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------
def chat_completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(
            instrumentor.client, "openai.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get("stream"):
            state = ChatStreamAccumulator()
            return SyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_chat_completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(
            instrumentor.client, "openai.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get("stream"):
            state = ChatStreamAccumulator()
            return AsyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: state.absorb(chunk),
                on_finish=lambda: state.finalize(span),
            )
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# Legacy text completions (prompt= instead of messages=)
# ---------------------------------------------------------------------
def completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "openai.completions.create", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.TEXT_COMPLETION,
            operation_gen_ai=GenAIOperation.TEXT_COMPLETION,
        )
        prompt = kwargs.get("prompt")
        if prompt:
            safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": prompt}])
            safe_set(span, GenAIAttributes.PROMPT, prompt)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get("stream"):
            return SyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: None,
                on_finish=lambda: None,
            )
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "openai.completions.create", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.TEXT_COMPLETION,
            operation_gen_ai=GenAIOperation.TEXT_COMPLETION,
        )
        prompt = kwargs.get("prompt")
        if prompt:
            safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": prompt}])
            safe_set(span, GenAIAttributes.PROMPT, prompt)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get("stream"):
            return AsyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: None,
                on_finish=lambda: None,
            )
        set_chat_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# Embeddings — different response shape, own helpers.
# ---------------------------------------------------------------------
def embeddings_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "openai.embeddings.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_embeddings_request(span, kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        _set_embeddings_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_embeddings_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "openai.embeddings.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_embeddings_request(span, kwargs)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        _set_embeddings_response(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _set_embeddings_request(span: DisseqtSpan, kwargs: dict[str, Any]) -> None:
    model = kwargs.get("model", "")
    span.set_model_info(model, PROVIDER)
    span.set_operation(AgenticOperation.EMBEDDINGS)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.EMBEDDINGS)

    inp = kwargs.get("input")
    if isinstance(inp, str):
        safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": inp}])
    elif isinstance(inp, list) and inp and isinstance(inp[0], str):
        # Only materialize string inputs; token-array inputs would blow up the span.
        safe_set(
            span,
            AgenticAttributes.INPUT_MESSAGES,
            [{"role": "user", "content": s} for s in inp],
        )


def _set_embeddings_response(span: DisseqtSpan, response: Any) -> None:
    resp_model = read(response, "model")
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)
    usage = read(response, "usage")
    if usage is not None:
        prompt_tokens = read(usage, "prompt_tokens") or 0
        span.set_token_usage(prompt_tokens, 0)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, prompt_tokens)
