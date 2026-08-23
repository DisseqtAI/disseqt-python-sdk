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
from disseqt_agentic_sdk.instrumentation._embeddings import (
    from_openai_request as _emb_from_openai_request,
)
from disseqt_agentic_sdk.instrumentation._embeddings import (
    from_openai_response as _emb_from_openai_response,
)
from disseqt_agentic_sdk.instrumentation._embeddings import (
    set_embedding_request_attrs,
    set_embedding_response_attrs,
)
from disseqt_agentic_sdk.instrumentation._kwargs import KW_INPUT, KW_MODEL, KW_PROMPT, KW_STREAM
from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
    set_chat_response,
    set_common_chat_request,
)
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span, safe_call, safe_set
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
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(
            instrumentor.client, "openai.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        safe_call(
            set_common_chat_request,
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


def async_chat_completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(
            instrumentor.client, "openai.chat.completions.create", SpanKind.MODEL_EXEC
        )
        span = scope.span
        safe_call(
            set_common_chat_request,
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


# ---------------------------------------------------------------------
# Legacy text completions (prompt= instead of messages=)
# ---------------------------------------------------------------------
def completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.completions.create", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(
            set_common_chat_request,
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.TEXT_COMPLETION,
            operation_gen_ai=GenAIOperation.TEXT_COMPLETION,
        )
        prompt = kwargs.get(KW_PROMPT)
        if prompt:
            safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": prompt}])
            safe_set(span, GenAIAttributes.PROMPT, prompt)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get(KW_STREAM):
            return SyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: None,
                on_finish=lambda: None,
            )
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_completions_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.completions.create", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(
            set_common_chat_request,
            span,
            kwargs,
            provider=PROVIDER,
            system=SYSTEM,
            operation_agentic=AgenticOperation.TEXT_COMPLETION,
            operation_gen_ai=GenAIOperation.TEXT_COMPLETION,
        )
        prompt = kwargs.get(KW_PROMPT)
        if prompt:
            safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": prompt}])
            safe_set(span, GenAIAttributes.PROMPT, prompt)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if kwargs.get(KW_STREAM):
            return AsyncStreamWrapper(
                stream=result,
                scope=scope,
                on_chunk=lambda chunk: None,
                on_finish=lambda: None,
            )
        safe_call(set_chat_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


# ---------------------------------------------------------------------
# Embeddings — different response shape, own helpers.
# ---------------------------------------------------------------------
def embeddings_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.embeddings.create", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_embeddings_request, span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        safe_call(_set_embeddings_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def async_embeddings_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.embeddings.create", SpanKind.MODEL_EXEC)
        span = scope.span
        safe_call(_set_embeddings_request, span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        safe_call(_set_embeddings_response, span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _set_embeddings_request(span: DisseqtSpan, kwargs: dict[str, Any]) -> None:
    """
    Emit request-side attributes for an OpenAI embeddings call.

    Provider tagging, model, and operation type are set here; embedding-
    specific fields (dimensions_requested, encoding_format, user,
    input_count) flow through the canonical adapter so the same shape is
    reused when other providers (Mistral, LiteLLM, Cohere, Gemini) are
    added later.
    """
    model = kwargs.get(KW_MODEL, "")
    span.set_model_info(model, PROVIDER)
    span.set_operation(AgenticOperation.EMBEDDINGS)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.EMBEDDINGS)

    inp = kwargs.get(KW_INPUT)
    if isinstance(inp, str):
        safe_set(span, AgenticAttributes.INPUT_MESSAGES, [{"role": "user", "content": inp}])
    elif isinstance(inp, list) and inp and isinstance(inp[0], str):
        # Only materialize string inputs; token-array inputs would blow up the span.
        safe_set(
            span,
            AgenticAttributes.INPUT_MESSAGES,
            [{"role": "user", "content": s} for s in inp],
        )

    set_embedding_request_attrs(span, _emb_from_openai_request(kwargs))


def _set_embeddings_response(span: DisseqtSpan, response: Any) -> None:
    """
    Emit response-side attributes for an OpenAI embeddings call.

    Model/token bookkeeping + embedding-specific fields (count,
    dimensions_actual) go through the canonical adapter.
    """
    set_embedding_response_attrs(span, _emb_from_openai_response(response))
