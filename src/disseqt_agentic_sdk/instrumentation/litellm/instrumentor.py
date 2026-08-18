"""
LiteLLM instrumentor.

LiteLLM is a router / proxy over many LLM providers, but exposes a single
OpenAI-shape surface: `litellm.completion(model=..., messages=...)` and
`litellm.acompletion(...)`. Response is a ModelResponse mimicking
`openai.types.chat.ChatCompletion`, so `_oai_compat` handles it directly.

Provider labelling: we tag the span with `agentic.provider.name="litellm"`
and `gen_ai.system="litellm"`; the downstream provider is derivable from
`gen_ai.response.model` (e.g. "gpt-4o-mini" or "claude-3-5-haiku-latest").

Note: `litellm.completion` is defined in `litellm.main`. Patch there rather
than on the `litellm` module attribute so wrapt sees the underlying function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._oai_compat import (
    ChatStreamAccumulator,
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


class LiteLLMInstrumentor(DisseqtInstrumentor):
    package_name = "litellm"
    min_version = "1.40.0"

    def _instrument(self) -> None:
        # litellm re-exports completion/acompletion from litellm.main; the
        # top-level module binding captures the original function at import
        # time. We wrap the module attribute itself so `litellm.completion(...)`
        # invokes our wrapper.
        self._wrap("litellm", "completion", _sync_completion(self))
        self._wrap("litellm", "acompletion", _async_completion(self))


def _sync_completion(instrumentor: LiteLLMInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "litellm.completion", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.LITELLM,
            system=GenAISystem.LITELLM,
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


def _async_completion(instrumentor: LiteLLMInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "litellm.acompletion", SpanKind.MODEL_EXEC)
        span = scope.span
        set_common_chat_request(
            span,
            kwargs,
            provider=AgenticProvider.LITELLM,
            system=GenAISystem.LITELLM,
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
