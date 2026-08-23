"""
OpenAI Batch API wrappers.

Instruments ``client.batches.create`` / ``.retrieve`` / ``.cancel`` (plus
async variants). Each call becomes its own MODEL_EXEC span; consumers
correlate them via the shared ``agentic.batch.id`` attribute.

The response-shape mapping lives in ``_batches.from_openai``; attribute
writing lives in ``_batches.set_batch_attrs``. This module is only wrapper
skeletons + provider tagging, so adding a new provider (Anthropic,
Mistral) means writing an equivalent skeleton — no attribute code is
duplicated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._batches import from_openai, set_batch_attrs
from disseqt_agentic_sdk.instrumentation._utils import open_llm_span, safe_call, safe_set
from disseqt_agentic_sdk.semantics import (
    AgenticAttributes,
    AgenticOperation,
    AgenticProvider,
    GenAIAttributes,
    GenAISystem,
)

if TYPE_CHECKING:
    from disseqt_agentic_sdk.instrumentation.openai.instrumentor import OpenAIInstrumentor


PROVIDER = AgenticProvider.OPENAI
SYSTEM = GenAISystem.OPENAI


def _emit_provider_tags(span: Any) -> None:
    """Provider/system tags that are cheap constants on every batch span."""
    safe_set(span, AgenticAttributes.PROVIDER_NAME, PROVIDER)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)


# ---------------------------------------------------------------------
# Create — sync + async
# ---------------------------------------------------------------------
def batches_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_CREATE)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper


def async_batches_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.create", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_CREATE)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper


# ---------------------------------------------------------------------
# Retrieve — sync + async
# ---------------------------------------------------------------------
def batches_retrieve(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.retrieve", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_RETRIEVE)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper


def async_batches_retrieve(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.retrieve", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_RETRIEVE)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper


# ---------------------------------------------------------------------
# Cancel — sync + async
# ---------------------------------------------------------------------
def batches_cancel(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.cancel", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_CANCEL)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper


def async_batches_cancel(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "openai.batches.cancel", SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), AgenticOperation.BATCH_CANCEL)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return wrapper
