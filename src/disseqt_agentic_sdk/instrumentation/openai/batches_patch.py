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


def _make_batch_wrappers(
    instrumentor: OpenAIInstrumentor,
    *,
    span_name: str,
    op: str,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Build (sync, async) wrappers for a batch endpoint that follows the
    same shape: open span → emit provider tags → call → map response
    through ``from_openai`` + ``set_batch_attrs`` → close span.

    Rolls up the identical skeleton that used to be copy-pasted across
    all six batch functions (TP-2128 P4 #4.4). New endpoints (e.g. a
    hypothetical ``batches.list``) can be wired with two lines.
    """

    def sync_wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, span_name, SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), op)
        finally:
            scope.__exit__(None, None, None)
        return batch

    async def async_wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, span_name, SpanKind.MODEL_EXEC)
        span = scope.span
        _emit_provider_tags(span)
        try:
            batch = await wrapped(*args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        try:
            safe_call(set_batch_attrs, span, from_openai(batch), op)
        finally:
            scope.__exit__(None, None, None)
        return batch

    return sync_wrapper, async_wrapper


# ---------------------------------------------------------------------
# Create / Retrieve / Cancel — each pair produced by the factory above
# ---------------------------------------------------------------------
def batches_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    sync_w, _ = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.create",
        op=AgenticOperation.BATCH_CREATE,
    )
    return sync_w


def async_batches_create(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    _, async_w = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.create",
        op=AgenticOperation.BATCH_CREATE,
    )
    return async_w


def batches_retrieve(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    sync_w, _ = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.retrieve",
        op=AgenticOperation.BATCH_RETRIEVE,
    )
    return sync_w


def async_batches_retrieve(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    _, async_w = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.retrieve",
        op=AgenticOperation.BATCH_RETRIEVE,
    )
    return async_w


def batches_cancel(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    sync_w, _ = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.cancel",
        op=AgenticOperation.BATCH_CANCEL,
    )
    return sync_w


def async_batches_cancel(instrumentor: OpenAIInstrumentor) -> Callable[..., Any]:
    _, async_w = _make_batch_wrappers(
        instrumentor,
        span_name="openai.batches.cancel",
        op=AgenticOperation.BATCH_CANCEL,
    )
    return async_w
