"""
Google Gemini SDK instrumentor.

Targets the current `google-genai` SDK (the legacy `google-generativeai`
package is end-of-life per Google's own deprecation notice). Patches:
  * models.Models.generate_content          (sync, non-streaming)
  * models.Models.generate_content_stream   (sync, streaming)
  * models.AsyncModels.generate_content     (async, non-streaming)
  * models.AsyncModels.generate_content_stream (async, streaming)

Gemini's shape is unique:
  * request kwargs: `model=`, `contents=[...]`, `config=GenerateContentConfig(...)`.
  * response: `candidates[i].content.parts[j].text`; usage on
    `usage_metadata.prompt_token_count` / `.response_token_count`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._oai_compat import read
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._utils import (
    open_llm_span,
    safe_set,
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


PROVIDER = AgenticProvider.GOOGLE
SYSTEM = GenAISystem.GEMINI


class GeminiInstrumentor(DisseqtInstrumentor):
    package_name = "google-genai"
    min_version = "1.0.0"

    def _instrument(self) -> None:
        self._wrap("google.genai.models", "Models.generate_content", _sync_generate(self))
        self._wrap("google.genai.models", "Models.generate_content_stream", _sync_stream(self))
        self._wrap("google.genai.models", "AsyncModels.generate_content", _async_generate(self))
        self._wrap(
            "google.genai.models", "AsyncModels.generate_content_stream", _async_stream(self)
        )


# ---------------------------------------------------------------------
# Attribute writers
# ---------------------------------------------------------------------
def _set_request_attrs(span: DisseqtSpan, kwargs: dict[str, Any]) -> None:
    model = kwargs.get("model", "")
    span.set_model_info(model, PROVIDER)
    span.set_operation(AgenticOperation.GENERATE_CONTENT)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.GENERATE_CONTENT)

    # Gemini bundles generation params in a `config=GenerateContentConfig(...)` object.
    config = kwargs.get("config")
    for cfg_key, agentic_key, gen_ai_key in (
        ("temperature", AgenticAttributes.REQUEST_TEMPERATURE, GenAIAttributes.REQUEST_TEMPERATURE),
        (
            "max_output_tokens",
            AgenticAttributes.REQUEST_MAX_TOKENS,
            GenAIAttributes.REQUEST_MAX_TOKENS,
        ),
        ("top_p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        ("top_k", AgenticAttributes.REQUEST_TOP_K, GenAIAttributes.REQUEST_TOP_K),
    ):
        val = read(config, cfg_key) if config is not None else None
        if val is not None:
            safe_set(span, agentic_key, val)
            safe_set(span, gen_ai_key, val)

    system_instruction = read(config, "system_instruction") if config is not None else None
    if system_instruction:
        safe_set(span, AgenticAttributes.SYSTEM_INSTRUCTIONS, str(system_instruction))

    contents = kwargs.get("contents")
    normalized = _normalize_contents(contents)
    if normalized:
        span.set_messages(input_messages=normalized)
        safe_set(span, GenAIAttributes.PROMPT, normalized)


def _normalize_contents(contents: Any) -> list[dict[str, Any]]:
    """Coerce Gemini contents (str | Content | list[...]) into role/content dicts."""
    if contents is None:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        # Delegate to serialize_messages for dict-shaped entries; strings become user turns.
        out: list[dict[str, Any]] = []
        for entry in contents:
            if isinstance(entry, str):
                out.append({"role": "user", "content": entry})
            else:
                parts = read(entry, "parts") or []
                text = "".join(read(p, "text") or "" for p in parts)
                out.append({"role": read(entry, "role") or "user", "content": text})
        return out
    # Single Content object.
    parts = read(contents, "parts") or []
    text = "".join(read(p, "text") or "" for p in parts)
    return [{"role": read(contents, "role") or "user", "content": text}]


def _set_response_attrs(span: DisseqtSpan, response: Any) -> None:
    resp_id = read(response, "response_id")
    resp_model = read(response, "model_version")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    usage = read(response, "usage_metadata")
    if usage is not None:
        prompt_tokens = read(usage, "prompt_token_count") or 0
        response_tokens = read(usage, "response_token_count") or 0
        total = read(usage, "total_token_count") or (prompt_tokens + response_tokens)
        span.set_token_usage(prompt_tokens, response_tokens)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, response_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, total)

    candidates = read(response, "candidates") or []
    if candidates:
        first = candidates[0]
        finish_reason = read(first, "finish_reason")
        if finish_reason is not None:
            fr_str = str(finish_reason)
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, fr_str)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [fr_str])
        text = _extract_candidate_text(first)
        if text:
            msgs = [{"role": "model", "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)


def _extract_candidate_text(candidate: Any) -> str:
    content = read(candidate, "content")
    if content is None:
        return ""
    parts = read(content, "parts") or []
    return "".join(read(p, "text") or "" for p in parts)


# ---------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------
def _sync_generate(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "gemini.generate_content", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_request_attrs(span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        _set_response_attrs(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


def _async_generate(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(instrumentor.client, "gemini.generate_content", SpanKind.MODEL_EXEC)
        span = scope.span
        _set_request_attrs(span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        _set_response_attrs(span, result)
        scope.__exit__(None, None, None)
        return result

    return wrapper


class _StreamAccumulator:
    """Collects Gemini stream chunks (each chunk is a GenerateContentResponse)."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.finish_reason: str | None = None
        self.model_version: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.response_tokens: int | None = None

    def absorb(self, chunk: Any) -> None:
        self.model_version = self.model_version or read(chunk, "model_version")
        self.response_id = self.response_id or read(chunk, "response_id")

        usage = read(chunk, "usage_metadata")
        if usage is not None:
            pt = read(usage, "prompt_token_count")
            rt = read(usage, "response_token_count")
            if pt is not None:
                self.prompt_tokens = pt
            if rt is not None:
                self.response_tokens = rt

        candidates = read(chunk, "candidates") or []
        if candidates:
            first = candidates[0]
            finish_reason = read(first, "finish_reason")
            if finish_reason is not None:
                self.finish_reason = str(finish_reason)
            text = _extract_candidate_text(first)
            if text:
                self.buffer.append(text)

    def finalize(self, span: DisseqtSpan) -> None:
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": "model", "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)
        if self.model_version:
            safe_set(span, AgenticAttributes.RESPONSE_MODEL, self.model_version)
            safe_set(span, GenAIAttributes.RESPONSE_MODEL, self.model_version)
        if self.response_id:
            safe_set(span, AgenticAttributes.RESPONSE_ID, self.response_id)
            safe_set(span, GenAIAttributes.RESPONSE_ID, self.response_id)
        if self.finish_reason:
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, self.finish_reason)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [self.finish_reason])
        if self.prompt_tokens is not None and self.response_tokens is not None:
            span.set_token_usage(self.prompt_tokens, self.response_tokens)
            safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, self.prompt_tokens)
            safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, self.response_tokens)
            safe_set(
                span,
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self.prompt_tokens + self.response_tokens,
            )


def _sync_stream(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(
            instrumentor.client, "gemini.generate_content_stream", SpanKind.MODEL_EXEC
        )
        span = scope.span
        _set_request_attrs(span, kwargs)
        try:
            result = wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = _StreamAccumulator()
        return SyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda chunk: state.absorb(chunk),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


def _async_stream(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    async def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]
        scope = open_llm_span(
            instrumentor.client, "gemini.generate_content_stream", SpanKind.MODEL_EXEC
        )
        span = scope.span
        _set_request_attrs(span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise
        state = _StreamAccumulator()
        return AsyncStreamWrapper(
            stream=result,
            scope=scope,
            on_chunk=lambda chunk: state.absorb(chunk),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper
