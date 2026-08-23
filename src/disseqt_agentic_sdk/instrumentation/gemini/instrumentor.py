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

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.enums import SpanKind
from disseqt_agentic_sdk.instrumentation._kwargs import (
    KW_CONFIG,
    KW_CONTENTS,
    KW_MODEL,
    KW_TOOLS,
)
from disseqt_agentic_sdk.instrumentation._oai_compat import read
from disseqt_agentic_sdk.instrumentation._stream import AsyncStreamWrapper, SyncStreamWrapper
from disseqt_agentic_sdk.instrumentation._tool_calls import from_gemini as _tc_from_gemini
from disseqt_agentic_sdk.instrumentation._tool_result import (
    _notify_planned_tool_calls,
)
from disseqt_agentic_sdk.instrumentation._utils import (
    open_llm_span,
    safe_call,
    safe_set,
    set_messages_if_capturing,
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
    model = kwargs.get(KW_MODEL, "")
    span.set_model_info(model, PROVIDER)
    span.set_operation(AgenticOperation.GENERATE_CONTENT)
    safe_set(span, GenAIAttributes.SYSTEM, SYSTEM)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, GenAIOperation.GENERATE_CONTENT)

    # Gemini bundles generation params in a `config=GenerateContentConfig(...)` object.
    config = kwargs.get(KW_CONFIG)
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

    contents = kwargs.get(KW_CONTENTS)
    normalized = _normalize_contents(contents)
    if normalized:
        set_messages_if_capturing(span, input_messages=normalized)
        safe_set(span, GenAIAttributes.PROMPT, normalized)

    # google-genai callers pass tools either at top level or inside `config`.
    tools = kwargs.get(KW_TOOLS)
    if tools is None and config is not None:
        tools = read(config, "tools")
    if tools:
        try:
            tools_json = json.dumps(tools, default=str)
        except (TypeError, ValueError):
            tools_json = str(tools)
        safe_set(span, AgenticAttributes.REQUEST_TOOLS, tools_json)
        safe_set(span, GenAIAttributes.REQUEST_TOOLS, tools_json)


def _normalize_contents(contents: Any) -> list[dict[str, Any]]:
    """
    Coerce Gemini contents into ``[{role, content}, ...]`` dicts.

    google-genai's ``contents`` argument accepts several documented shapes:

      * ``str`` — single user turn.
      * ``Content`` — has ``.parts`` and optional ``.role``.
      * bare ``Part`` / ``File`` — has ``.text`` directly, NOT wrapped in
        ``Content``. This is a first-class, documented input shape.
      * ``list`` of any of the above (mixed).

    Before TP-2128 P1 #1.7, any non-``str``/non-``list`` value was
    assumed to be ``Content``-shaped: ``read(entry, "parts")`` returned
    ``None`` for a bare ``Part``, joining produced ``""``, and the
    span recorded an empty prompt even though real text was sent.
    """
    if contents is None:
        return []
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        return [_normalize_content_entry(e) for e in contents]
    # Single object — Content, bare Part, File, or something similar.
    return [_normalize_content_entry(contents)]


def _normalize_content_entry(entry: Any) -> dict[str, Any]:
    """Turn one Gemini content entry into ``{role, content}``."""
    if isinstance(entry, str):
        return {"role": "user", "content": entry}
    # Content-shaped: has .parts (possibly empty), possibly .role.
    parts = read(entry, "parts")
    if parts is not None:
        text = "".join(read(p, "text") or "" for p in parts)
        return {"role": read(entry, "role") or "user", "content": text}
    # Bare Part / File: has .text directly.
    text = read(entry, "text")
    if isinstance(text, str) and text:
        return {"role": "user", "content": text}
    # Nothing usable — record the string form so we don't lose visibility.
    return {"role": "user", "content": str(entry)}


def _set_response_attrs(span: DisseqtSpan, response: Any) -> None:
    resp_id = read(response, "response_id")
    resp_model = read(response, "model_version")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    usage = read(response, "usage_metadata")
    if usage is not None:
        # NOTE: the real google-genai `GenerateContentResponseUsageMetadata`
        # exposes `candidates_token_count`, not `response_token_count`
        # (which only exists on the unrelated Live-API type). Reading the
        # wrong field silently zeroed output-token telemetry for every real
        # Gemini call — the earlier bare-MagicMock test invented the field
        # so the bug went unnoticed. See TP-2128 P0 #0.3.
        prompt_tokens = read(usage, "prompt_token_count") or 0
        candidates_tokens = read(usage, "candidates_token_count") or 0
        total = read(usage, "total_token_count") or (prompt_tokens + candidates_tokens)
        span.set_token_usage(prompt_tokens, candidates_tokens)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, candidates_tokens)
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
            set_messages_if_capturing(span, output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)

        parts = _candidate_parts(first)
        # Pass the response_id so synthesized call ids stay unique across
        # separate responses in the same agent_span. See TP-2128 P1 #1.3.
        tool_calls = _tc_from_gemini(parts, response_id=resp_id)
        if tool_calls:
            safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
            _notify_planned_tool_calls(tool_calls)
            safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
            tc0 = tool_calls[0]
            safe_set(span, AgenticAttributes.TOOL_NAME, tc0["name"])
            safe_set(span, GenAIAttributes.TOOL_NAME, tc0["name"])
            safe_set(span, AgenticAttributes.TOOL_CALL_ID, tc0["id"])
            safe_set(span, GenAIAttributes.TOOL_CALL_ID, tc0["id"])
            safe_set(span, AgenticAttributes.TOOL_ARGS, tc0["arguments"])
            safe_set(span, GenAIAttributes.TOOL_ARGS, tc0["arguments"])


def _candidate_parts(candidate: Any) -> list[Any]:
    content = read(candidate, "content")
    if content is None:
        return []
    return read(content, "parts") or []


def _extract_candidate_text(candidate: Any) -> str:
    content = read(candidate, "content")
    if content is None:
        return ""
    parts = read(content, "parts") or []
    out: list[str] = []
    for p in parts:
        text = read(p, "text")
        if isinstance(text, str) and text:
            out.append(text)
    return "".join(out)


# ---------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------
def _sync_generate(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(instrumentor.client, "gemini.generate_content", SpanKind.MODEL_EXEC)
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


def _async_generate(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(instrumentor.client, "gemini.generate_content", SpanKind.MODEL_EXEC)
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


class _StreamAccumulator:
    """Collects Gemini stream chunks (each chunk is a GenerateContentResponse)."""

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.finish_reason: str | None = None
        self.model_version: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.response_tokens: int | None = None
        # Keyed by position within `candidate.content.parts` so:
        #   * repeated chunks carrying the same function_call at the same
        #     position overwrite each other cleanly (last-write-wins for
        #     the latest snapshot);
        #   * two legitimately-parallel calls to the same tool with the
        #     same arguments (e.g. two ``roll_die()``, two zero-arg calls,
        #     two identical shard queries) at positions 0 and 1 stay
        #     distinct instead of colliding under a content-hash key.
        # Fix for TP-2128 P1 #1.6.
        self._tool_parts_by_position: dict[int, Any] = {}

    def absorb(self, chunk: Any) -> None:
        self.model_version = self.model_version or read(chunk, "model_version")
        self.response_id = self.response_id or read(chunk, "response_id")

        usage = read(chunk, "usage_metadata")
        if usage is not None:
            pt = read(usage, "prompt_token_count")
            # Real field is `candidates_token_count`; see the note on the
            # non-streaming path above.
            rt = read(usage, "candidates_token_count")
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
            for position, part in enumerate(_candidate_parts(first)):
                fc = read(part, "function_call")
                if fc is None:
                    continue
                # Overwrite by position: later chunks carry newer snapshots
                # of the same call, and distinct calls at different
                # positions each get their own slot.
                self._tool_parts_by_position[position] = part

    def finalize(self, span: DisseqtSpan) -> None:
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": "model", "content": text}]
            set_messages_if_capturing(span, output_messages=msgs)
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

        # Emit in position order — matches the order the model produced.
        ordered_parts = [
            self._tool_parts_by_position[i] for i in sorted(self._tool_parts_by_position)
        ]
        tool_calls = _tc_from_gemini(ordered_parts, response_id=self.response_id)
        if tool_calls:
            safe_set(span, AgenticAttributes.TOOL_CALLS, tool_calls)
            _notify_planned_tool_calls(tool_calls)
            safe_set(span, GenAIAttributes.TOOL_CALLS, tool_calls)
            tc0 = tool_calls[0]
            safe_set(span, AgenticAttributes.TOOL_NAME, tc0["name"])
            safe_set(span, GenAIAttributes.TOOL_NAME, tc0["name"])
            safe_set(span, AgenticAttributes.TOOL_CALL_ID, tc0["id"])
            safe_set(span, GenAIAttributes.TOOL_CALL_ID, tc0["id"])
            safe_set(span, AgenticAttributes.TOOL_ARGS, tc0["arguments"])
            safe_set(span, GenAIAttributes.TOOL_ARGS, tc0["arguments"])


def _sync_stream(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    def wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        scope = open_llm_span(
            instrumentor.client, "gemini.generate_content_stream", SpanKind.MODEL_EXEC
        )
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
            on_chunk=lambda chunk: state.absorb(chunk),
            on_finish=lambda: state.finalize(span),
        )

    return wrapper


def _async_stream(instrumentor: GeminiInstrumentor) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        scope = open_llm_span(
            instrumentor.client, "gemini.generate_content_stream", SpanKind.MODEL_EXEC
        )
        span = scope.span
        safe_call(_set_request_attrs, span, kwargs)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exc:
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
