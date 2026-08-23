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
from types import SimpleNamespace
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
        # Iterate every candidate so callers using candidate_count>1 see
        # the full picture (TP-2128 P2 #2.8). RESPONSE_FINISH_REASONS is
        # already a list per the OTEL GenAI spec, so we naturally append
        # one entry per candidate. The single-value convenience attrs and
        # tool_calls stay on candidate 0 to match the OpenAI n>1 policy
        # (item 2.7) — mixing tool_calls across candidates would
        # misattribute which choice the tool call belongs to.
        first = candidates[0]

        finish_reasons: list[str] = []
        output_msgs: list[dict[str, Any]] = []
        for cand in candidates:
            fr = read(cand, "finish_reason")
            if fr is not None:
                finish_reasons.append(str(fr))
            cand_text = _extract_candidate_text(cand)
            if cand_text:
                output_msgs.append({"role": "model", "content": cand_text})

        if finish_reasons:
            # RESPONSE_FINISH_REASON (singular) keeps the choice-0
            # value for backward compatibility with the log-search
            # queries built against it.
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, finish_reasons[0])
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reasons)

        if output_msgs:
            set_messages_if_capturing(span, output_messages=output_msgs)
            safe_set(span, GenAIAttributes.COMPLETION, output_msgs)

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


def _assemble_partial_args(fragments: list[Any]) -> dict[str, Any]:
    """
    Reduce a list of Vertex ``PartialArg`` fragments into an args dict.

    Each fragment carries a ``json_path`` (RFC 9535) plus one of
    ``bool_value`` / ``string_value`` / ``number_value`` / ``null_value``.
    We flatten only the common shallow path form (``$.field`` or
    ``field``); nested / indexed paths are stored verbatim under their
    json_path string so nothing is lost even if we can't perfectly
    reconstruct the shape.
    """
    args: dict[str, Any] = {}
    for frag in fragments or []:
        path = read(frag, "json_path")
        if not isinstance(path, str) or not path:
            continue
        # Extract the typed value; only one of the *_value fields is set.
        if read(frag, "string_value") is not None:
            value: Any = read(frag, "string_value")
        elif read(frag, "number_value") is not None:
            value = read(frag, "number_value")
        elif read(frag, "bool_value") is not None:
            value = read(frag, "bool_value")
        elif read(frag, "null_value") is not None:
            value = None
        else:
            continue

        # Flatten simple $.field paths into args[field]; leave complex
        # paths (nested / indexed) keyed by the raw json_path so
        # dashboards still see the data even if the shape isn't perfect.
        key = path[2:] if path.startswith("$.") else path
        if "." not in key and "[" not in key:
            args[key] = value
        else:
            args[path] = value
    return args


def _synthesize_part_from_slot(slot: dict[str, Any]) -> Any | None:
    """
    Build a synthetic Part-shaped object (SimpleNamespace) from an
    accumulated tool slot so ``from_gemini`` can consume it uniformly.

    Prefers fully-formed ``args`` when the chunks carried them; falls
    back to assembling ``partial_args`` fragments (Vertex
    ``stream_function_call_arguments=True``). Returns None if there's
    nothing meaningful to emit (no name).
    """
    name = slot.get("name")
    if not name:
        return None
    args = slot.get("args")
    if not args:
        assembled = _assemble_partial_args(slot.get("partial_args") or [])
        if assembled:
            args = assembled
    return SimpleNamespace(
        function_call=SimpleNamespace(
            id=slot.get("id"),
            name=name,
            args=args or {},
        )
    )


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
        # Per-position accumulator for streamed function calls.
        #
        # Position (in ``candidate.content.parts``) keeps two identical
        # parallel calls distinct — e.g. two ``roll_die()``, two zero-arg
        # calls at positions 0 and 1 — instead of collapsing them under a
        # content-hash key (TP-2128 P1 #1.6).
        #
        # Each slot carries:
        #   * id, name — latest non-None seen for this position
        #   * args — latest non-None fully-formed args dict
        #   * partial_args — accumulated PartialArg fragments across
        #     chunks; used when ``stream_function_call_arguments=True``
        #     splits a single FunctionCall's args across chunks
        #     (TP-2128 P2 #2.10). Before the fix, only ``args`` was
        #     inspected — streamed args left the tool call with
        #     ``arguments: '{}'``.
        self._tool_slots: dict[int, dict[str, Any]] = {}

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
                slot = self._tool_slots.setdefault(
                    position,
                    {"id": None, "name": None, "args": None, "partial_args": []},
                )
                real_id = read(fc, "id")
                if isinstance(real_id, str) and real_id:
                    slot["id"] = real_id
                name = read(fc, "name")
                if isinstance(name, str) and name:
                    slot["name"] = name
                args = read(fc, "args")
                if isinstance(args, dict) and args:
                    slot["args"] = args
                # ``partial_args`` is a list of PartialArg fragments (Vertex
                # streaming). Extend rather than replace so fragments from
                # earlier chunks are preserved when will_continue=True.
                partial = read(fc, "partial_args")
                if partial:
                    try:
                        slot["partial_args"].extend(partial)
                    except TypeError:
                        # Non-iterable — store as single fragment.
                        slot["partial_args"].append(partial)

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
            _synthesize_part_from_slot(self._tool_slots[i]) for i in sorted(self._tool_slots)
        ]
        # Drop synthetic parts that never carried a name — nothing to
        # emit and _tc_from_gemini would skip them anyway.
        ordered_parts = [p for p in ordered_parts if p is not None]
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
