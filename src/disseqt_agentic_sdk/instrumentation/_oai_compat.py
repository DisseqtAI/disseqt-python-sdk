"""
Shared helpers for OpenAI-compatible provider SDKs.

OpenAI, Groq, Mistral, LiteLLM (and many others) return the same
response shape: `choices[0].message.content`, `usage.prompt_tokens`,
`usage.completion_tokens`. Streaming events all carry deltas under
`choices[0].delta`. Instrumentors for those SDKs share this module
instead of copy-pasting 200 lines of attribute-mapping logic.

Provider-specific instrumentors just pass the right `provider`
(AgenticProvider.*) and `system` (GenAISystem.*) values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from disseqt_agentic_sdk.instrumentation._utils import safe_set, serialize_messages
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes

if TYPE_CHECKING:
    from disseqt_agentic_sdk.span import DisseqtSpan


# ---------------------------------------------------------------------
# Request-side
# ---------------------------------------------------------------------
def set_common_chat_request(
    span: DisseqtSpan,
    kwargs: dict[str, Any],
    *,
    provider: str,
    system: str,
    operation_agentic: str,
    operation_gen_ai: str,
) -> None:
    """
    Populate the span with model + common request params before the wrapped
    call runs. Handles the params every OpenAI-shaped SDK accepts:
    temperature, max_tokens, top_p, frequency_penalty, presence_penalty,
    plus `stream` and `messages`.
    """
    model = kwargs.get("model", "")
    span.set_model_info(model, provider)
    span.set_operation(operation_agentic)

    safe_set(span, GenAIAttributes.SYSTEM, system)
    safe_set(span, GenAIAttributes.REQUEST_MODEL, model)
    safe_set(span, GenAIAttributes.OPERATION_NAME, operation_gen_ai)

    for key, agentic_key, gen_ai_key in (
        ("temperature", AgenticAttributes.REQUEST_TEMPERATURE, GenAIAttributes.REQUEST_TEMPERATURE),
        ("max_tokens", AgenticAttributes.REQUEST_MAX_TOKENS, GenAIAttributes.REQUEST_MAX_TOKENS),
        ("top_p", AgenticAttributes.REQUEST_TOP_P, GenAIAttributes.REQUEST_TOP_P),
        (
            "frequency_penalty",
            AgenticAttributes.REQUEST_FREQUENCY_PENALTY,
            GenAIAttributes.REQUEST_FREQUENCY_PENALTY,
        ),
        (
            "presence_penalty",
            AgenticAttributes.REQUEST_PRESENCE_PENALTY,
            GenAIAttributes.REQUEST_PRESENCE_PENALTY,
        ),
    ):
        val = kwargs.get(key)
        if val is not None:
            safe_set(span, agentic_key, val)
            safe_set(span, gen_ai_key, val)

    if "stream" in kwargs:
        safe_set(span, GenAIAttributes.REQUEST_IS_STREAM, bool(kwargs["stream"]))

    messages = serialize_messages(kwargs.get("messages"))
    if messages:
        span.set_messages(input_messages=messages)
        safe_set(span, GenAIAttributes.PROMPT, messages)


# ---------------------------------------------------------------------
# Response-side (non-streaming)
# ---------------------------------------------------------------------
def set_chat_response(span: DisseqtSpan, response: Any) -> None:
    """
    Populate response id/model/tokens/messages/finish_reason from an
    OpenAI-shaped ChatCompletion. Works for OpenAI, Groq, Mistral,
    LiteLLM, and any provider matching that structure.
    """
    resp_id = read(response, "id")
    resp_model = read(response, "model")
    safe_set(span, AgenticAttributes.RESPONSE_ID, resp_id)
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp_model)
    safe_set(span, GenAIAttributes.RESPONSE_ID, resp_id)
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp_model)

    usage = read(response, "usage")
    if usage is not None:
        prompt_tokens = read(usage, "prompt_tokens") or read(usage, "input_tokens") or 0
        completion_tokens = read(usage, "completion_tokens") or read(usage, "output_tokens") or 0
        span.set_token_usage(prompt_tokens, completion_tokens)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, prompt_tokens)
        safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, completion_tokens)
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, prompt_tokens + completion_tokens)

    choices = read(response, "choices") or []
    output_messages: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    for choice in choices:
        message = read(choice, "message")
        if message is not None:
            role = read(message, "role") or "assistant"
            content = read(message, "content") or ""
            output_messages.append({"role": role, "content": content})
        finish_reason = read(choice, "finish_reason")
        if finish_reason:
            finish_reasons.append(finish_reason)

    if output_messages:
        span.set_messages(output_messages=output_messages)
        safe_set(span, GenAIAttributes.COMPLETION, output_messages)
    if finish_reasons:
        safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, finish_reasons[0])
        safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reasons)


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------
class ChatStreamAccumulator:
    """
    Accumulates streamed OpenAI-shaped chat chunks into text + token counts.
    Providers hand each chunk to `absorb(chunk)`; when the stream ends,
    `finalize(span)` writes the aggregated attributes.
    """

    def __init__(self) -> None:
        self.buffer: list[str] = []
        self.role: str = "assistant"
        self.finish_reason: str | None = None
        self.model: str | None = None
        self.response_id: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    def absorb(self, chunk: Any) -> None:
        self.model = self.model or read(chunk, "model")
        self.response_id = self.response_id or read(chunk, "id")

        # `stream_options={"include_usage": True}` sends usage on a final chunk.
        usage = read(chunk, "usage")
        if usage is not None:
            self.prompt_tokens = (
                read(usage, "prompt_tokens") or read(usage, "input_tokens") or self.prompt_tokens
            )
            self.completion_tokens = (
                read(usage, "completion_tokens")
                or read(usage, "output_tokens")
                or self.completion_tokens
            )

        for choice in read(chunk, "choices") or []:
            delta = read(choice, "delta")
            if delta is not None:
                role = read(delta, "role")
                if role:
                    self.role = role
                content = read(delta, "content")
                if content:
                    self.buffer.append(content)
            finish_reason = read(choice, "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason

    def finalize(self, span: DisseqtSpan) -> None:
        text = "".join(self.buffer)
        if text:
            msgs = [{"role": self.role, "content": text}]
            span.set_messages(output_messages=msgs)
            safe_set(span, GenAIAttributes.COMPLETION, msgs)
        if self.model:
            safe_set(span, AgenticAttributes.RESPONSE_MODEL, self.model)
            safe_set(span, GenAIAttributes.RESPONSE_MODEL, self.model)
        if self.response_id:
            safe_set(span, AgenticAttributes.RESPONSE_ID, self.response_id)
            safe_set(span, GenAIAttributes.RESPONSE_ID, self.response_id)
        if self.finish_reason:
            safe_set(span, AgenticAttributes.RESPONSE_FINISH_REASON, self.finish_reason)
            safe_set(span, GenAIAttributes.RESPONSE_FINISH_REASONS, [self.finish_reason])
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            span.set_token_usage(self.prompt_tokens, self.completion_tokens)
            safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, self.prompt_tokens)
            safe_set(span, GenAIAttributes.USAGE_OUTPUT_TOKENS, self.completion_tokens)
            safe_set(
                span,
                GenAIAttributes.USAGE_TOTAL_TOKENS,
                self.prompt_tokens + self.completion_tokens,
            )


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def read(obj: Any, name: str) -> Any:
    """Attribute-or-key read; tolerates both Pydantic models and dicts."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
