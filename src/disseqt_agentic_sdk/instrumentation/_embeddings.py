"""
Cross-provider embeddings normalization.

Providers that expose embeddings differ in request kwargs and response
shape:

  * OpenAI / Mistral / LiteLLM
        request: ``input`` (str | list[str] | list[int]), plus optional
        ``dimensions``, ``encoding_format``, ``user``.
        response: ``data[i].embedding`` (list[float] | base64 str),
        ``usage.prompt_tokens`` / ``usage.total_tokens``.
  * Cohere
        request: ``texts=[...]``, ``model=...``, ``input_type=...``.
        response: ``embeddings[i]`` array, ``meta.billed_units.input_tokens``.
  * Google Gemini (``google-genai``)
        request: ``contents=str | list[str]``, ``model=...``.
        response: single ``EmbedContentResponse`` with ``embedding.values``,
        or ``embeddings=[...]`` for batch.

Same abstraction pattern as ``_tool_calls.py`` and ``_batches.py``:
canonical TypedDicts + per-provider adapters + one shared attribute
writer, so per-provider wrappers stay tiny.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from disseqt_agentic_sdk.instrumentation._utils import safe_set
from disseqt_agentic_sdk.semantics import AgenticAttributes, GenAIAttributes

if TYPE_CHECKING:
    from disseqt_agentic_sdk.span import DisseqtSpan


class CanonicalEmbeddingRequest(TypedDict, total=False):
    """
    Provider-agnostic view of an embeddings request.

    Fields the provider doesn't accept stay absent (total=False), and
    ``set_embedding_request_attrs`` skips missing keys.
    """

    model: str
    input_count: int  # 1 for a single string, N for a list
    dimensions_requested: int | None
    encoding_format: str | None
    user: str | None


class CanonicalEmbeddingResponse(TypedDict, total=False):
    """
    Provider-agnostic view of an embeddings response.

    ``dimensions_actual`` is measured from the first returned vector so
    callers can verify the model honored ``dimensions_requested``.
    """

    model: str
    count: int
    dimensions_actual: int | None
    input_tokens: int | None
    total_tokens: int | None


def _read(obj: Any, name: str) -> Any:
    """Attribute-or-key read tolerant of dicts and Pydantic models."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _count_inputs(inp: Any) -> int:
    """
    Count how many separate texts a request is embedding.

    OpenAI accepts ``str`` (one text) or ``list[str | int-tokens]`` (batch).
    We count the length of a list, treat everything else as 1.
    """
    if isinstance(inp, list):
        return len(inp)
    if inp is None:
        return 0
    return 1


# ---------------------------------------------------------------------
# OpenAI (and any provider that ships OpenAI-shaped embeddings, e.g.
# Mistral, LiteLLM proxy targets). Groq / Anthropic have no embeddings.
# ---------------------------------------------------------------------
def from_openai_request(kwargs: dict[str, Any]) -> CanonicalEmbeddingRequest:
    """Normalize openai-shape kwargs to CanonicalEmbeddingRequest."""
    return {
        "model": str(kwargs.get("model") or ""),
        "input_count": _count_inputs(kwargs.get("input")),
        "dimensions_requested": kwargs.get("dimensions"),
        "encoding_format": kwargs.get("encoding_format"),
        "user": kwargs.get("user"),
    }


def from_openai_response(response: Any) -> CanonicalEmbeddingResponse:
    """
    Normalize an ``openai.types.CreateEmbeddingResponse`` (or dict).

    ``dimensions_actual`` is None when the first entry uses base64 encoding
    (a string, not a list) — we don't decode it just for the length.
    """
    data = _read(response, "data") or []
    first_vec = _read(data[0], "embedding") if data else None
    dim_actual: int | None
    if isinstance(first_vec, list):
        dim_actual = len(first_vec)
    else:
        dim_actual = None

    usage = _read(response, "usage")
    input_tokens = _read(usage, "prompt_tokens") if usage is not None else None
    total_tokens = _read(usage, "total_tokens") if usage is not None else None
    return {
        "model": str(_read(response, "model") or ""),
        "count": len(data),
        "dimensions_actual": dim_actual,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
    }


# ---------------------------------------------------------------------
# Shared attribute writers — one place to add a new attribute later
# ---------------------------------------------------------------------
def set_embedding_request_attrs(span: DisseqtSpan, req: CanonicalEmbeddingRequest) -> None:
    """Emit canonical embedding request attributes onto ``span``."""
    safe_set(span, AgenticAttributes.EMBEDDINGS_INPUT_COUNT, req.get("input_count"))
    safe_set(span, AgenticAttributes.EMBEDDINGS_DIMENSIONS_REQUESTED, req.get("dimensions_requested"))
    safe_set(span, AgenticAttributes.EMBEDDINGS_ENCODING_FORMAT, req.get("encoding_format"))
    safe_set(span, AgenticAttributes.REQUEST_USER, req.get("user"))


def set_embedding_response_attrs(span: DisseqtSpan, resp: CanonicalEmbeddingResponse) -> None:
    """Emit canonical embedding response attributes onto ``span``."""
    safe_set(span, AgenticAttributes.RESPONSE_MODEL, resp.get("model"))
    safe_set(span, GenAIAttributes.RESPONSE_MODEL, resp.get("model"))
    safe_set(span, AgenticAttributes.EMBEDDINGS_COUNT, resp.get("count"))
    safe_set(span, AgenticAttributes.EMBEDDINGS_DIMENSIONS_ACTUAL, resp.get("dimensions_actual"))
    input_tokens = resp.get("input_tokens")
    total_tokens = resp.get("total_tokens")
    if input_tokens is not None:
        # Embeddings have no completion tokens — pass 0 for the second arg
        # so downstream tooling still gets a total.
        span.set_token_usage(input_tokens, 0)
        safe_set(span, GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
    if total_tokens is not None:
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, total_tokens)
    elif input_tokens is not None:
        safe_set(span, GenAIAttributes.USAGE_TOTAL_TOKENS, input_tokens)
