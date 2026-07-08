"""Agentic behaviour request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _AgenticFieldsMixin, _LLMTextFieldsMixin


@dataclass(slots=True)
class AgenticBehaviourRequest(_AgenticFieldsMixin, _LLMTextFieldsMixin):
    """Request model for agentic behaviour validators.

    Carries the agentic arrays 1:1 AND the optional LLM text fields
    (``prompt``/``context``/``response``). A realtime policy can mix
    agentic validators (``tool_call_accuracy``, ``topic_adherence``, …)
    with text validators (``factual_consistency``, ``data_leakage``, …)
    in one bundle — ``validate(request, policies=[...])`` sends a single
    input bag, so the carrier must be able to hold the union of every
    field the policy's validators read. Text fields left as ``None`` are
    omitted from the wire exactly like on the text-only request models.
    """

    # All fields optional in SDK; you will usually provide them
    conversation_history: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    agent_responses: list[str] | None = None
    reference_data: dict[str, Any] | None = None
    # LLM text fields — same rename rules as InputValidationRequest et al.
    # (prompt → llm_input_query, context → llm_input_context,
    # response → llm_output).
    prompt: str | None = None
    context: str | None = None
    response: str | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload.

        Agentic fields map 1:1; text fields are renamed to the wire
        shape. The union feeds every validator in a mixed policy.
        """
        return {**self._to_llm_dict(), **self._to_agentic_dict()}
