"""Base models and mixins for Disseqt SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SDKConfigInput:
    """Configuration input for SDK validators."""

    threshold: float
    custom_labels: list[str] | None = None
    label_thresholds: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        out: dict[str, Any] = {"threshold": self.threshold}
        if self.custom_labels:
            out["custom_labels"] = self.custom_labels
        if self.label_thresholds:
            out["label_thresholds"] = self.label_thresholds
        return out


class _LLMTextFieldsMixin:
    """Common mixin for LLM text fields (maps to llm_* on wire)."""

    prompt: str | None = None
    context: str | None = None
    response: str | None = None

    def _to_llm_dict(self) -> dict[str, Any]:
        """Convert LLM fields to wire format."""
        d: dict[str, Any] = {}
        if getattr(self, "prompt", None) is not None:
            d["llm_input_query"] = self.prompt  # SDK prompt → wire
        if getattr(self, "context", None) is not None:
            d["llm_input_context"] = self.context
        if getattr(self, "response", None) is not None:
            d["llm_output"] = self.response
        return d


class _AgenticFieldsMixin:
    """Common mixin for agentic arrays/maps (1:1 with Postman)."""

    conversation_history: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    agent_responses: list[str] | None = None
    reference_data: dict[str, Any] | None = None

    def _to_agentic_dict(self) -> dict[str, Any]:
        """Convert agentic fields to wire format."""
        out: dict[str, Any] = {}
        if self.conversation_history is not None:
            out["conversation_history"] = self.conversation_history
        if self.tool_calls is not None:
            out["tool_calls"] = self.tool_calls
        if self.agent_responses is not None:
            out["agent_responses"] = self.agent_responses
        if self.reference_data is not None:
            out["reference_data"] = self.reference_data
        return out
