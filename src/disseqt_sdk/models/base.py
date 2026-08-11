"""Base models and mixins for Disseqt SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SDKConfigInput:
    """Configuration input for SDK validators.

    ``custom_labels`` / ``label_thresholds`` apply to classic ML validators
    AND to LLM judges (when ``llm_as_a_judge=True``). For judges they are
    relabel-only — the server-owned score, pass/fail verdict, and rubric are
    untouched — and the labels follow the judge's score axis: safety judges
    score severity, so a HIGHER score is WORSE (order your labels
    accordingly, e.g. ``["OK", "Bad", "Awful", "Severe"]``).
    """

    threshold: float
    custom_labels: list[str] | None = None
    label_thresholds: list[float] | None = None
    # Allow/block intent labels for the intent-guard / intent-compliance
    # validators. Sent inside config_input; an empty/None list is omitted so the
    # server falls back to the project's dashboard-configured intent list.
    intents: list[str] | None = None
    # Reroute this validation to the paired certified LLM judge instead of
    # the classic ML validator. Validators without a judge pairing fall back
    # gracefully to the ML path (no error).
    llm_as_a_judge: bool = False
    # Optional per-call judge override, honored only with llm_as_a_judge=True.
    # Keys: "custom_llm_id" (integration to use instead of the project's
    # default judge integration), "model", "criteria". The provider remains
    # server-authoritative.
    #
    # Finding custom_llm_id: Dashboard -> AI Inventory -> LLM Integrations —
    # the ID column (and the row's view modal) has one-click copy. It is the
    # INTEGRATION's id, not a model name. Only Permanent integrations have
    # one; Temporary models are session-only and cannot be used here.
    #
    # "criteria" shapes QUALITY judges only. Certified SAFETY judges run
    # their frozen rubric verbatim and ignore caller criteria (the response
    # stamps others.criteria_ignored=true when that happens).
    judge: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        out: dict[str, Any] = {"threshold": self.threshold}
        if self.custom_labels:
            out["custom_labels"] = self.custom_labels
        if self.label_thresholds:
            out["label_thresholds"] = self.label_thresholds
        if self.intents:
            out["intents"] = self.intents
        if self.llm_as_a_judge:
            out["llm_as_a_judge"] = True
        if self.judge:
            out["judge"] = self.judge
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
