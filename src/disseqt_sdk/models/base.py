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
    # Which LLM Integration judges the run. REQUIRED when llm_as_a_judge=True
    # (enforced in __post_init__): an explicit id keeps judge selection
    # auditable and fails fast at construction, instead of a server-side 4xx
    # — or worse, the silent wrong-integration fallback a misspelled dict key
    # used to produce. It serializes into the wire's nested judge block as
    # "custom_llm_id", so the server contract is unchanged.
    #
    # NOTE: the SERVER also supports a project-default judge integration
    # (custom_llms.is_default_judge); this SDK deliberately does not expose
    # that fallback — there is currently no dashboard UI to set the default.
    # If that UI ships, relax the __post_init__ check rather than adding a
    # second selection mechanism here.
    #
    # Finding the id: Dashboard -> AI Inventory -> LLM Integrations — the ID
    # column (and the row's view modal) has one-click copy. It is the
    # INTEGRATION's id, not a model name. Only Permanent integrations have
    # one; Temporary models are session-only and cannot be used here.
    llm_id: str | None = None
    # Optional per-call judge override, honored only with llm_as_a_judge=True.
    # Keys: "custom_llm_id" (prefer the first-class llm_id field, which wins
    # on conflict), "model", "criteria". The provider remains
    # server-authoritative.
    #
    # "criteria" shapes QUALITY judges only. Certified SAFETY judges run
    # their frozen rubric verbatim and ignore caller criteria (the response
    # stamps others.criteria_ignored=true when that happens).
    judge: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        has_integration_id = bool(self.llm_id or (self.judge or {}).get("custom_llm_id"))
        if self.llm_as_a_judge and not has_integration_id:
            raise ValueError(
                "llm_as_a_judge=True requires llm_id — the LLM Integration "
                "that judges the run. Copy it from Dashboard -> AI Inventory "
                "-> LLM Integrations (ID column)."
            )
        if self.llm_id and not self.llm_as_a_judge:
            raise ValueError(
                "llm_id is only used with llm_as_a_judge=True — set the flag, "
                "or drop llm_id to run the traditional ML validator."
            )

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
        # The wire format nests the integration selector under "judge" as
        # "custom_llm_id"; the flat llm_id is client-side sugar and wins
        # over a conflicting dict key.
        judge_block = dict(self.judge) if self.judge else {}
        if self.llm_id:
            judge_block["custom_llm_id"] = self.llm_id
        if judge_block:
            out["judge"] = judge_block
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
