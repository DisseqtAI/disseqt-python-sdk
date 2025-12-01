"""Composite score request model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompositeScoreRequest:
    """Request model for composite score evaluation.

    This is a special evaluation that combines multiple validators
    into a single weighted score with breakdown.
    """

    llm_input_query: str
    llm_output: str
    llm_input_context: str | None = None
    evaluation_mode: str = "binary_threshold"
    weights_override: dict[str, Any] | None = None
    labels_thresholds_override: dict[str, Any] | None = None
    overall_confidence: dict[str, Any] | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to API input_data format for composite score evaluation.

        Returns a dictionary with input_data and options following the API structure.
        """
        # Build input_data section
        input_data: dict[str, Any] = {
            "llm_input_query": self.llm_input_query,
            "llm_output": self.llm_output,
        }

        if self.llm_input_context:
            input_data["llm_input_context"] = self.llm_input_context

        # Build options section
        options: dict[str, Any] = {
            "evaluation_mode": self.evaluation_mode,
        }

        if self.weights_override:
            options["weights_override"] = self.weights_override

        if self.labels_thresholds_override:
            options["labels_thresholds_override"] = self.labels_thresholds_override

        if self.overall_confidence:
            options["overall_confidence"] = self.overall_confidence

        # Return complete payload structure
        return {
            "input_data": input_data,
            "options": options,
        }

