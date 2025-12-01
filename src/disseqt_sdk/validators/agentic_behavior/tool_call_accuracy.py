"""Agentic behavior tool call accuracy validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...enums import AgenticBehavior, ValidatorDomain
from ...registry import register_validator
from ..base import AgenticBehaviourValidator


@register_validator(
    domain=ValidatorDomain.AGENTIC_BEHAVIOR,
    slug=AgenticBehavior.TOOL_CALL_ACCURACY.value,

)
@dataclass(slots=True)
class ToolCallAccuracyValidator(AgenticBehaviourValidator):
    """Validator for checking tool call accuracy in agentic behavior."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.AGENTIC_BEHAVIOR)
        object.__setattr__(self, "_slug", AgenticBehavior.TOOL_CALL_ACCURACY.value)

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload with custom tool call accuracy formatting."""
        # Custom payload formatting for tool call accuracy
        payload = super().to_payload()

        # Could add custom formatting here, e.g.:
        # - Validate tool_calls structure
        # - Add tool call accuracy specific config
        # - Transform tool call format if needed

        return payload

    def normalize_response(self, server_response: dict[str, Any]) -> dict[str, Any]:
        """Normalize tool call accuracy response to SDK format."""
        # Use default handling but could customize for tool call accuracy metrics
        result = super().normalize_response(server_response)

        # Custom tool call accuracy processing could go here
        # e.g., parse tool call success rates, accuracy metrics, etc.

        return result
