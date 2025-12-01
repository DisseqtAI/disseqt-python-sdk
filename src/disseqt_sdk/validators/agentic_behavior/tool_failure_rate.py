"""Agentic behavior tool failure rate validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import AgenticBehavior, ValidatorDomain
from ...registry import register_validator
from ..base import AgenticBehaviourValidator


@register_validator(
    domain=ValidatorDomain.AGENTIC_BEHAVIOR,
    slug=AgenticBehavior.TOOL_FAILURE_RATE.value,
)
@dataclass(slots=True)
class ToolFailureRateValidator(AgenticBehaviourValidator):
    """Validator for measuring tool failure rate in agentic behavior."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.AGENTIC_BEHAVIOR)
        object.__setattr__(self, "_slug", AgenticBehavior.TOOL_FAILURE_RATE.value)
