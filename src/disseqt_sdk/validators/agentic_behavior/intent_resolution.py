"""Agentic behavior intent resolution validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import AgenticBehavior, ValidatorDomain
from ...registry import register_validator
from ..base import AgenticBehaviourValidator


@register_validator(
    domain=ValidatorDomain.AGENTIC_BEHAVIOR,
    slug=AgenticBehavior.INTENT_RESOLUTION.value,
)
@dataclass(slots=True)
class IntentResolutionValidator(AgenticBehaviourValidator):
    """Validator for measuring intent resolution in agentic behavior."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.AGENTIC_BEHAVIOR)
        object.__setattr__(self, "_slug", AgenticBehavior.INTENT_RESOLUTION.value)
