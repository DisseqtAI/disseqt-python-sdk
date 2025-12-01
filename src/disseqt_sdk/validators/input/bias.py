"""Input validation bias validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.BIAS.value,
)
@dataclass(slots=True)
class BiasValidator(InputValidator):
    """Validator for detecting bias in input."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.BIAS.value)

    def normalize_response(self, server_response: dict[str, Any]) -> dict[str, Any]:
        """Normalize bias detection response to SDK format.

        Custom handling for bias detection responses.
        """

        # Use default handling but could customize here
        # For example, bias detection might have specific labels or scoring
        result = super().normalize_response(server_response)

        # Custom bias-specific processing could go here
        # e.g., map specific bias labels, adjust scoring, etc.

        return result
