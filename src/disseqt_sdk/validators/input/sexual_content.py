"""Input validation sexual content validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.SEXUAL_CONTENT.value,
)
@dataclass(slots=True)
class SexualContentValidator(InputValidator):
    """Validator for detecting sexual content in input."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.SEXUAL_CONTENT.value)
