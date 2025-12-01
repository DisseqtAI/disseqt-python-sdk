"""Input validation violence validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.VIOLENCE.value,

)
@dataclass(slots=True)
class ViolenceValidator(InputValidator):
    """Validator for detecting violent content in input."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.VIOLENCE.value)
