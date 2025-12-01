"""Input validation hate speech validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.HATE_SPEECH.value,

)
@dataclass(slots=True)
class HateSpeechValidator(InputValidator):
    """Validator for detecting hate speech in input."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.HATE_SPEECH.value)
