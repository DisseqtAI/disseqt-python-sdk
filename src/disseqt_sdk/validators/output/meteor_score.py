"""Output validation METEOR score validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.METEOR_SCORE.value,
)
@dataclass(slots=True)
class MeteorScoreValidator(OutputValidator):
    """Validator for calculating METEOR score of output."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.METEOR_SCORE.value)
