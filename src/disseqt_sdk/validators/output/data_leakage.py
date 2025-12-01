"""Output validation data leakage validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.DATA_LEAKAGE.value,
)
@dataclass(slots=True)
class OutputDataLeakageValidator(OutputValidator):
    """Validator for detecting data leakage in output."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.DATA_LEAKAGE.value)
