"""Output validation diversity validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.DIVERSITY.value,
)
@dataclass(slots=True)
class DiversityValidator(OutputValidator):
    """Validator for measuring the diversity and non-repetitiveness of content.

    Pass the text to evaluate in llm_output (response field).
    llm_input_query and llm_input_context can be empty.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.DIVERSITY.value)

