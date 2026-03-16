"""Output validation child safety validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.CHILD_SAFETY.value,
)
@dataclass(slots=True)
class OutputChildSafetyValidator(OutputValidator):
    """Validator for detecting harmful, age-inappropriate content in LLM output.

    input_requirements: ["Query", "Response"] — evaluation_parameters: ["Q", "R"]

    Pass the LLM response in ``response`` (llm_output / R).
    Optionally pass the original user query in ``prompt`` (llm_input_query / Q).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.CHILD_SAFETY.value)
