"""Input validation child safety validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.CHILD_SAFETY.value,
)
@dataclass(slots=True)
class ChildSafetyValidator(InputValidator):
    """Validator for detecting content that poses risks to children's wellbeing.

    input_requirements: ["Query", "Response"] — evaluation_parameters: ["Q", "R"]

    Pass the user query in ``prompt`` (llm_input_query / Q).
    Optionally pass the LLM response in ``response`` (llm_output / R).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.CHILD_SAFETY.value)
