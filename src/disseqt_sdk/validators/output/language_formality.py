"""Output validation language formality validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.LANGUAGE_FORMALITY.value,
)
@dataclass(slots=True)
class LanguageFormalityValidator(OutputValidator):
    """Validator for classifying the response register as formal or casual.

    actual_value is the probability the response is in a formal register; it
    fails when the response is too casual (below the threshold). Pass the text
    to evaluate in llm_output (response field); llm_input_query and
    llm_input_context can be empty.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.LANGUAGE_FORMALITY.value)
