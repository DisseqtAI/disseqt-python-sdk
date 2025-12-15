"""Output validation creativity validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.CREATIVITY.value,
)
@dataclass(slots=True)
class CreativityValidator(OutputValidator):
    """Validator for measuring how creative and original the response is.

    Analyzes relevance, novelty, and linguistic diversity through weighted scoring
    of contextual appropriateness, innovative content generation, and lexical variation.

    Both llm_input_context (context) and llm_output (response) are mandatory fields.
    llm_input_query can be empty.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.CREATIVITY.value)

