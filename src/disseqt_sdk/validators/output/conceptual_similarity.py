"""Output validation conceptual similarity validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.CONCEPTUAL_SIMILARITY.value,
)
@dataclass(slots=True)
class ConceptualSimilarityValidator(OutputValidator):
    """Validator for measuring semantic similarity between concepts in prompt and response.

    Both llm_input_query (prompt) and llm_output (response) are mandatory fields.
    It evaluates if llm_output is semantically similar to llm_input_query.
    llm_input_context can be empty.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.CONCEPTUAL_SIMILARITY.value)

