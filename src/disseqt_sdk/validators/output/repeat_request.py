"""Output validation repeat request validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.REPEAT_REQUEST.value,
)
@dataclass(slots=True)
class RepeatRequestValidator(OutputValidator):
    """Validator for detecting a request to repeat or clarify.

    Inverted: a detected repeat request is a negative finding, so a HIGH score
    fails (as with the safety validators). In an audio conversation it flags
    that the listener could not follow. Pass the text to evaluate in llm_output
    (response field); llm_input_query and llm_input_context can be empty.
    English only.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.REPEAT_REQUEST.value)
