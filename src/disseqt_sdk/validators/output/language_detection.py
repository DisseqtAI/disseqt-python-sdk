"""Output validation language detection validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.LANGUAGE_DETECTION.value,
)
@dataclass(slots=True)
class LanguageDetectionValidator(OutputValidator):
    """Validator for checking the response is in the expected language.

    Scores the share of the response written in the expected language, so
    wrong-language output fails even when the detector is confident. Pass the
    text to evaluate in llm_output (response field); llm_input_query and
    llm_input_context can be empty.

    Set the expected language with ``requested_language`` in the config
    (ISO 639-1, e.g. "de"); it defaults to English server-side. ``min_share``
    tunes which minor languages are dropped from the breakdown.
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.LANGUAGE_DETECTION.value)
