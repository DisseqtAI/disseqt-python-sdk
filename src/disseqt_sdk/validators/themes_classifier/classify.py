"""Themes classifier validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...enums import ThemesClassifier, ValidatorDomain
from ...registry import register_validator
from ..base import ThemesClassifierValidator


def themes_request_handler(validator: ClassifyValidator) -> dict[str, Any]:
    """Custom request handler for themes classifier.

    Themes classifier sends data directly, not wrapped in input_data/config_input.
    """
    return validator.data.to_input_data()


def themes_response_handler(server_response: dict[str, Any]) -> dict[str, Any]:
    """Custom response handler for themes classifier.

    Themes classifier has its own response format with themes, confidence, sub_themes, etc.
    We return it as-is without forcing normalization to a standard format.
    """
    # Return the themes response as-is, preserving its natural structure
    return server_response


@register_validator(
    domain=ValidatorDomain.THEMES_CLASSIFIER,
    slug=ThemesClassifier.CLASSIFY.value,

    request_handler=themes_request_handler,
    response_handler=themes_response_handler,
)
@dataclass(slots=True)
class ClassifyValidator(ThemesClassifierValidator):
    """Validator for classifying themes in text."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.THEMES_CLASSIFIER)
        object.__setattr__(self, "_slug", ThemesClassifier.CLASSIFY.value)
