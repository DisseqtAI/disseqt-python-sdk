"""Output validation intent-compliance validator (per-project ALLOW list)."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.INTENT_COMPLIANCE.value,
)
@dataclass(slots=True)
class OutputIntentComplianceValidator(OutputValidator):
    """Checks the model's OUTPUT intent against an allow list.

    Evaluate the response text via ``response`` (sent as ``llm_output``); pass the
    allow list via ``config.intents`` (empty defers to the project's
    dashboard-configured allow list). A non-compliant intent yields
    ``threshold_validated_result == "Fail"`` and the response carries
    ``enforcement == "advisory"`` (surfaced as a flag; does not block the turn).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.INTENT_COMPLIANCE.value)
