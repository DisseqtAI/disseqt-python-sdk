"""Input validation intent-compliance validator (per-project ALLOW list)."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.INTENT_COMPLIANCE.value,
)
@dataclass(slots=True)
class IntentComplianceValidator(InputValidator):
    """Validator for checking input intent against an allow list.

    Pass the allow list via ``config.intents`` (``SDKConfigInput(intents=[...])``);
    leave it empty to use the project's dashboard-configured allow list. A
    non-compliant intent yields ``threshold_validated_result == "Fail"`` and the
    response carries ``enforcement == "advisory"`` (surfaced as a flag; does not
    block the turn).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.INTENT_COMPLIANCE.value)
