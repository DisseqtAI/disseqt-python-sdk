"""Input validation intent-guard validator (per-project BLOCK list)."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import InputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import InputValidator


@register_validator(
    domain=ValidatorDomain.INPUT_VALIDATION,
    slug=InputValidation.INTENT_GUARD.value,
)
@dataclass(slots=True)
class IntentGuardValidator(InputValidator):
    """Validator for detecting disallowed (blocked) intents in input.

    Pass the block list via ``config.intents`` (``SDKConfigInput(intents=[...])``);
    leave it empty to use the project's dashboard-configured block list. A match
    above threshold yields ``threshold_validated_result == "Fail"`` and the
    response carries ``enforcement == "blocking"`` (callers should block the turn).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.INPUT_VALIDATION)
        object.__setattr__(self, "_slug", InputValidation.INTENT_GUARD.value)
