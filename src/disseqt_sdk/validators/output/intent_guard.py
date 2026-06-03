"""Output validation intent-guard validator (per-project BLOCK list)."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import OutputValidation, ValidatorDomain
from ...registry import register_validator
from ..base import OutputValidator


@register_validator(
    domain=ValidatorDomain.OUTPUT_VALIDATION,
    slug=OutputValidation.INTENT_GUARD.value,
)
@dataclass(slots=True)
class OutputIntentGuardValidator(OutputValidator):
    """Detects disallowed (blocked) intents in the model's OUTPUT.

    Evaluate the response text via ``response`` (sent as ``llm_output``); pass the
    block list via ``config.intents`` (empty defers to the project's
    dashboard-configured block list). A match above threshold yields
    ``threshold_validated_result == "Fail"`` and the response carries
    ``enforcement == "blocking"`` (callers should block the turn).
    """

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.OUTPUT_VALIDATION)
        object.__setattr__(self, "_slug", OutputValidation.INTENT_GUARD.value)
