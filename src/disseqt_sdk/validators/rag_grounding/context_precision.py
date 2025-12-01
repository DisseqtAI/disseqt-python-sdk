"""RAG grounding context precision validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import RagGrounding, ValidatorDomain
from ...registry import register_validator
from ..base import RagGroundingValidator


@register_validator(
    domain=ValidatorDomain.RAG_GROUNDING,
    slug=RagGrounding.CONTEXT_PRECISION.value,

)
@dataclass(slots=True)
class ContextPrecisionValidator(RagGroundingValidator):
    """Validator for measuring context precision in RAG systems."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.RAG_GROUNDING)
        object.__setattr__(self, "_slug", RagGrounding.CONTEXT_PRECISION.value)
