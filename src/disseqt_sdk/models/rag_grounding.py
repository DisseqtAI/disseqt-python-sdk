"""RAG grounding request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _LLMTextFieldsMixin


@dataclass(slots=True)
class RagGroundingRequest(_LLMTextFieldsMixin):
    """Request model for RAG grounding validators."""

    # Typically you'll set at least prompt/context or response
    prompt: str | None = None
    context: str | None = None
    response: str | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return self._to_llm_dict()
