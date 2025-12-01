"""Output validation request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _LLMTextFieldsMixin


@dataclass(slots=True)
class OutputValidationRequest(_LLMTextFieldsMixin):
    """Request model for output validation validators."""

    response: str  # required

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return self._to_llm_dict()
