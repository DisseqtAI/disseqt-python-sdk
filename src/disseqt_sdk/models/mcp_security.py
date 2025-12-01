"""MCP security request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _LLMTextFieldsMixin


@dataclass(slots=True)
class McpSecurityRequest(_LLMTextFieldsMixin):
    """Request model for MCP security validators."""

    # Minimal SDK surface:
    prompt: str
    # Optional:
    context: str | None = None
    response: str | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return self._to_llm_dict()
