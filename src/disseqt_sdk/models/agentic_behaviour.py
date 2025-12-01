"""Agentic behaviour request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _AgenticFieldsMixin


@dataclass(slots=True)
class AgenticBehaviourRequest(_AgenticFieldsMixin):
    """Request model for agentic behaviour validators."""

    # All fields optional in SDK; you will usually provide them
    conversation_history: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    agent_responses: list[str] | None = None
    reference_data: dict[str, Any] | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload.

        1:1 mapping to Postman input_data.
        """
        return self._to_agentic_dict()
