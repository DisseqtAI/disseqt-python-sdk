"""MCP security request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _LLMTextFieldsMixin


@dataclass(slots=True)
class McpSecurityRequest(_LLMTextFieldsMixin):
    """Request model for MCP security validators.

    Field requirements per validator:

    - **Prompt Injection** (``evaluation_parameters: ["Q"]``):
      pass the user query in ``prompt`` — only ``llm_input_query`` is sent.

    - **Data Leakage** (``evaluation_parameters: ["R"]``):
      pass the LLM response in ``response`` — only ``llm_output`` is sent.

    - **Insecure Output** (``evaluation_parameters: ["R"]``):
      pass the LLM response in ``response`` — only ``llm_output`` is sent.
    """

    # All fields optional — different MCP validators use different fields:
    # PromptInjection → prompt (llm_input_query / Q)
    # DataLeakage / InsecureOutput → response (llm_output / R)
    prompt: str | None = None
    context: str | None = None
    response: str | None = None

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return self._to_llm_dict()
