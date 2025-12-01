"""MCP security validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import McpSecurity, ValidatorDomain
from ...registry import register_validator
from ..base import McpSecurityValidator


@register_validator(
    domain=ValidatorDomain.MCP_SECURITY,
    slug=McpSecurity.PROMPT_INJECTION.value,

)
@dataclass(slots=True)
class McpPromptInjectionValidator(McpSecurityValidator):
    """Validator for detecting prompt injection in MCP security."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.MCP_SECURITY)
        object.__setattr__(self, "_slug", McpSecurity.PROMPT_INJECTION.value)
