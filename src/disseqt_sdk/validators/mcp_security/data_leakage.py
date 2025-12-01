"""MCP security data leakage validators."""

from __future__ import annotations

from dataclasses import dataclass

from ...enums import McpSecurity, ValidatorDomain
from ...registry import register_validator
from ..base import McpSecurityValidator


@register_validator(
    domain=ValidatorDomain.MCP_SECURITY,
    slug=McpSecurity.DATA_LEAKAGE.value,

)
@dataclass(slots=True)
class DataLeakageValidator(McpSecurityValidator):
    """Validator for detecting data leakage in MCP security."""

    def __post_init__(self) -> None:
        """Set domain and slug after initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.MCP_SECURITY)
        object.__setattr__(self, "_slug", McpSecurity.DATA_LEAKAGE.value)
