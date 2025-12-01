"""Base validator classes for Disseqt SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..enums import ValidatorDomain
from ..models.base import SDKConfigInput

if TYPE_CHECKING:
    from ..models.agentic_behaviour import AgenticBehaviourRequest
    from ..models.input_validation import InputValidationRequest
    from ..models.mcp_security import McpSecurityRequest
    from ..models.output_validation import OutputValidationRequest
    from ..models.rag_grounding import RagGroundingRequest
    from ..models.themes_classifier import ThemesClassifierRequest


@dataclass(slots=True)
class BaseValidator:
    """Base class for all validators."""

    config: SDKConfigInput
    _domain: ValidatorDomain = field(init=False, repr=False)
    _slug: str = field(init=False, repr=False)
    _path_template: str = field(
        init=False, repr=False, default="/api/v1/sdk/validators/{domain}/{validator}"
    )

    @property
    def domain(self) -> ValidatorDomain:
        """Get the validator domain."""
        return self._domain

    @property
    def slug(self) -> str:
        """Get the validator slug."""
        return self._slug

    def to_payload(self) -> dict[str, Any]:
        """Convert validator to API payload."""
        raise NotImplementedError



# Domain specializations (note the domain-specific request objects)


@dataclass(slots=True)
class InputValidator(BaseValidator):
    """Base class for input validation validators."""

    data: InputValidationRequest

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "input_data": self.data.to_input_data(),
            "config_input": self.config.to_dict(),
        }


@dataclass(slots=True)
class OutputValidator(BaseValidator):
    """Base class for output validation validators."""

    data: OutputValidationRequest

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "input_data": self.data.to_input_data(),
            "config_input": self.config.to_dict(),
        }


@dataclass(slots=True)
class RagGroundingValidator(BaseValidator):
    """Base class for RAG grounding validators."""

    data: RagGroundingRequest

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "input_data": self.data.to_input_data(),
            "config_input": self.config.to_dict(),
        }


@dataclass(slots=True)
class AgenticBehaviourValidator(BaseValidator):
    """Base class for agentic behaviour validators."""

    data: AgenticBehaviourRequest

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "input_data": self.data.to_input_data(),
            "config_input": self.config.to_dict(),
        }


@dataclass(slots=True)
class McpSecurityValidator(BaseValidator):
    """Base class for MCP security validators."""

    data: McpSecurityRequest

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "input_data": self.data.to_input_data(),
            "config_input": self.config.to_dict(),
        }


@dataclass(slots=True)
class ThemesClassifierValidator:
    """Base class for themes classifier validators."""

    data: ThemesClassifierRequest
    _domain: ValidatorDomain = field(init=False, repr=False)
    _slug: str = field(init=False, repr=False)
    _path_template: str = field(
        init=False, repr=False, default="/api/v1/sdk/validators/{domain}/{validator}"
    )

    @property
    def domain(self) -> ValidatorDomain:
        """Get the validator domain."""
        return self._domain

    @property
    def slug(self) -> str:
        """Get the validator slug."""
        return self._slug

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        # Themes classifier doesn't use config_input, just the data directly
        return self.data.to_input_data()

