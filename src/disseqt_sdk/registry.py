"""Registry system for Disseqt SDK validators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .enums import ValidatorDomain

T = TypeVar("T")

# Global registry to store validator metadata
_VALIDATOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_validator(
    domain: ValidatorDomain,
    slug: str,
    path_template: str = "/api/v1/sdk/validators/{domain}/{validator}",
    request_handler: Callable[[Any], dict[str, Any]] | None = None,
    response_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator to register a validator with the SDK.

    Args:
        domain: The validator domain enum
        slug: The validator slug (should be from appropriate enum)
        path_template: URL path template for the validator
        request_handler: Optional custom request handler function
        response_handler: Optional custom response handler function

    Returns:
        The decorated validator class
    """

    def decorator(cls: type[T]) -> type[T]:
        # Create a unique key for this validator
        key = f"{domain.value}:{slug}"

        # Store metadata in registry
        _VALIDATOR_REGISTRY[key] = {
            "class": cls,
            "domain": domain,
            "slug": slug,
            "path_template": path_template,
            "request_handler": request_handler,
            "response_handler": response_handler,
        }

        return cls

    return decorator


def get_validator_metadata(domain: ValidatorDomain, slug: str) -> dict[str, Any]:
    """Get metadata for a registered validator.

    Args:
        domain: The validator domain
        slug: The validator slug

    Returns:
        Validator metadata dictionary

    Raises:
        KeyError: If validator is not registered
    """
    key = f"{domain.value}:{slug}"
    if key not in _VALIDATOR_REGISTRY:
        raise KeyError(f"Validator not registered: {domain.value}:{slug}")

    return _VALIDATOR_REGISTRY[key]


def list_registered_validators() -> dict[str, dict[str, Any]]:
    """List all registered validators.

    Returns:
        Dictionary of all registered validators
    """
    return _VALIDATOR_REGISTRY.copy()
