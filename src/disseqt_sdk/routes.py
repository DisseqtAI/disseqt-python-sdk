"""Route handling for Disseqt SDK."""

from __future__ import annotations

from .enums import ValidatorDomain


def build_validator_url(
    base_url: str, domain: ValidatorDomain, slug: str, path_template: str
) -> str:
    """Build the full URL for a validator endpoint.

    Args:
        base_url: Base URL for the API
        domain: Validator domain
        slug: Validator slug
        path_template: URL path template

    Returns:
        Complete URL for the validator endpoint
    """
    # Format the path template
    path = path_template.format(domain=domain.value, validator=slug)

    # Combine base URL and path
    if base_url.endswith("/"):
        base_url = base_url.rstrip("/")

    if not path.startswith("/"):
        path = "/" + path

    return f"{base_url}{path}"
