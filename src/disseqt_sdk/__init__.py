"""Disseqt SDK for Python.

Python SDK for Disseqt validators via the Dataset API.
Decorator-based dynamic registry. Enum-driven slugs.
Normalized responses with a dynamic `others` bag.
"""

from .api_client import DisseqtAPIClient
from .client import Client
from .models.base import SDKConfigInput
from .policy import (
    PolicyDecision,
    PolicyRule,
    PolicyRuleset,
    is_async,
    is_blocking,
)
from .policy import parse as parse_policy

__version__ = "0.5.0"
__all__ = [
    "Client",
    "DisseqtAPIClient",
    "PolicyDecision",
    "PolicyRule",
    "PolicyRuleset",
    "SDKConfigInput",
    "is_async",
    "is_blocking",
    "parse_policy",
]
