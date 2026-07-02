"""Disseqt SDK for Python.

Python SDK for Disseqt validators via the Dataset API.
Decorator-based dynamic registry. Enum-driven slugs.
Normalized responses with a dynamic `others` bag.
"""

from disseqt_logging import configure as configure_logging
from disseqt_logging import get_logger
from disseqt_logging import set_level as set_log_level

from .api_client import DisseqtAPIClient
from .client import Client
from .models.base import SDKConfigInput
from .policy import (
    PolicyDecision,
    PolicyRule,
    PolicyRuleset,
    is_async,
    is_blocking,
    is_policy_skipped,
)
from .policy import parse as parse_policy

__version__ = "0.7.0"
__all__ = [
    "Client",
    "DisseqtAPIClient",
    "PolicyDecision",
    "PolicyRule",
    "PolicyRuleset",
    "SDKConfigInput",
    "configure_logging",
    "get_logger",
    "is_async",
    "is_blocking",
    "is_policy_skipped",
    "parse_policy",
    "set_log_level",
]
