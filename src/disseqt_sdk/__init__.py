"""Disseqt SDK for Python.

Python SDK for Disseqt validators via the Dataset API.
Decorator-based dynamic registry. Enum-driven slugs.
Normalized responses with a dynamic `others` bag.
"""

from disseqt_logging import configure as configure_logging
from disseqt_logging import get_logger
from disseqt_logging import set_level as set_log_level

from ._version import SDK_VERSION as __version__
from .api_client import DisseqtAPIClient
from .client import Client
from .models.base import SDKConfigInput
from .policy import (
    PolicyDecision,
    PolicyRule,
    PolicyRuleset,
    any_blocking,
    is_async,
    is_blocking,
)
from .policy import parse as parse_policy

__all__ = [
    "Client",
    "DisseqtAPIClient",
    "PolicyDecision",
    "PolicyRule",
    "PolicyRuleset",
    "SDKConfigInput",
    "any_blocking",
    "configure_logging",
    "get_logger",
    "is_async",
    "is_blocking",
    "parse_policy",
    "set_log_level",
]
