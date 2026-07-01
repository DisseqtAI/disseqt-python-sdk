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

__version__ = "0.5.0"
__all__ = [
    "Client",
    "DisseqtAPIClient",
    "SDKConfigInput",
    "configure_logging",
    "get_logger",
    "set_log_level",
]
