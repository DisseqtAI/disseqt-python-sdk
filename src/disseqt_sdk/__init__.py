"""Disseqt SDK for Python.

Python SDK for Disseqt validators via the Dataset API.
Decorator-based dynamic registry. Enum-driven slugs.
Normalized responses with a dynamic `others` bag.
"""

from .client import Client
from .models.base import SDKConfigInput

__version__ = "0.2.3"
__all__ = ["Client", "SDKConfigInput"]
