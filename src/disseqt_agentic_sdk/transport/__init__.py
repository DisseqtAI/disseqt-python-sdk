"""
Transport Module

HTTP transport layer for sending traces/spans to the backend API.
"""

from .http import HTTPTransport

__all__ = ["HTTPTransport"]
