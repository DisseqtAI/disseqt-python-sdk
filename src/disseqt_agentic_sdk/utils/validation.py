"""Shared HTTP header-value validation (re-export).

The implementation lives in :mod:`disseqt_logging.header_validation` so the
validation SDK (``disseqt_sdk``) can use it without importing the agentic SDK.
"""

from disseqt_logging.header_validation import validate_header_value

__all__ = ["validate_header_value"]
