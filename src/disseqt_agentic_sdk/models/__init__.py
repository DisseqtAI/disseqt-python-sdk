"""
Models Module

Data models matching the backend PostgreSQL schema:
- EnrichedSpan - matches backend models.EnrichedSpan (primary span model)
"""

from .span import EnrichedSpan

__all__ = ["EnrichedSpan"]
