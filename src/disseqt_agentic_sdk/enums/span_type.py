"""
SpanType enum - defines the type of span for categorization.

Matches backend span_type values used in the database schema.
"""

from enum import Enum


class SpanType(str, Enum):
    """
    Span type - used for categorization and filtering.

    Values match the backend schema.
    """

    LLM_CALL = "llm_call"  # LLM model call
    AGENT_ACTION = "agent_action"  # Agent action/workflow
