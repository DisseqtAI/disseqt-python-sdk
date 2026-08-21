"""
Semantics Module

Semantic conventions for agentic AI (agent, model, tool attributes).
"""

from .agentic import (
    AgenticAttributes,
    AgenticCacheOperation,
    AgenticFinishReason,
    AgenticOperation,
    AgenticOutputType,
    AgenticProvider,
    BatchStatus,
)
from .gen_ai import GenAIAttributes, GenAIOperation, GenAISystem

__all__ = [
    "AgenticOperation",
    "AgenticAttributes",
    "AgenticOutputType",
    "AgenticFinishReason",
    "AgenticProvider",
    "AgenticCacheOperation",
    "BatchStatus",
    "GenAIAttributes",
    "GenAISystem",
    "GenAIOperation",
]
