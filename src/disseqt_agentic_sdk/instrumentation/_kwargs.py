"""
Named constants for LLM SDK request kwarg keys.

Extracting these prevents typos like ``kwargs.get("stram")`` that would
silently return None and quietly disable capture for that field. Every
provider wrapper reaches into request kwargs by string key; the constants
below are the ones that appear in more than one place or are load-bearing
for downstream behaviour (streaming path selection, model tagging).
"""

from __future__ import annotations

# Common across most chat/completions APIs.
KW_MODEL = "model"
KW_STREAM = "stream"
KW_MESSAGES = "messages"

# Legacy / provider-specific request payloads.
KW_PROMPT = "prompt"  # openai legacy completions
KW_INPUT = "input"  # openai embeddings
KW_SYSTEM = "system"  # anthropic system prompt
KW_CONTENTS = "contents"  # gemini
KW_CONFIG = "config"  # gemini generation config
