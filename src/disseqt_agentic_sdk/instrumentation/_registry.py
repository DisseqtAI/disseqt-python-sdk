"""
Static registry mapping provider names to their instrumentor classes.

Kept as string paths so importing the registry doesn't force-import
provider modules (each provider module has to import its SDK to declare
patch targets — we only want to pay that cost when the user actually
uses the provider).
"""

# name → dotted path to the instrumentor class. Iterated by
# ``instrument_all(...)``; one entry per instrumentor.
INSTRUMENTOR_CLASSES: dict[str, str] = {
    # LLM providers
    "openai": "disseqt_agentic_sdk.instrumentation.openai.OpenAIInstrumentor",
    "anthropic": "disseqt_agentic_sdk.instrumentation.anthropic.AnthropicInstrumentor",
    "groq": "disseqt_agentic_sdk.instrumentation.groq.GroqInstrumentor",
    "mistralai": "disseqt_agentic_sdk.instrumentation.mistral.MistralInstrumentor",
    "cohere": "disseqt_agentic_sdk.instrumentation.cohere.CohereInstrumentor",
    "google-genai": "disseqt_agentic_sdk.instrumentation.gemini.GeminiInstrumentor",
    # Router / proxy
    "litellm": "disseqt_agentic_sdk.instrumentation.litellm.LiteLLMInstrumentor",
}

# User-friendly alias → canonical registry key. These match the
# ``pyproject.toml`` extras names ([gemini], [mistral]) so a user who
# installed ``disseqt-ai-sdk[gemini]`` and calls ``instrument("gemini",
# client)`` reaches the right instrumentor instead of a silent
# unknown_provider no-op (TP-2128 P2 #2.13). Aliases are NOT iterated
# by ``instrument_all`` — that walks INSTRUMENTOR_CLASSES only so each
# instrumentor is applied exactly once.
INSTRUMENTOR_ALIASES: dict[str, str] = {
    "gemini": "google-genai",
    "mistral": "mistralai",
}


def resolve_provider_name(name: str) -> str:
    """Return the canonical registry key for ``name`` (aliases resolved)."""
    return INSTRUMENTOR_ALIASES.get(name, name)
