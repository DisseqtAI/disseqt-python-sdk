"""
Static registry mapping provider names to their instrumentor classes.

Kept as string paths so importing the registry doesn't force-import
provider modules (each provider module has to import its SDK to declare
patch targets — we only want to pay that cost when the user actually
uses the provider).
"""

# name → dotted path to the instrumentor class.
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
