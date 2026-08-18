"""
OpenTelemetry GenAI semantic conventions.

Instrumentors dual-emit these `gen_ai.*` attributes alongside our
`agentic.*` attributes so traces are consumable by OTel-native tooling
(Grafana Tempo, Jaeger, OTel Collector) without extra translation.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""


class GenAIAttributes:
    """OpenTelemetry GenAI semantic-convention attribute names."""

    # System / provider
    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"

    # Request
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_IS_STREAM = "gen_ai.request.stream"

    # Prompt (input) / completion (output) — legacy langtrace-style
    PROMPT = "gen_ai.prompt"
    COMPLETION = "gen_ai.completion"

    # Response
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Tools
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call_id"


class GenAISystem:
    """Canonical `gen_ai.system` values."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GROQ = "groq"
    MISTRAL_AI = "mistral_ai"
    GEMINI = "gemini"
    VERTEX_AI = "vertex_ai"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI = "az.ai.inference"
    # LiteLLM proxies to other providers. Downstream tools should read
    # `gen_ai.response.model` / `agentic.provider.name` for the real target.
    LITELLM = "litellm"


class GenAIOperation:
    """Canonical `gen_ai.operation.name` values."""

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
