"""
Agentic semantic conventions - attribute names for agentic AI operations.

Based on OpenTelemetry GenAI conventions but using 'agentic.*' prefix
for agentic-specific operations.
"""


# Operation names
class AgenticOperation:
    """Agentic operation types"""

    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    EXECUTE_TOOL = "execute_tool"
    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"


# Attribute name constants
class AgenticAttributes:
    """Agentic semantic convention attribute names"""

    # Operation
    OPERATION_NAME = "agentic.operation.name"

    # Agent attributes
    AGENT_NAME = "agentic.agent.name"
    AGENT_ID = "agentic.agent.id"
    AGENT_VERSION = "agentic.agent.version"

    # Tool attributes
    TOOL_NAME = "agentic.tool.name"
    TOOL_CALL_ID = "agentic.tool.call_id"
    TOOL_DEFINITIONS = "agentic.tool.definitions"

    # Model/Provider attributes
    REQUEST_MODEL = "agentic.request.model"
    PROVIDER_NAME = "agentic.provider.name"

    # Request parameters
    REQUEST_TEMPERATURE = "agentic.request.temperature"
    REQUEST_MAX_TOKENS = "agentic.request.max_tokens"
    REQUEST_TOP_P = "agentic.request.top_p"
    REQUEST_TOP_K = "agentic.request.top_k"
    REQUEST_FREQUENCY_PENALTY = "agentic.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "agentic.request.presence_penalty"

    # Prompt attributes
    PROMPT_NAME = "agentic.prompt.name"
    PROMPT_VARIABLES = "agentic.prompt.variables"
    PROMPT_VERSION = "agentic.prompt.version"

    # Usage attributes
    USAGE_INPUT_TOKENS = "agentic.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "agentic.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "agentic.usage.total_tokens"
    USAGE_INPUT_CHARACTERS = "agentic.usage.input_characters"
    USAGE_OUTPUT_CHARACTERS = "agentic.usage.output_characters"

    # Messages
    INPUT_MESSAGES = "agentic.input.messages"
    OUTPUT_MESSAGES = "agentic.output.messages"
    SYSTEM_INSTRUCTIONS = "agentic.system_instructions"

    # Output
    OUTPUT_TYPE = "agentic.output.type"

    # Response
    RESPONSE_ID = "agentic.response.id"
    RESPONSE_MODEL = "agentic.response.model"
    RESPONSE_FINISH_REASON = "agentic.response.finish_reason"

    # Cache
    CACHE_HIT = "agentic.cache.hit"
    CACHE_OPERATION = "agentic.cache.operation"

    # Error
    ERROR_TYPE = "agentic.error.type"
    ERROR_MESSAGE = "agentic.error.message"
    ERROR_CODE = "agentic.error.code"


# Output types
class AgenticOutputType:
    """Agentic output types"""

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    SPEECH = "speech"


# Finish reasons
class AgenticFinishReason:
    """Agentic finish reasons"""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"
    RECITATION = "recitation"
    ERROR = "error"
    OTHER = "other"


# Cache operations
class AgenticCacheOperation:
    """Agentic cache operation types"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


# Provider names
class AgenticProvider:
    """Agentic provider names"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI = "azure.ai"
    COHERE = "cohere"
    MISTRAL_AI = "mistral_ai"
    GROQ = "groq"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"
