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
    # Async batch-inference lifecycle. Each SDK call gets its own span
    # tagged with `agentic.batch.id` so consumers can group create/retrieve/
    # cancel spans for the same job across time.
    BATCH_CREATE = "batch.create"
    BATCH_RETRIEVE = "batch.retrieve"
    BATCH_CANCEL = "batch.cancel"


class BatchStatus:
    """Canonical batch-job status values, normalized across providers."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


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
    TOOL_ARGS = "agentic.tool.args"
    TOOL_RESULT = "agentic.tool.result"
    # Full list of tool calls emitted by the model on a MODEL_EXEC span, or
    # accumulated across an AGENT_EXEC span. Read by the tool-call-accuracy,
    # plan-optimality, plan-coherence, and tool-failure-rate validators.
    TOOL_CALLS = "agentic.tool_calls"
    REQUEST_TOOLS = "agentic.request.tools"

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
    # Wall-clock duration of the wrapped SDK call, in milliseconds.
    REQUEST_DURATION_MS = "agentic.request.duration_ms"

    # Cache
    CACHE_HIT = "agentic.cache.hit"
    CACHE_OPERATION = "agentic.cache.operation"

    # Error
    ERROR_TYPE = "agentic.error.type"
    ERROR_MESSAGE = "agentic.error.message"
    ERROR_CODE = "agentic.error.code"

    # Batch-job attributes. Emitted on every create/retrieve/cancel span
    # tagged with the same batch id so downstream can group by lifecycle.
    BATCH_ID = "agentic.batch.id"
    BATCH_STATUS = "agentic.batch.status"
    BATCH_ENDPOINT = "agentic.batch.endpoint"
    BATCH_REQUEST_COUNT = "agentic.batch.request_count"
    BATCH_COMPLETED_COUNT = "agentic.batch.completed_count"
    BATCH_FAILED_COUNT = "agentic.batch.failed_count"
    BATCH_INPUT_FILE_ID = "agentic.batch.input_file_id"
    BATCH_OUTPUT_FILE_ID = "agentic.batch.output_file_id"
    BATCH_ERROR_FILE_ID = "agentic.batch.error_file_id"
    BATCH_CREATED_AT = "agentic.batch.created_at"
    BATCH_COMPLETED_AT = "agentic.batch.completed_at"


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
    LITELLM = "litellm"
