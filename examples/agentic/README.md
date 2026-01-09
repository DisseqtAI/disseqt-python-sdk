# Disseqt Agentic SDK - Usage Examples

## Quick Start

### 1. Initialize SDK Client
```python
from disseqt_agentic_sdk import DisseqtAgenticClient

client = DisseqtAgenticClient(
    api_key="your-api-key",
    org_id="org_123",
    project_id="proj_456",
    service_name="my-service",
    endpoint="http://localhost:8080/v1/traces"
)
```

### 2. Create Traces and Spans
```python
from disseqt_agentic_sdk import start_trace
from disseqt_agentic_sdk.api.helpers import trace_llm_call, trace_agent_action, trace_tool_call
from disseqt_agentic_sdk.enums import SpanKind

with start_trace(client, name="my_workflow") as trace:
    # LLM call
    span = trace_llm_call(
        trace,
        name="chat",
        model_name="gpt-4",
        provider="openai",
        input_tokens=100,
        output_tokens=50
    )
    span.end()  # End span manually if not using 'with'

    # Agent action
    span = trace_agent_action(
        trace,
        name="planning",
        agent_name="assistant"
    )
    span.end()

    # Tool call
    span = trace_tool_call(
        trace,
        name="api_call",
        tool_name="get_weather"
    )
    span.end()

# Cleanup (optional - automatically called on exit)
client.shutdown()
```

## Complete Examples

- `example.py` - Complete example with multiple spans using `with` statements
- `example_without_with.py` - Manual span management without `with` statements
- `ai_consultant_agent.py` - Full AI agent integration example

## API Reference

### Client Initialization
```python
client = DisseqtAgenticClient(
    api_key: str,              # Required: API key for authentication
    org_id: str,               # Required: Organization ID
    project_id: str,            # Required: Project ID
    service_name: str,          # Required: Service name
    endpoint: str,              # Required: Backend endpoint URL
    service_version: str = "1.0.0",
    environment: str = "production",
    user_id: str = "",
    max_batch_size: int = 100,
    flush_interval: float = 1.0,
    timeout: float = 10.0,
    max_retries: int = 3,
    verify_ssl: bool = True
)
```

### Creating Traces
```python
from disseqt_agentic_sdk import start_trace

with start_trace(
    client: DisseqtAgenticClient,  # Required: Client instance
    name: str,                     # Required: Trace name
    trace_id: Optional[str] = None,
    intent_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None
) as trace:
    # Create spans here
    # Trace is automatically sent when exiting the 'with' block
```

### Helper Functions
Helper functions create spans and return them. Use `with` statements or call `.end()` manually:

```python
from disseqt_agentic_sdk.api.helpers import (
    trace_llm_call,
    trace_agent_action,
    trace_tool_call,
    trace_function
)

# LLM call span
span = trace_llm_call(
    trace,                    # DisseqtTrace instance
    name: str,
    model_name: str,
    provider: str,
    input_messages: Optional[List[Dict]] = None,
    output_messages: Optional[List[Dict]] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
)
span.end()  # Must end manually if not using 'with'

# Agent action span
span = trace_agent_action(
    trace,                    # DisseqtTrace instance
    name: str,
    agent_name: str,
    agent_id: Optional[str] = None,
    agent_version: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs
)
span.end()

# Tool call span
span = trace_tool_call(
    trace,                    # DisseqtTrace instance
    name: str,
    tool_name: str,
    tool_call_id: Optional[str] = None,
    tool_definitions: Optional[List[Dict]] = None,
    **kwargs
)
span.end()

# Function decorator (requires client)
@trace_function(client, name="my_function", kind=SpanKind.INTERNAL)
def my_function():
    pass
```

### Manual Span Creation
```python
from disseqt_agentic_sdk.enums import SpanKind

# Using 'with' statement (recommended)
with trace.start_span("operation", SpanKind.INTERNAL) as span:
    span.set_attribute("key", "value")
    # Span automatically ends when exiting 'with' block

# Manual creation (without 'with')
span = trace.start_span("operation", SpanKind.INTERNAL)
span.set_attribute("key", "value")
span.end()  # Must manually end
```

### Client Methods
```python
# Flush buffered spans immediately
client.flush()

# Shutdown client (flushes remaining spans)
client.shutdown()  # Automatically called on program exit via atexit
```

## Key Concepts

### Parent-Child Relationships
Spans automatically establish parent-child relationships when nested:
```python
with start_trace(client, "workflow") as trace:
    with trace.start_span("parent", SpanKind.AGENT_EXEC) as parent:
        # parent.span_id is automatically set

        with trace.start_span("child", SpanKind.MODEL_EXEC) as child:
            # child.parent_span_id = parent.span_id (automatic!)
            pass
```

### Automatic Timing
- **Start time**: Set when entering `with` block
- **End time**: Set when exiting `with` block
- **Duration**: Automatically calculated

### Buffering and Batching
- Spans are buffered and sent in batches
- Automatic flushing when batch size is reached
- Automatic flushing at intervals
- Manual flush via `client.flush()`
