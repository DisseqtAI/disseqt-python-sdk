"""
Auto-instrumentation for popular LLM SDKs and agent frameworks.

Usage:

    from disseqt_agentic_sdk import DisseqtAgenticClient
    from disseqt_agentic_sdk.instrumentation import instrument_all

    client = DisseqtAgenticClient(api_key=..., project_id=..., service_name="my-app")
    instrument_all(client)

    # From here on, every openai.chat.completions.create() / anthropic
    # Messages.create() / etc. call emits a DisseqtSpan automatically,
    # parented to any active trace/span in the current thread.
"""

from disseqt_agentic_sdk.instrumentation._custom_attrs import (
    clear_span_attributes,
    set_span_attributes,
    span_context,
)
from disseqt_agentic_sdk.instrumentation._tool_result import (
    agent_span,
    record_tool_result,
)
from disseqt_agentic_sdk.instrumentation._utils import (
    get_capture_content,
    get_slow_call_threshold_ms,
    set_capture_content,
    set_slow_call_threshold_ms,
)
from disseqt_agentic_sdk.instrumentation.auto import (
    AVAILABLE_INSTRUMENTORS,
    get_instrumented_client,
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
)
from disseqt_agentic_sdk.instrumentation.base import (
    DisseqtInstrumentor,
    InstrumentationError,
)

__all__ = [
    "DisseqtInstrumentor",
    "InstrumentationError",
    "instrument",
    "instrument_all",
    "uninstrument",
    "uninstrument_all",
    "get_instrumented_client",
    "get_slow_call_threshold_ms",
    "set_slow_call_threshold_ms",
    "get_capture_content",
    "set_capture_content",
    "set_span_attributes",
    "clear_span_attributes",
    "span_context",
    "agent_span",
    "record_tool_result",
    "AVAILABLE_INSTRUMENTORS",
]
