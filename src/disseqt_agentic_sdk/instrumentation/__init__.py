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

from disseqt_agentic_sdk.instrumentation.auto import (
    AVAILABLE_INSTRUMENTORS,
    instrument,
    instrument_all,
    uninstrument,
    uninstrument_all,
)
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor

__all__ = [
    "DisseqtInstrumentor",
    "instrument",
    "instrument_all",
    "uninstrument",
    "uninstrument_all",
    "AVAILABLE_INSTRUMENTORS",
]
