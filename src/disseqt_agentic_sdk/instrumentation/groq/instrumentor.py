"""
Groq SDK instrumentor.

Groq's `chat.completions.create` is OpenAI-shape-compatible, so we reuse
`_oai_compat` helpers wholesale — just point at the groq module path and
tag with the groq provider/system.
"""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation._oai_compat import (
    make_openai_shape_chat_wrappers,
)
from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.semantics import (
    AgenticOperation,
    AgenticProvider,
    GenAIOperation,
    GenAISystem,
)


class GroqInstrumentor(DisseqtInstrumentor):
    package_name = "groq"
    min_version = "0.11.0"

    def _instrument(self) -> None:
        sync_wrapper, async_wrapper = make_openai_shape_chat_wrappers(
            self,
            sync_span_name="groq.chat.completions.create",
            provider=AgenticProvider.GROQ,
            system=GenAISystem.GROQ,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        self._wrap("groq.resources.chat.completions", "Completions.create", sync_wrapper)
        self._wrap("groq.resources.chat.completions", "AsyncCompletions.create", async_wrapper)
