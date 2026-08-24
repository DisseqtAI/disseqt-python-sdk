"""
LiteLLM instrumentor.

LiteLLM is a router / proxy over many LLM providers, but exposes a single
OpenAI-shape surface: `litellm.completion(model=..., messages=...)` and
`litellm.acompletion(...)`. Response is a ModelResponse mimicking
`openai.types.chat.ChatCompletion`, so `_oai_compat` handles it directly.

Provider labelling: we tag the span with `agentic.provider.name="litellm"`
and `gen_ai.system="litellm"`; the downstream provider is derivable from
`gen_ai.response.model` (e.g. "gpt-4o-mini" or "claude-3-5-haiku-latest").

Note: `litellm.completion` is defined in `litellm.main`. Patch there rather
than on the `litellm` module attribute so wrapt sees the underlying function.
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


class LiteLLMInstrumentor(DisseqtInstrumentor):
    package_name = "litellm"
    min_version = "1.40.0"

    def _instrument(self) -> None:
        # litellm re-exports completion/acompletion from litellm.main; the
        # top-level module binding captures the original function at import
        # time. We wrap the module attribute itself so `litellm.completion(...)`
        # invokes our wrapper.
        sync_wrapper, async_wrapper = make_openai_shape_chat_wrappers(
            self,
            sync_span_name="litellm.completion",
            async_span_name="litellm.acompletion",
            provider=AgenticProvider.LITELLM,
            system=GenAISystem.LITELLM,
            operation_agentic=AgenticOperation.CHAT,
            operation_gen_ai=GenAIOperation.CHAT,
        )
        self._wrap("litellm", "completion", sync_wrapper)
        self._wrap("litellm", "acompletion", async_wrapper)
