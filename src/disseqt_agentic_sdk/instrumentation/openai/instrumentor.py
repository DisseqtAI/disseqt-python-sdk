"""
OpenAI SDK instrumentor.

Patches openai v1.x resource methods:
  * chat.completions.create           (sync + async, streaming + non-streaming)
  * embeddings.create                 (sync + async)
  * completions.create (legacy)       (sync + async, streaming + non-streaming)
"""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.instrumentation.openai import patch


class OpenAIInstrumentor(DisseqtInstrumentor):
    package_name = "openai"
    min_version = "1.50.0"

    def _instrument(self) -> None:
        # Chat completions — sync + async
        self._wrap(
            "openai.resources.chat.completions",
            "Completions.create",
            patch.chat_completions_create(self),
        )
        self._wrap(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            patch.async_chat_completions_create(self),
        )

        # Embeddings — sync + async
        self._wrap(
            "openai.resources.embeddings",
            "Embeddings.create",
            patch.embeddings_create(self),
        )
        self._wrap(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            patch.async_embeddings_create(self),
        )

        # Legacy completions — sync + async
        self._wrap(
            "openai.resources.completions",
            "Completions.create",
            patch.completions_create(self),
        )
        self._wrap(
            "openai.resources.completions",
            "AsyncCompletions.create",
            patch.async_completions_create(self),
        )
