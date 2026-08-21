"""
OpenAI SDK instrumentor.

Patches openai v1.x resource methods:
  * chat.completions.create           (sync + async, streaming + non-streaming)
  * embeddings.create                 (sync + async)
  * completions.create (legacy)       (sync + async, streaming + non-streaming)
  * batches.create / retrieve / cancel (sync + async)
"""

from __future__ import annotations

from disseqt_agentic_sdk.instrumentation.base import DisseqtInstrumentor
from disseqt_agentic_sdk.instrumentation.openai import batches_patch, patch


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

        # Batch API — sync + async, three lifecycle methods each.
        self._wrap(
            "openai.resources.batches",
            "Batches.create",
            batches_patch.batches_create(self),
        )
        self._wrap(
            "openai.resources.batches",
            "Batches.retrieve",
            batches_patch.batches_retrieve(self),
        )
        self._wrap(
            "openai.resources.batches",
            "Batches.cancel",
            batches_patch.batches_cancel(self),
        )
        self._wrap(
            "openai.resources.batches",
            "AsyncBatches.create",
            batches_patch.async_batches_create(self),
        )
        self._wrap(
            "openai.resources.batches",
            "AsyncBatches.retrieve",
            batches_patch.async_batches_retrieve(self),
        )
        self._wrap(
            "openai.resources.batches",
            "AsyncBatches.cancel",
            batches_patch.async_batches_cancel(self),
        )
